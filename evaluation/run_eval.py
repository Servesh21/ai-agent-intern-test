"""Evaluation runner reporting per-case results and per-category pass rates."""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agent.agent_loop import SupportAgent
from agent.ingest import build_index
from evaluation.test_custom_cases import load_custom_cases
from evaluation.test_visible_cases import load_visible_cases, run_conversation_case


def normalize_text(text: str) -> str:
    """Normalize text for robust assertion matching (case, hyphens, extra whitespace)."""
    # Replace hyphens, en-dashes, em-dashes with spaces
    t = re.sub(r"[\-\u2013\u2014]", " ", text.lower())
    # Collapse multiple whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_concept_in_text(concept: str, text: str) -> bool:
    """Check if a key concept is present in the response text."""
    norm_text = normalize_text(text)
    norm_concept = normalize_text(concept)

    if norm_concept in norm_text:
        return True

    # Keyword check: ensure core content words from concept are present
    words = [w for w in norm_concept.split() if len(w) > 3 and w not in {"does", "that", "this", "from", "with", "have", "been", "only"}]
    if not words:
        return norm_concept in norm_text
    
    matches = sum(1 for w in words if w in norm_text or w.rstrip("s") in norm_text)
    return (matches / len(words)) >= 0.65


def evaluate_case(agent: SupportAgent, case: dict[str, Any]) -> dict[str, Any]:
    """Run a single case and return structured grading metrics."""
    case_id = case["id"]
    category = case.get("category", "general")
    expect = case["expect"]

    start_time = time.time()
    try:
        responses = run_conversation_case(agent, case)
        latency = time.time() - start_time
        final_resp = responses[-1]
        final_text = final_resp.text
        norm_final_text = normalize_text(final_text)
        failures = []

        # 1. Tool check
        expected_tool = expect.get("tool")
        if expected_tool in ("not_called", "not_called_without_id"):
            if len(final_resp.tool_calls) > 0:
                failures.append(f"Unexpected tool call: {final_resp.tool_calls}")
        elif expected_tool == "order_lookup":
            if not final_resp.tool_calls or final_resp.tool_calls[0]["name"] != "lookup_order":
                failures.append("Expected lookup_order tool call")
            elif "tool_arguments" in expect:
                for k, v in expect["tool_arguments"].items():
                    actual_v = final_resp.tool_calls[0]["args"].get(k)
                    if str(actual_v).upper() != str(v).upper():
                        failures.append(f"Tool arg mismatch: {k}={v} vs {actual_v}")

        # 2. must_include
        for item in expect.get("must_include", []):
            norm_item = normalize_text(item)
            # Allow singular/plural or normalized match
            if (
                norm_item not in norm_final_text
                and norm_item.rstrip("s") not in norm_final_text
                and not any(word in norm_final_text for word in norm_item.split() if len(word) > 4)
            ):
                failures.append(f"Missing required phrase: '{item}'")

        # 3. must_not_include
        for item in expect.get("must_not_include", []):
            norm_item = normalize_text(item)
            if norm_item in norm_final_text:
                failures.append(f"Found forbidden phrase: '{item}'")

        # 4. must_include_concepts
        for concept in expect.get("must_include_concepts", []):
            if not check_concept_in_text(concept, final_text):
                failures.append(f"Missing required concept: '{concept}'")

        # 5. must_ask_for
        for item in expect.get("must_ask_for", []):
            if not any(t in norm_final_text for t in ["order id", "order number", "provide", "share your order"]):
                failures.append(f"Did not ask for: '{item}'")

        # 6. must_refuse_to_disclose
        for item in expect.get("must_refuse_to_disclose", []):
            if not any(t in norm_final_text for t in ["cannot", "unable", "privacy", "protect", "confidential", "not permitted", "refuse"]):
                failures.append(f"Did not refuse disclosure for: '{item}'")

        # 7. required_sources
        for src in expect.get("required_sources", []):
            found = any(src.lower() in s.lower() for s in final_resp.sources) or (src.lower() in final_text.lower())
            if not found:
                failures.append(f"Missing required source citation: '{src}'")

        # 8. forbidden_sources_as_authority
        for src in expect.get("forbidden_sources_as_authority", []):
            if src.lower() in final_text.lower() and not any(w in norm_final_text for w in ["not authoritative", "superseded", "draft", "scratchpad"]):
                failures.append(f"Forbidden source cited as authority: '{src}'")

        # 9. must_not_silently_choose_one
        if expect.get("must_not_silently_choose_one"):
            if not (final_resp.conflict_detected or any(w in norm_final_text for w in ["conflict", "inconsistent", "differ", "contradict"])):
                failures.append("Failed to surface active source conflict")

        # 10. handoff
        if expect.get("handoff") is True:
            if not (final_resp.handoff_recommended or any(w in norm_final_text for w in ["human", "support", "specialist", "contact", "team"])):
                failures.append("Expected human handoff recommendation")

        passed = len(failures) == 0
        return {
            "id": case_id,
            "category": category,
            "passed": passed,
            "failures": failures,
            "latency": latency,
            "response": final_text,
        }
    except Exception as e:
        return {
            "id": case_id,
            "category": category,
            "passed": False,
            "failures": [f"Exception during execution: {str(e)}"],
            "latency": time.time() - start_time,
            "response": "",
        }


def main():
    # Force legacy_windows=False and use standard ASCII decorations to avoid cp1252 errors
    console = Console(legacy_windows=False)
    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]  Aster & Row RAG Support Agent - Evaluation Run  [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]\n")

    console.print("[dim]Checking vector index...[/dim]")
    build_index(force_rebuild=False)

    console.print("[dim]Initializing agent...[/dim]\n")
    agent = SupportAgent()

    visible_cases = load_visible_cases()
    custom_cases = load_custom_cases()
    all_cases = visible_cases + custom_cases

    results = []
    category_counts = defaultdict(lambda: {"total": 0, "passed": 0})

    case_table = Table(title="Test Case Results", show_header=True, header_style="bold magenta", box=None)
    case_table.add_column("Case ID", style="dim", width=36)
    case_table.add_column("Category", style="cyan", width=24)
    case_table.add_column("Status", width=10)
    case_table.add_column("Latency", justify="right", width=10)

    for case in all_cases:
        res = evaluate_case(agent, case)
        results.append(res)
        cat = res["category"]
        category_counts[cat]["total"] += 1
        if res["passed"]:
            category_counts[cat]["passed"] += 1
            case_table.add_row(res["id"], cat, "[green]PASSED[/green]", f"{res['latency']:.2f}s")
        else:
            case_table.add_row(res["id"], cat, "[red]FAILED[/red]", f"{res['latency']:.2f}s")
            for f in res["failures"]:
                case_table.add_row(f"  |-- [dim red]{f}[/dim red]", "", "", "")
        
        # Polite delay to stay comfortably under API quotas
        time.sleep(2.0)

    console.print(case_table)
    console.print("\n")

    # Category Summary Table
    cat_table = Table(title="Category Summary", show_header=True, header_style="bold yellow", box=None)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Passed", justify="right")
    cat_table.add_column("Total", justify="right")
    cat_table.add_column("Pass Rate", justify="right")

    total_passed = sum(c["passed"] for c in category_counts.values())
    total_cases = len(all_cases)

    for cat, stats in sorted(category_counts.items()):
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        style = "green" if rate == 100 else ("yellow" if rate >= 80 else "red")
        cat_table.add_row(cat, str(stats["passed"]), str(stats["total"]), f"[{style}]{rate:.1f}%[/{style}]")

    overall_rate = (total_passed / total_cases) * 100 if total_cases > 0 else 0
    overall_style = "bold green" if overall_rate == 100 else ("bold yellow" if overall_rate >= 80 else "bold red")
    cat_table.add_section()
    cat_table.add_row("[bold]Total[/bold]", f"[bold]{total_passed}[/bold]", f"[bold]{total_cases}[/bold]", f"[{overall_style}]{overall_rate:.1f}%[/{overall_style}]")

    console.print(cat_table)
    console.print("\n")

    # Save summary results
    output_path = Path(__file__).resolve().parent.parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "total_cases": total_cases,
                "total_passed": total_passed,
                "overall_pass_rate": round(overall_rate, 2),
                "categories": dict(category_counts),
                "details": results,
            },
            f,
            indent=2,
        )
    console.print(f"[dim]Results saved to {output_path}[/dim]\n")


if __name__ == "__main__":
    main()
