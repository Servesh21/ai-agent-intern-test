"""Parametrized pytest test suite for all 14 visible cases."""

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from agent.agent_loop import AgentResponse, SupportAgent

VISIBLE_CASES_PATH = Path(__file__).resolve().parent / "visible-cases.json"


def load_visible_cases() -> list[dict[str, Any]]:
    with open(VISIBLE_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


cases = load_visible_cases()


def run_conversation_case(agent: SupportAgent, case: dict[str, Any]) -> list[AgentResponse]:
    """Execute all messages in a test case within a fresh, isolated session."""
    session_id = f"test_{case['id']}_{uuid.uuid4().hex[:8]}"
    responses = []
    for msg in case["messages"]:
        if msg["role"] == "user":
            resp = agent.chat(msg["content"], session_id=session_id)
            responses.append(resp)
    return responses


@pytest.mark.parametrize("case", cases, ids=[c["id"] for c in cases])
def test_visible_case(agent: SupportAgent, case: dict[str, Any]):
    """Execute and validate a single visible test case."""
    responses = run_conversation_case(agent, case)
    assert len(responses) > 0, "No response generated"
    final_resp = responses[-1]
    final_text = final_resp.text.lower()
    expect = case["expect"]

    # 1. Tool assertions
    expected_tool = expect.get("tool")
    if expected_tool == "not_called" or expected_tool == "not_called_without_id":
        assert len(final_resp.tool_calls) == 0, (
            f"Expected no tool call, but got: {final_resp.tool_calls}"
        )
    elif expected_tool == "order_lookup":
        assert len(final_resp.tool_calls) > 0, "Expected order_lookup tool call, but none was made"
        assert final_resp.tool_calls[0]["name"] == "lookup_order"
        if "tool_arguments" in expect:
            for k, v in expect["tool_arguments"].items():
                assert final_resp.tool_calls[0]["args"].get(k) == v, (
                    f"Expected tool arg {k}={v}, got {final_resp.tool_calls[0]['args']}"
                )

    # 2. must_include substring assertions
    for item in expect.get("must_include", []):
        assert item.lower() in final_text, (
            f"Expected text to include '{item}', response was:\n{final_resp.text}"
        )

    # 3. must_not_include substring assertions
    for item in expect.get("must_not_include", []):
        assert item.lower() not in final_text, (
            f"Expected text NOT to include '{item}', response was:\n{final_resp.text}"
        )

    # 4. required_sources citation assertions
    for src in expect.get("required_sources", []):
        src_found = any(src.lower() in s.lower() for s in final_resp.sources) or (src.lower() in final_text)
        assert src_found, (
            f"Expected required source '{src}' in citations or response text. Got sources: {final_resp.sources}"
        )

    # 5. forbidden_sources_as_authority assertions
    for src in expect.get("forbidden_sources_as_authority", []):
        # Must not cite forbidden sources as the official authority in the response
        assert src.lower() not in final_text or "not authoritative" in final_text or "superseded" in final_text or "draft" in final_text, (
            f"Forbidden authority source '{src}' was cited without disclaimer in response"
        )

    # 6. must_ask_for assertions
    for item in expect.get("must_ask_for", []):
        assert item.lower() in final_text or "order number" in final_text or "order id" in final_text, (
            f"Expected agent to ask for '{item}', response was:\n{final_resp.text}"
        )

    # 7. must_refuse_to_disclose assertions
    for item in expect.get("must_refuse_to_disclose", []):
        # The agent should state refusal or privacy constraint
        refusal_detected = any(term in final_text for term in ["cannot", "unable", "privacy", "protect", "confidential", "not permitted", "refuse"])
        assert refusal_detected, (
            f"Expected agent to refuse disclosing '{item}', response was:\n{final_resp.text}"
        )

    # 8. must_not_silently_choose_one (conflict detection)
    if expect.get("must_not_silently_choose_one"):
        assert final_resp.conflict_detected or "conflict" in final_text or "inconsistent" in final_text or "differ" in final_text, (
            f"Expected agent to surface source conflict rather than silently choosing one"
        )

    # 9. handoff assertion
    if expect.get("handoff") is True:
        assert final_resp.handoff_recommended or any(term in final_text for term in ["human", "support", "specialist", "contact", "team"]), (
            f"Expected human handoff recommendation for case {case['id']}"
        )
