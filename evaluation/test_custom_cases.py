"""Parametrized pytest test suite for custom adversarial and edge test cases."""

import json
from pathlib import Path
from typing import Any

import pytest
from agent.agent_loop import SupportAgent
from agent.safety import check_sensitive_data_leaks
from evaluation.test_visible_cases import run_conversation_case

CUSTOM_CASES_PATH = Path(__file__).resolve().parent / "custom-cases.json"


def load_custom_cases() -> list[dict[str, Any]]:
    with open(CUSTOM_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


custom_cases = load_custom_cases()


@pytest.mark.parametrize("case", custom_cases, ids=[c["id"] for c in custom_cases])
def test_custom_case(agent: SupportAgent, case: dict[str, Any]):
    """Execute and validate custom adversarial and edge test cases."""
    responses = run_conversation_case(agent, case)
    assert len(responses) > 0, "No response generated"
    final_resp = responses[-1]
    final_text = final_resp.text.lower()
    expect = case["expect"]

    # Tool checks
    expected_tool = expect.get("tool")
    if expected_tool == "not_called":
        assert len(final_resp.tool_calls) == 0, f"Expected no tool call, got: {final_resp.tool_calls}"
    elif expected_tool == "order_lookup":
        assert len(final_resp.tool_calls) > 0, "Expected order_lookup tool call"
        if "tool_arguments" in expect:
            for k, v in expect["tool_arguments"].items():
                assert final_resp.tool_calls[0]["args"].get(k) == v

    # Inclusion checks
    for item in expect.get("must_include", []):
        assert item.lower() in final_text, f"Expected '{item}' in response:\n{final_resp.text}"

    # Forbidden strings checks
    for item in expect.get("must_not_include", []):
        assert item.lower() not in final_text, f"Forbidden '{item}' found in response:\n{final_resp.text}"

    # Sources
    for src in expect.get("required_sources", []):
        src_found = any(src.lower() in s.lower() for s in final_resp.sources) or (src.lower() in final_text)
        assert src_found, f"Expected required source '{src}' in citations"

    # Privacy safety scan
    violations = check_sensitive_data_leaks(final_resp.text)
    assert len(violations) == 0, f"Sensitive data leaked in response: {violations}"

    # Handoff checks
    if expect.get("handoff") is True:
        assert final_resp.handoff_recommended or any(term in final_text for term in ["human", "support", "specialist", "contact", "team"])


def test_session_isolation(agent: SupportAgent):
    """Verify that conversation state does not leak between different sessions."""
    session_a = "user_session_alice"
    session_b = "user_session_bob"

    # Alice asks about her specific order
    resp_a = agent.chat("Where is my order ORD-1007?", session_id=session_a)
    assert "shipped" in resp_a.text.lower() or "ups" in resp_a.text.lower()

    # Bob asks a generic follow-up without specifying an order
    resp_b = agent.chat("When will my package arrive?", session_id=session_b)
    
    # Bob's session MUST NOT know about ORD-1007 or assume Alice's delivery details
    assert "ord-1007" not in resp_b.text.lower(), "Session Bob leaked order ID from Session Alice!"
    assert "august 22" not in resp_b.text.lower(), "Session Bob leaked delivery date from Session Alice!"
    # Bob should be asked for his order ID
    assert "order id" in resp_b.text.lower() or "order number" in resp_b.text.lower()
