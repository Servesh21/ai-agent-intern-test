"""Deterministic unit tests for order sanitization, metadata parsing, safety filters, and session isolation."""

import pytest
from agent.orders import OrderStore, lookup_order
from agent.ingest import _chunk_by_heading, _parse_frontmatter
from agent.safety import (
    check_action_hallucinations,
    check_sensitive_data_leaks,
    detect_handoff_signal,
    sanitize_response_text,
)
from agent.session import SessionManager


# ==============================================================================
# Order Lookup & Sanitization Tests
# ==============================================================================

def test_order_id_normalization():
    """Verify case-insensitivity and whitespace stripping."""
    res1 = lookup_order("ord-1001")
    assert res1["success"] is True
    assert res1["order"]["order_id"] == "ORD-1001"

    res2 = lookup_order("  ORD-1002  ")
    assert res2["success"] is True
    assert res2["order"]["order_id"] == "ORD-1002"

    res3 = lookup_order("\"ord-1003\"")
    assert res3["success"] is True
    assert res3["order"]["order_id"] == "ORD-1003"


def test_malformed_and_unknown_order_ids():
    """Verify safety on invalid IDs."""
    res_bad = lookup_order("INVALID_ID_999")
    assert res_bad["success"] is False
    assert res_bad["error_type"] == "invalid_format"

    res_not_found = lookup_order("ORD-9999")
    assert res_not_found["success"] is False
    assert res_not_found["error_type"] == "not_found"


def test_internal_fields_completely_stripped():
    """Ensure customer email, shipping address, and internal warehouse notes/scores are never in returned order."""
    res = lookup_order("ORD-1007")
    assert res["success"] is True
    order = res["order"]

    # Forbidden root fields
    assert "internal" not in order
    assert "customer" not in order

    # Forbidden specific sensitive fields
    assert "risk_score" not in order
    assert "warehouse_note" not in order
    assert "support_tags" not in order
    assert "email" not in order
    assert "shipping_address" not in order
    assert "ava.morgan@example.test" not in str(order)
    assert "220 King Street" not in str(order)
    assert "82" not in str(order)  # Risk score 82


def test_cancelled_order_suppresses_stale_delivery_eta():
    """Cancelled order ORD-1004 had stale UPS/ETA in raw record; must be suppressed."""
    res = lookup_order("ORD-1004")
    assert res["success"] is True
    order = res["order"]
    assert order["status"] == "cancelled"
    assert order["carrier"] is None
    assert order["estimated_delivery"] is None
    assert order["tracking_number"] is None
    assert "cancelled" in order["customer_safe_message"].lower()


def test_returned_order_suppresses_stale_delivery_eta():
    """Returned order ORD-1008 must suppress active delivery tracking."""
    res = lookup_order("ORD-1008")
    assert res["success"] is True
    order = res["order"]
    assert order["status"] == "returned"
    assert order["carrier"] is None
    assert order["estimated_delivery"] is None
    assert order["tracking_number"] is None


def test_shipped_without_eta_annotated():
    """ORD-1011 is shipped with Canada Post but missing ETA; must annotate unavailable."""
    res = lookup_order("ORD-1011")
    assert res["success"] is True
    order = res["order"]
    assert order["status"] == "shipped"
    assert order["carrier"] == "Canada Post"
    assert order["estimated_delivery"] is None
    assert "delivery_estimate_note" in order


def test_order_exception_annotated():
    """ORD-1010 is exception; must have exception note."""
    res = lookup_order("ORD-1010")
    assert res["success"] is True
    order = res["order"]
    assert order["status"] == "exception"
    assert "exception_note" in order


# ==============================================================================
# Safety Scanner Tests
# ==============================================================================

def test_sensitive_data_leak_detector():
    """Test that customer PII and internal notes trigger leak detection."""
    clean_text = "Your order ORD-1001 is currently pending."
    assert len(check_sensitive_data_leaks(clean_text)) == 0

    leaky_email = "The customer email is maya.reed@example.test."
    assert len(check_sensitive_data_leaks(leaky_email)) > 0

    leaky_address = "Shipping to 18 Cedar Lane, Portland."
    assert len(check_sensitive_data_leaks(leaky_address)) > 0

    leaky_risk = "The internal risk score is 14."
    assert len(check_sensitive_data_leaks(leaky_risk)) > 0


def test_action_hallucination_detector():
    """Detect false claims of completed cancellations or refunds."""
    clean_text = "I cannot cancel your order directly; please contact support."
    assert len(check_action_hallucinations(clean_text)) == 0

    hallucinated_text = "I have cancelled your order ORD-1001 and processed your refund."
    assert len(check_action_hallucinations(hallucinated_text)) > 0


def test_handoff_signal_detector():
    """Detect human support handoff phrasing."""
    assert detect_handoff_signal("Please contact our human support specialist.") is True
    assert detect_handoff_signal("I recommend escalating this to our support team.") is True
    assert detect_handoff_signal("Your package is on its way with USPS.") is False


def test_response_sanitizer():
    """Ensure response sanitizer redacts PII."""
    raw = "Contact maya.reed@example.test at 18 Cedar Lane."
    sanitized = sanitize_response_text(raw)
    assert "maya.reed@example.test" not in sanitized
    assert "18 Cedar Lane" not in sanitized
    assert "[REDACTED" in sanitized


# ==============================================================================
# Session Isolation Tests
# ==============================================================================

def test_session_manager_isolation():
    """Ensure separate sessions have independent history."""
    mgr = SessionManager()
    sess1 = mgr.get_or_create("session_1")
    sess2 = mgr.get_or_create("session_2")

    sess1.add_user_message("My order is ORD-1001")
    sess1.add_model_message("ORD-1001 is pending")

    sess2.add_user_message("Do you sell tumblers?")

    assert len(sess1.history) == 2
    assert len(sess2.history) == 1
    assert "ORD-1001" not in str(sess2.history)
