"""Order lookup tool: load, validate, sanitize, and return customer-safe order data."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import ORDERS_FILE

# --- Fields that are NEVER exposed to the customer or model ---
INTERNAL_FIELDS = {"customer", "internal"}
# customer.name, customer.email, customer.shipping_address
# internal.risk_score, internal.warehouse_note, internal.support_tags

# --- Statuses where delivery fields are stale ---
TERMINAL_STATUSES = {"cancelled", "returned"}

# --- Valid order ID pattern ---
ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")

# --- Snapshot time (from orders.json, used for cancellation window checks) ---
SNAPSHOT_TIME: datetime | None = None


class OrderStore:
    """In-memory order store loaded from orders.json."""

    def __init__(self, orders_file: Path | None = None):
        self._orders: dict[str, dict[str, Any]] = {}
        self._snapshot_at: datetime | None = None
        self._load(orders_file or ORDERS_FILE)

    def _load(self, filepath: Path) -> None:
        """Load orders from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._snapshot_at = datetime.fromisoformat(
            data.get("snapshot_at", "2026-08-15T12:00:00Z").replace("Z", "+00:00")
        )

        for order in data.get("orders", []):
            order_id = order["order_id"]
            self._orders[order_id] = order

    @property
    def snapshot_at(self) -> datetime:
        return self._snapshot_at

    def lookup(self, raw_order_id: str) -> dict[str, Any]:
        """Look up an order by ID, returning sanitized customer-safe data.

        This method:
        1. Normalizes the input (strip, uppercase, remove quotes)
        2. Validates the format
        3. Looks up the order
        4. Strips all internal/sensitive fields
        5. Suppresses stale delivery fields for terminal statuses
        6. Annotates missing estimates

        Returns a dict with either:
            {"success": True, "order": {...sanitized...}}
            {"success": False, "error": "...", "error_type": "not_found"|"invalid_format"}
        """
        # Step 1: Normalize
        normalized = self._normalize_order_id(raw_order_id)

        # Step 2: Validate format
        if not ORDER_ID_PATTERN.match(normalized):
            return {
                "success": False,
                "error": f"'{raw_order_id}' is not a valid order ID format. "
                         f"Order IDs look like ORD-1001.",
                "error_type": "invalid_format",
            }

        # Step 3: Lookup
        order = self._orders.get(normalized)
        if order is None:
            return {
                "success": False,
                "error": f"No order found with ID '{normalized}'. "
                         f"Please double-check the order ID or contact support for help.",
                "error_type": "not_found",
            }

        # Step 4: Sanitize — build customer-safe response
        sanitized = self._sanitize(order)

        return {"success": True, "order": sanitized}

    def _normalize_order_id(self, raw: str) -> str:
        """Normalize an order ID: strip whitespace, remove quotes, uppercase."""
        cleaned = raw.strip().strip("'\"").strip()
        return cleaned.upper()

    def _sanitize(self, order: dict[str, Any]) -> dict[str, Any]:
        """Remove internal fields and suppress stale delivery data.

        Returns ONLY customer-safe fields as specified in the data dictionary.
        """
        status = order.get("status", "")

        # Build customer-safe output
        safe: dict[str, Any] = {
            "order_id": order["order_id"],
            "membership_tier": order.get("membership_tier"),
            "items": [
                {
                    "name": item["name"],
                    "quantity": item["quantity"],
                    "final_sale": item.get("final_sale", False),
                }
                for item in order.get("items", [])
            ],
            "placed_at": order.get("placed_at"),
            "status": status,
            "status_updated_at": order.get("status_updated_at"),
        }

        # Add shipping fields — but suppress stale ones for terminal statuses
        if status in TERMINAL_STATUSES:
            # Order is cancelled or returned — delivery fields are stale
            safe["carrier"] = None
            safe["tracking_number"] = None
            safe["estimated_delivery"] = None
            safe["shipped_at"] = None
            safe["delivered_at"] = order.get("delivered_at")  # Keep for returned (was delivered)
            if status == "cancelled":
                safe["customer_safe_message"] = (
                    "This order has been cancelled and will not be shipped."
                )
                safe["delivered_at"] = None
            elif status == "returned":
                safe["customer_safe_message"] = (
                    "This order has been returned and processed."
                )
        else:
            safe["shipped_at"] = order.get("shipped_at")
            safe["delivered_at"] = order.get("delivered_at")
            safe["carrier"] = order.get("carrier")
            safe["tracking_number"] = order.get("tracking_number")
            safe["estimated_delivery"] = order.get("estimated_delivery")
            safe["customer_safe_message"] = order.get("customer_safe_message")

        # Annotate missing delivery estimate
        if (
            status == "shipped"
            and safe.get("estimated_delivery") is None
        ):
            safe["delivery_estimate_note"] = (
                "A delivery estimate is not currently available for this shipment."
            )

        # Annotate exception status
        if status == "exception":
            safe["exception_note"] = (
                "This shipment has an exception that requires support review. "
                "Please contact a human support specialist for assistance."
            )

        return safe


# Singleton instance
_store: OrderStore | None = None


def get_order_store() -> OrderStore:
    """Get or create the singleton OrderStore."""
    global _store
    if _store is None:
        _store = OrderStore()
    return _store


def lookup_order(order_id: str) -> dict[str, Any]:
    """Public API for order lookup. Used as the Gemini function-calling tool implementation.

    Args:
        order_id: The order ID to look up (e.g., "ORD-1007")

    Returns:
        Sanitized order data or error information.
    """
    store = get_order_store()
    return store.lookup(order_id)
