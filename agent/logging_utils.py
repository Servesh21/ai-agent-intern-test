"""Structured JSON logging and observability for the Aster & Row Support Agent."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from agent.config import DEBUG

# Configure logger
logger = logging.getLogger("aster_row_agent")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Formatter that writes single-line structured JSON logs
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger.addHandler(handler)


def sanitize_log_dict(data: Any) -> Any:
    """Recursively scrub sensitive keys (api_key, email, address, internal notes) from log objects."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(secret_term in k_lower for secret_term in ["api_key", "secret", "password", "token"]):
                cleaned[k] = "[REDACTED_SECRET]"
            elif k_lower in ("email", "shipping_address", "address", "risk_score", "warehouse_note"):
                cleaned[k] = "[REDACTED_PII]"
            else:
                cleaned[k] = sanitize_log_dict(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_log_dict(item) for item in data]
    return data


def log_turn(
    session_id: str,
    turn_index: int,
    user_message: str,
    retrieved_chunks: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    final_response: str,
    handoff_recommended: bool,
    conflict_detected: bool,
    latency_ms: float | None = None,
) -> None:
    """Emit a single structured JSON log line for an agent conversation turn."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "turn": turn_index,
        "user_message": user_message,
        "retrieval": {
            "num_chunks": len(retrieved_chunks),
            "conflict_detected": conflict_detected,
            "chunks": [
                {
                    "filename": c.get("filename"),
                    "heading_path": c.get("heading_path"),
                    "score": round(c.get("score", 0.0), 4) if isinstance(c.get("score"), (int, float)) else None,
                    "status": c.get("status"),
                }
                for c in retrieved_chunks
            ],
        },
        "tool_calls": [
            {
                "name": tc.get("name"),
                "args": sanitize_log_dict(tc.get("args", {})),
                "success": tc.get("result", {}).get("success", True) if isinstance(tc.get("result"), dict) else True,
            }
            for tc in tool_calls
        ],
        "handoff_recommended": handoff_recommended,
        "response_preview": final_response[:200] + "..." if len(final_response) > 200 else final_response,
    }

    if latency_ms is not None:
        log_entry["latency_ms"] = round(latency_ms, 2)

    if DEBUG:
        # In debug mode, include full response text
        log_entry["full_response"] = final_response

    # Output as single JSON line
    json_line = json.dumps(sanitize_log_dict(log_entry))
    logger.info(json_line)
