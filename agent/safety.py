"""Safety guardrails and post-generation verification."""

import re
from typing import Any

# Forbidden private patterns from mock orders
KNOWN_PRIVATE_EMAILS = [
    r"[a-zA-Z0-9_.+-]+@example\.test",
    r"maya\.reed@",
    r"noah\.kim@",
    r"olivia\.chen@",
    r"ethan\.brooks@",
    r"sofia\.patel@",
    r"liam\.jones@",
    r"ava\.morgan@",
    r"lucas\.green@",
    r"isabella\.stone@",
    r"henry\.diaz@",
    r"emma\.wilson@",
    r"james\.taylor@",
]

KNOWN_PRIVATE_ADDRESSES = [
    r"18 Cedar Lane",
    r"44 Lake Street",
    r"79 Market Street",
    r"12 Harbor Road",
    r"96 Peachtree Avenue",
    r"55 Congress Avenue",
    r"220 King Street",
    r"310 Pine Street",
    r"7 Ocean Drive",
    r"801 Larimer Street",
    r"1010 Robson Street",
    r"400 Walnut Street",
]

KNOWN_INTERNAL_STRINGS = [
    r"risk_score",
    r"risk score",
    r"fraud review cleared",
    r"Manual fraud review",
    r"Refund batch REF-7781",
    r"issue a \$100 coupon",
    r"Carrier ETA feed unavailable",
    r"Payment verification completed",
    r"Package damage scan; carrier case pending",
    r"Picking complete; label not printed",
]

# Action completion claims that are forbidden because the system is read-only
FORBIDDEN_ACTION_CLAIMS = [
    r"(?:i have|i've|we have|we've)\s+(?:cancelled|canceled)\s+your order",
    r"your order\s+(?:has been|is now)\s+(?:cancelled|canceled)\s+by me",
    r"(?:i have|i've|we have|we've)\s+(?:processed|issued|approved)\s+your refund",
    r"your refund\s+(?:has been|is now)\s+(?:processed|issued|approved)",
    r"(?:i have|i've|we have|we've)\s+(?:updated|changed)\s+your (?:shipping )?address",
    r"your address\s+(?:has been|is now)\s+(?:updated|changed)",
]

HANDOFF_INDICATORS = [
    r"human support",
    r"human specialist",
    r"support specialist",
    r"contact support",
    r"contact our support team",
    r"reach out to support",
    r"escalat",
    r"customer service team",
    r"human assistance",
    r"support agent will need to",
    r"specialist reviews",
    r"human review",
    r"human confirmation",
]


def check_sensitive_data_leaks(text: str) -> list[str]:
    """Scan text for leaks of sensitive customer or internal data.
    
    Returns a list of violation descriptions.
    """
    violations = []

    for pattern in KNOWN_PRIVATE_EMAILS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Exposed customer email pattern: {pattern}")

    for pattern in KNOWN_PRIVATE_ADDRESSES:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Exposed customer address pattern: {pattern}")

    for pattern in KNOWN_INTERNAL_STRINGS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Exposed internal operational note/field: {pattern}")

    return violations


def check_action_hallucinations(text: str) -> list[str]:
    """Scan text for false claims of completed actions."""
    violations = []
    for pattern in FORBIDDEN_ACTION_CLAIMS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Claimed unperformed action completion: {pattern}")
    return violations


def sanitize_response_text(text: str) -> str:
    """Post-process response to ensure no sensitive tokens slip through."""
    sanitized = text
    for pattern in KNOWN_PRIVATE_EMAILS:
        sanitized = re.sub(pattern, "[REDACTED EMAIL]", sanitized, flags=re.IGNORECASE)
    for pattern in KNOWN_PRIVATE_ADDRESSES:
        sanitized = re.sub(pattern, "[REDACTED ADDRESS]", sanitized, flags=re.IGNORECASE)
    return sanitized


def detect_handoff_signal(text: str) -> bool:
    """Check if the text recommends human handoff or human support assistance."""
    for pattern in HANDOFF_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
