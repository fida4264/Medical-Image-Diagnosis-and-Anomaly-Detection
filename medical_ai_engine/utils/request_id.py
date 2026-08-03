"""
Medical AI Engine — Request ID Generator

Generates thread-safe, UUID4-based request identifiers with optional custom prefixes.
"""

from __future__ import annotations

import uuid


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a unique request identifier string.

    Args:
        prefix: Short string prefix (default: "req").

    Returns:
        String format: "{prefix}_{short_uuid}" (e.g. "req_c4f9a1b2e3d4")
    """
    raw_uuid = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw_uuid}"


def generate_session_id(request_id: str) -> str:
    """
    Generate a session identifier correlated with a request ID.
    """
    return f"sess_{request_id}"
