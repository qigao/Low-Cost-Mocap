"""Fail-closed defaults for the legacy Socket.IO controller.

This module intentionally does not authenticate remote users. Legacy serial
commands are permitted only when explicitly enabled and issued from loopback.
Remote control requires a separately reviewed authentication design.
"""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit


DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def configured_origins(value: str | None) -> list[str]:
    """Return a strict explicit origin allowlist; wildcard origins are forbidden."""
    if value is None:
        return list(DEFAULT_ALLOWED_ORIGINS)
    origins = [item.strip() for item in value.split(",") if item.strip()]
    if not origins:
        raise ValueError("MOCAP_ALLOWED_ORIGINS must contain at least one origin")
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("MOCAP_ALLOWED_ORIGINS must contain explicit HTTP(S) origins without paths")
    return list(dict.fromkeys(origins))


def legacy_control_peer_allowed(remote_addr: str | None, enabled: str | None = None) -> bool:
    """Permit legacy serial-control events only with explicit opt-in from loopback."""
    setting = os.environ.get("MOCAP_ENABLE_LEGACY_FLIGHT_CONTROL") if enabled is None else enabled
    if setting != "1" or not remote_addr:
        return False
    try:
        return ipaddress.ip_address(remote_addr).is_loopback
    except ValueError:
        return False
