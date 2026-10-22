"""Fail-closed defaults for the legacy Socket.IO controller.

This module intentionally does not authenticate remote users. Legacy serial
commands are permitted only when explicitly enabled and issued from loopback.
Remote control requires a separately reviewed authentication design.
"""
from __future__ import annotations

import ipaddress
import os
import threading
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

NUM_DRONES = 2
MAX_SAFE_COUNTER = (2**53) - 1
PID_PARAMETER_COUNT = 17
TRIM_PARAMETER_COUNT = 4
MAX_ABS_PID_PARAMETER = 1_000.0
MAX_ABS_SETPOINT_COORDINATE = 100.0
MAX_ABS_TRIM_PARAMETER = 1_000


def _numeric_value(value: object, *, label: str, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be numeric")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    import math
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if integer:
        if not parsed.is_integer():
            raise ValueError(f"{label} must be an integer")
        return int(parsed)
    return parsed


def _drone_index(value: object) -> int:
    if type(value) is not int or not 0 <= value < NUM_DRONES:
        raise ValueError("droneIndex must be an integer in the configured drone range")
    return value


def validate_arm_payload(data: object) -> tuple[list[bool], int, int]:
    if not isinstance(data, dict):
        raise ValueError("arm payload must be an object")
    if set(data) != {"droneArmed", "count", "currentDroneIndex"}:
        raise ValueError("arm payload has missing or unknown fields")
    armed = data["droneArmed"]
    if not isinstance(armed, list) or len(armed) != NUM_DRONES or any(type(v) is not bool for v in armed):
        raise ValueError(f"droneArmed must contain exactly {NUM_DRONES} booleans")
    count = data["count"]
    if type(count) is not int or not 0 <= count <= MAX_SAFE_COUNTER:
        raise ValueError("count must be a nonnegative safe integer")
    current_index = _drone_index(data["currentDroneIndex"])
    return list(armed), count, current_index


def validate_pid_payload(data: object) -> tuple[int, list[float]]:
    if not isinstance(data, dict) or set(data) != {"dronePID", "droneIndex"}:
        raise ValueError("PID payload has missing or unknown fields")
    index = _drone_index(data["droneIndex"])
    raw = data["dronePID"]
    if not isinstance(raw, (list, tuple)) or len(raw) != PID_PARAMETER_COUNT:
        raise ValueError(f"dronePID must contain exactly {PID_PARAMETER_COUNT} values")
    values = [_numeric_value(v, label="PID parameter") for v in raw]
    if any(abs(v) > MAX_ABS_PID_PARAMETER for v in values):
        raise ValueError("PID parameter exceeds configured numeric bounds")
    return index, values


def validate_setpoint_payload(data: object) -> tuple[int, list[float]]:
    if not isinstance(data, dict) or set(data) != {"droneSetpoint", "droneIndex"}:
        raise ValueError("setpoint payload has missing or unknown fields")
    index = _drone_index(data["droneIndex"])
    raw = data["droneSetpoint"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("droneSetpoint must contain exactly 3 coordinates")
    values = [_numeric_value(v, label="setpoint coordinate") for v in raw]
    if any(abs(v) > MAX_ABS_SETPOINT_COORDINATE for v in values):
        raise ValueError("setpoint coordinate exceeds configured numeric bounds")
    return index, values


def validate_trim_payload(data: object) -> tuple[int, list[int]]:
    if not isinstance(data, dict) or set(data) != {"droneTrim", "droneIndex"}:
        raise ValueError("trim payload has missing or unknown fields")
    index = _drone_index(data["droneIndex"])
    raw = data["droneTrim"]
    if not isinstance(raw, (list, tuple)) or len(raw) != TRIM_PARAMETER_COUNT:
        raise ValueError(f"droneTrim must contain exactly {TRIM_PARAMETER_COUNT} values")
    values = [_numeric_value(v, label="trim parameter", integer=True) for v in raw]
    if any(abs(v) > MAX_ABS_TRIM_PARAMETER for v in values):
        raise ValueError("trim parameter exceeds configured numeric bounds")
    return index, values

class ArmCounterRegistry:
    """Bounded per-connection monotonic counters to reject replayed arm events."""

    def __init__(self, max_clients: int = 64) -> None:
        if type(max_clients) is not int or max_clients < 1:
            raise ValueError("max_clients must be a positive integer")
        self._max_clients = max_clients
        self._last_by_client: dict[str, int] = {}
        self._lock = threading.Lock()

    def accept(self, client_id: object, count: object) -> bool:
        if not isinstance(client_id, str) or not client_id:
            return False
        if type(count) is not int or not 0 <= count <= MAX_SAFE_COUNTER:
            return False
        with self._lock:
            previous = self._last_by_client.get(client_id)
            if previous is None:
                if len(self._last_by_client) >= self._max_clients:
                    return False
                self._last_by_client[client_id] = count
                return True
            if count <= previous:
                return False
            self._last_by_client[client_id] = count
            return True

    def discard(self, client_id: object) -> None:
        if not isinstance(client_id, str) or not client_id:
            return
        with self._lock:
            self._last_by_client.pop(client_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._last_by_client)
