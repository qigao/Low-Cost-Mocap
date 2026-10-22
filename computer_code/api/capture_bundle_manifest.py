"""Strict capture-bundle manifest and artifact-integrity validator.

A valid report proves only that the supplied manifest and bytes satisfy this
software schema and its declared hashes/sizes. It does NOT authenticate the
devices, prove that a reference logger was independent, or verify that a
timestamp was captured at sensor exposure.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


MANIFEST_NAME = "capture-bundle.json"
SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1_000_000
DEFAULT_MAX_ARTIFACTS = 32
DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024**3
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024**3
CHUNK_BYTES = 1024 * 1024

_REQUIRED_ROLES = (
    "raw_camera_frames",
    "reference_trigger_log",
    "independent_marker_labels",
    "device_metadata",
    "led_controller_log",
    "codebook",
    "controller_config",
    "calibration_config",
)
_OPTIONAL_ROLES = {"supplemental"}
_TIMESTAMP_SEMANTICS = {
    "sensor_exposure", "hardware_trigger", "driver_dequeue",
    "host_monotonic", "unknown",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUNDLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.:/-]{0,127}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise ValueError(f"{name} fields invalid: {'; '.join(details)}")


def _integer(
    value: object, name: str, *,
    minimum: int = 0, maximum: int = (1 << 63) - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _version(value: object, name: str) -> str:
    result = _text(value, name, maximum=128)
    if not _VERSION.fullmatch(result) or result.lower() in {
        "unknown", "n/a", "na", "unavailable", "unspecified",
    }:
        raise ValueError(f"{name} must be a declared printable version")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _utc_datetime(value: object, name: str) -> datetime:
    raw = _text(value, name, maximum=40)
    try:
        parsed = datetime.fromisoformat(
            raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        )
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must include an explicit UTC timezone")
    return parsed


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("manifest is not finite canonical JSON") from exc


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def load_manifest_bytes(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest must be bytes no larger than {MAX_MANIFEST_BYTES}")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest must be valid UTF-8 JSON") from exc
    return _object(parsed, "manifest")
