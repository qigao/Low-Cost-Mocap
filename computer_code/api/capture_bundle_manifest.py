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



def _validate_structure(
    manifest: dict[str, Any],
    *,
    max_artifacts: int,
    max_artifact_bytes: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version", "bundle_id", "capture", "software",
            "revisions", "acceptance_limits", "artifacts",
        },
        "manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must equal integer 1")
    if not isinstance(manifest["bundle_id"], str) or not _BUNDLE_ID.fullmatch(manifest["bundle_id"]):
        raise ValueError("bundle_id must be a safe versioned identifier")

    capture = _object(manifest["capture"], "capture")
    _exact_keys(
        capture,
        {
            "started_at_utc", "ended_at_utc", "camera_ids", "clock_domain",
            "timestamp_semantics", "timestamp_unit", "reference_source",
            "independent_reference_declared",
        },
        "capture",
    )
    started = _utc_datetime(capture["started_at_utc"], "capture.started_at_utc")
    ended = _utc_datetime(capture["ended_at_utc"], "capture.ended_at_utc")
    if ended <= started:
        raise ValueError("capture.ended_at_utc must be after capture.started_at_utc")
    camera_ids = capture["camera_ids"]
    if not isinstance(camera_ids, list) or not 1 <= len(camera_ids) <= 32:
        raise ValueError("capture.camera_ids must contain 1..32 camera IDs")
    ids = [_integer(value, "camera_id", maximum=255) for value in camera_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("capture.camera_ids must be unique")
    _text(capture["clock_domain"], "capture.clock_domain", maximum=128)
    if capture["timestamp_semantics"] not in _TIMESTAMP_SEMANTICS:
        raise ValueError("capture.timestamp_semantics is not a supported declaration")
    if capture["timestamp_unit"] != "ns":
        raise ValueError("capture.timestamp_unit must be ns")
    _text(capture["reference_source"], "capture.reference_source", maximum=128)
    if type(capture["independent_reference_declared"]) is not bool:
        raise ValueError("capture.independent_reference_declared must be boolean")

    software = _object(manifest["software"], "software")
    _exact_keys(
        software,
        {
            "python_version", "host_os", "host_os_version",
            "application_commit_sha", "pseyepy_version",
            "led_controller_model", "led_controller_firmware_version",
            "devices",
        },
        "software",
    )
    for key in (
        "python_version", "host_os_version", "pseyepy_version",
        "led_controller_firmware_version",
    ):
        _version(software[key], "software." + key)
    _text(software["host_os"], "software.host_os", maximum=128)
    _text(software["led_controller_model"], "software.led_controller_model", maximum=128)
    if not isinstance(software["application_commit_sha"], str) or not _COMMIT.fullmatch(
        software["application_commit_sha"]
    ):
        raise ValueError("software.application_commit_sha must be a 40/64-character lowercase commit SHA")
    devices = software["devices"]
    if not isinstance(devices, list) or len(devices) != len(ids):
        raise ValueError("software.devices must describe each declared camera exactly once")
    device_ids = []
    for index, raw_device in enumerate(devices):
        device = _object(raw_device, f"software.devices[{index}]")
        _exact_keys(
            device,
            {
                "camera_id", "model", "firmware_version", "driver_version",
                "frame_width", "frame_height", "nominal_fps",
            },
            f"software.devices[{index}]",
        )
        device_ids.append(_integer(device["camera_id"], "device camera_id", maximum=255))
        _text(device["model"], "device model", maximum=128)
        _version(device["firmware_version"], "device firmware_version")
        _version(device["driver_version"], "device driver_version")
        _integer(device["frame_width"], "device frame_width", minimum=1, maximum=32768)
        _integer(device["frame_height"], "device frame_height", minimum=1, maximum=32768)
        fps = device["nominal_fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("device nominal_fps must be finite and positive")
        if not math.isfinite(float(fps)) or not 0 < float(fps) <= 10000:
            raise ValueError("device nominal_fps must be finite and positive")
    if len(set(device_ids)) != len(device_ids) or set(device_ids) != set(ids):
        raise ValueError("software.devices camera IDs must exactly match capture.camera_ids")

    revisions = _object(manifest["revisions"], "revisions")
    _exact_keys(
        revisions,
        {
            "pattern_sha256", "codebook_sha256",
            "controller_config_sha256", "calibration_sha256",
        },
        "revisions",
    )
    for key, value in revisions.items():
        _sha256(value, "revisions." + key)

    acceptance = _object(manifest["acceptance_limits"], "acceptance_limits")
    _exact_keys(
        acceptance,
        {
            "predeclared_at_utc", "max_exposure_skew_ns",
            "max_reprojection_rms_px", "max_marker_id_switches",
            "min_labeled_frames", "min_labeled_tracks",
        },
        "acceptance_limits",
    )
    predeclared = _utc_datetime(
        acceptance["predeclared_at_utc"], "acceptance_limits.predeclared_at_utc"
    )
    _integer(acceptance["max_exposure_skew_ns"], "max_exposure_skew_ns", minimum=1)
    rms = acceptance["max_reprojection_rms_px"]
    if isinstance(rms, bool) or not isinstance(rms, (int, float)):
        raise ValueError("max_reprojection_rms_px must be finite and positive")
    if not math.isfinite(float(rms)) or not 0 < float(rms) <= 10000:
        raise ValueError("max_reprojection_rms_px must be finite and positive")
    _integer(acceptance["max_marker_id_switches"], "max_marker_id_switches", maximum=1_000_000)
    _integer(acceptance["min_labeled_frames"], "min_labeled_frames", minimum=1, maximum=10_000_000)
    _integer(acceptance["min_labeled_tracks"], "min_labeled_tracks", minimum=1, maximum=1_000_000)

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= max_artifacts:
        raise ValueError(f"artifacts must contain 1..{max_artifacts} entries")
    seen_roles: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    declared_total = 0
    for index, raw_artifact in enumerate(artifacts):
        artifact = _object(raw_artifact, f"artifacts[{index}]")
        _exact_keys(
            artifact, {"role", "path", "size_bytes", "sha256", "media_type"},
            f"artifacts[{index}]",
        )
        role = _text(artifact["role"], "artifact role", maximum=64)
        if role not in _REQUIRED_ROLES and role not in _OPTIONAL_ROLES:
            raise ValueError(f"unknown artifact role: {role}")
        seen_roles[role] = seen_roles.get(role, 0) + 1
        rel = _text(artifact["path"], "artifact path", maximum=512)
        path_key = rel.casefold()
        seen_paths[path_key] = seen_paths.get(path_key, 0) + 1
        size = _integer(artifact["size_bytes"], "artifact size_bytes", minimum=1)
        if size > max_artifact_bytes:
            raise ValueError("declared artifact exceeds per-file byte cap")
        declared_total += size
        if declared_total > max_total_bytes:
            raise ValueError("declared artifacts exceed total byte cap")
        _sha256(artifact["sha256"], "artifact sha256")
        _text(artifact["media_type"], "artifact media_type", maximum=128)
    duplicate_roles = sorted(
        role for role, count in seen_roles.items()
        if count > 1 and role != "supplemental"
    )
    if duplicate_roles:
        raise ValueError("required artifact roles must be unique: " + ", ".join(duplicate_roles))
    return {
        "started_at": started,
        "ended_at": ended,
        "predeclared_at": predeclared,
        "camera_ids": ids,
        "role_counts": seen_roles,
        "path_counts": seen_paths,
        "declared_total_bytes": declared_total,
    }
