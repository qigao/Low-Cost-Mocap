"""Strict capture-bundle manifest and artifact-integrity validator.

A valid report proves only that the supplied manifest and bytes satisfy this
software schema and its declared hashes/sizes. It does NOT authenticate the
devices, prove that a reference logger was independent, or verify that a
timestamp was captured at sensor exposure.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import math
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
    "pattern_config",
    "codebook",
    "controller_config",
    "calibration_config",
)
_OPTIONAL_ROLES = {"supplemental", "offline_replay_input"}
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
    if (
        not isinstance(capture["timestamp_semantics"], str)
        or capture["timestamp_semantics"] not in _TIMESTAMP_SEMANTICS
    ):
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
        try:
            fps_value = float(fps)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("device nominal_fps must be finite and positive") from exc
        if not math.isfinite(fps_value) or not 0 < fps_value <= 10000:
            raise ValueError("device nominal_fps must be finite and positive")
    if len(set(device_ids)) != len(device_ids) or set(device_ids) != set(ids):
        raise ValueError("software.devices camera IDs must exactly match capture.camera_ids")

    revisions = _object(manifest["revisions"], "revisions")
    _exact_keys(
        revisions,
        {
            "pattern_sha256", "pattern_config_artifact_sha256",
            "codebook_sha256", "controller_config_sha256",
            "calibration_sha256",
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
    try:
        rms_value = float(rms)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("max_reprojection_rms_px must be finite and positive") from exc
    if not math.isfinite(rms_value) or not 0 < rms_value <= 10000:
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



def _canonical_artifact_parts(raw_path: str) -> tuple[str, ...]:
    if (
        not raw_path
        or "\\" in raw_path
        or "\x00" in raw_path
        or raw_path.startswith("/")
        or re.match(r"^[A-Za-z]:", raw_path)
    ):
        raise ValueError("artifact path must be a canonical relative POSIX path")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or str(relative) != raw_path
    ):
        raise ValueError("artifact path contains traversal or noncanonical components")
    return relative.parts


def _safe_artifact_path(root: Path, raw_path: str) -> Path:
    """Compatibility path resolver; security-sensitive reads use directory FDs."""
    parts = _canonical_artifact_parts(raw_path)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact path traverses a symlink")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ValueError("artifact path cannot be resolved safely") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path escapes bundle root")
    return resolved


def _require_dirfd_primitives() -> None:
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ValueError("secure bundle validation requires O_NOFOLLOW, O_DIRECTORY and O_NONBLOCK")
    if os.open not in getattr(os, "supports_dir_fd", set()):
        raise ValueError("secure bundle validation requires descriptor-relative open support")


def _open_directory_path_no_follow(path: Path) -> int:
    """Open an absolute directory by walking components without following links."""
    _require_dirfd_primitives()
    if not path.is_absolute():
        raise ValueError("bundle root must be an absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path.anchor or "/", flags)
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("bundle root must be a directory")
        return descriptor
    except OSError as exc:
        raise ValueError("bundle root cannot be opened without following links") from exc
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_file_at(root_fd: int, raw_path: str) -> int:
    """Open a regular-file candidate using only no-follow descriptor-relative steps."""
    parts = _canonical_artifact_parts(raw_path)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        descriptor = os.open(parts[-1], file_flags, dir_fd=parent_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("artifact path cannot be opened without following links") from exc
    finally:
        os.close(parent_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("artifact is not a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_at(
    root_fd: int,
    raw_path: str,
    *,
    max_bytes: int,
    cap_error: str,
) -> bytes:
    descriptor = _open_regular_file_at(root_fd, raw_path)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > max_bytes:
            raise ValueError(cap_error)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(CHUNK_BYTES, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(cap_error)
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError("file cannot be read safely") from exc
    finally:
        os.close(descriptor)


def _hash_regular_file_at(
    root_fd: int,
    raw_path: str,
    *,
    max_bytes: int,
    cap_error: str,
) -> tuple[int, str]:
    descriptor = _open_regular_file_at(root_fd, raw_path)
    digest = hashlib.sha256()
    total = 0
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > max_bytes:
            raise ValueError(cap_error)
        while True:
            chunk = os.read(descriptor, min(CHUNK_BYTES, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(cap_error)
            digest.update(chunk)
        return total, digest.hexdigest()
    except OSError as exc:
        raise ValueError("artifact cannot be read safely") from exc
    finally:
        os.close(descriptor)


def validate_capture_bundle(
    manifest: object,
    bundle_root: str | Path,
    *,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Validate a manifest and hash artifacts through an anchored root descriptor."""
    root_input = Path(bundle_root)
    if root_input.is_symlink():
        raise ValueError("bundle root must not itself be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("bundle root must exist and resolve safely") from exc
    if not root.is_dir():
        raise ValueError("bundle root must be a directory")
    root_fd = _open_directory_path_no_follow(root)
    try:
        return _validate_capture_bundle_at(
            manifest,
            root,
            root_fd,
            max_artifacts=max_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            max_total_bytes=max_total_bytes,
        )
    finally:
        os.close(root_fd)


def _validate_capture_bundle_at(
    manifest: object,
    bundle_root: Path,
    root_fd: int,
    *,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Validate manifest schema and artifact bytes without authenticating hardware.

    Unsafe paths, missing files and digest/size mismatches appear as per-file
    failures in the returned report. Unknown schema fields or unbounded
    declarations fail the entire invocation before opening artifacts.
    """
    manifest_obj = _object(manifest, "manifest")
    max_count = _integer(max_artifacts, "max_artifacts", minimum=1, maximum=1024)
    max_one = _integer(max_artifact_bytes, "max_artifact_bytes", minimum=1)
    max_all = _integer(max_total_bytes, "max_total_bytes", minimum=1)
    metadata = _validate_structure(
        manifest_obj, max_artifacts=max_count,
        max_artifact_bytes=max_one, max_total_bytes=max_all,
    )
    root = bundle_root

    canonical = _canonical_json(manifest_obj)
    manifest_digest = hashlib.sha256(canonical).hexdigest()
    artifact_checks = []
    failures = []
    required_missing = sorted(set(_REQUIRED_ROLES) - set(metadata["role_counts"]))
    if required_missing:
        failures.append("missing required roles: " + ", ".join(required_missing))
    capture = manifest_obj["capture"]
    acceptance = manifest_obj["acceptance_limits"]
    if not capture["independent_reference_declared"]:
        failures.append("independent reference source is not declared")
    predeclared_before_capture = metadata["predeclared_at"] < metadata["started_at"]
    if not predeclared_before_capture:
        failures.append("acceptance limits were not timestamped before capture start")

    role_to_artifact = {
        artifact["role"]: artifact
        for artifact in manifest_obj["artifacts"]
        if artifact["role"] != "supplemental"
    }
    revision_links = (
        ("pattern_config", "pattern_config_artifact_sha256"),
        ("codebook", "codebook_sha256"),
        ("controller_config", "controller_config_sha256"),
        ("calibration_config", "calibration_sha256"),
    )
    for role, revision_key in revision_links:
        artifact = role_to_artifact.get(role)
        if artifact is not None and artifact["sha256"] != manifest_obj["revisions"][revision_key]:
            failures.append(f"{role} artifact SHA does not match revisions.{revision_key}")

    total_actual_bytes = 0
    path_counts = metadata["path_counts"]
    for index, artifact in enumerate(manifest_obj["artifacts"]):
        entry = {
            "index": index,
            "role": artifact["role"],
            "path": artifact["path"],
            "declared_size_bytes": artifact["size_bytes"],
            "declared_sha256": artifact["sha256"],
            "actual_size_bytes": None,
            "actual_sha256": None,
            "status": "unchecked",
            "errors": [],
        }
        if path_counts[artifact["path"].casefold()] > 1:
            entry["status"] = "duplicate_path"
            entry["errors"].append("artifact path duplicates another entry")
        else:
            try:
                remaining_total = max_all - total_actual_bytes
                if remaining_total <= 0:
                    raise ValueError("actual artifacts exceed total byte cap")
                cap = min(max_one, remaining_total)
                cap_error = (
                    "actual artifacts exceed total byte cap"
                    if remaining_total < max_one
                    else "actual artifact exceeds per-file byte cap"
                )
                actual_size, actual_hash = _hash_regular_file_at(
                    root_fd,
                    artifact["path"],
                    max_bytes=cap,
                    cap_error=cap_error,
                )
                total_actual_bytes += actual_size
                entry["actual_size_bytes"] = actual_size
                entry["actual_sha256"] = actual_hash
                if actual_size != artifact["size_bytes"]:
                    entry["errors"].append("size mismatch")
                if actual_hash != artifact["sha256"]:
                    entry["errors"].append("SHA-256 mismatch")
                entry["status"] = "ok" if not entry["errors"] else "mismatch"
            except FileNotFoundError:
                entry["status"] = "missing"
                entry["errors"].append("artifact file is missing")
            except (OSError, ValueError) as exc:
                entry["status"] = "unsafe_or_unreadable"
                entry["errors"].append(str(exc))
        if entry["status"] != "ok":
            failures.append(
                f"artifact[{index}] {artifact['role']} ({artifact['path']}): "
                + "; ".join(entry["errors"])
            )
        artifact_checks.append(entry)

    revisions = manifest_obj["revisions"]
    report = {
        "schema_version": 1,
        "qualification": "capture_bundle_integrity_only",
        "bundle_id": manifest_obj["bundle_id"],
        "manifest_sha256": manifest_digest,
        "bundle_integrity_valid": not failures,
        "failures": failures,
        "artifact_count": len(artifact_checks),
        "required_roles_missing": required_missing,
        "artifact_checks": artifact_checks,
        "total_actual_bytes_hashed": total_actual_bytes,
        "declared_capture": {
            "started_at_utc": capture["started_at_utc"],
            "ended_at_utc": capture["ended_at_utc"],
            "camera_ids": list(metadata["camera_ids"]),
            "clock_domain": capture["clock_domain"],
            "timestamp_semantics": capture["timestamp_semantics"],
            "timestamp_unit": capture["timestamp_unit"],
            "reference_source": capture["reference_source"],
            "independent_reference_declared": capture["independent_reference_declared"],
            "acceptance_predeclared_at_utc": acceptance["predeclared_at_utc"],
            "acceptance_predeclared_before_capture": predeclared_before_capture,
        },
        "declared_revisions": dict(revisions),
        "declared_acceptance_limits": dict(acceptance),
        "declared_software": {
            key: value for key, value in manifest_obj["software"].items()
            if key != "devices"
        },
        "declarations_not_verified": {
            "hardware_provenance_verified": False,
            "device_authenticity_verified": False,
            "independent_reference_verified": False,
            "timestamp_semantics_verified": False,
            "physical_metric_scale_verified": False,
        },
    }
    return report


def validate_capture_bundle_directory(
    bundle_root: str | Path,
    *,
    manifest_name: str = MANIFEST_NAME,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Read the manifest and validate every artifact under one anchored root FD."""
    root_input = Path(bundle_root)
    if root_input.is_symlink():
        raise ValueError("bundle root must not itself be a symlink")
    if manifest_name != MANIFEST_NAME:
        raise ValueError("manifest name must be capture-bundle.json")
    try:
        root = root_input.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("bundle root must exist and resolve safely") from exc
    if not root.is_dir():
        raise ValueError("bundle root must be a directory")

    root_fd = _open_directory_path_no_follow(root)
    try:
        raw = _read_regular_file_at(
            root_fd,
            MANIFEST_NAME,
            max_bytes=MAX_MANIFEST_BYTES,
            cap_error="manifest exceeds byte cap",
        )
        manifest = load_manifest_bytes(raw)
        report = _validate_capture_bundle_at(
            manifest,
            root,
            root_fd,
            max_artifacts=max_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            max_total_bytes=max_total_bytes,
        )
        report["manifest_file_sha256"] = hashlib.sha256(raw).hexdigest()
        return report
    finally:
        os.close(root_fd)
