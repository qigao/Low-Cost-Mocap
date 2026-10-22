"""Strict offline CLI for paired optical-only versus fused attitude evaluation.

Example:
    python fusion_ab_report_cli.py capture-ab.json > fusion-ab-report.json

The report evaluates the supplied data only. Hashes identify bytes and declared
revisions; they do not authenticate devices or prove independent ground truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from fusion_replay import (
    OpticalOnlyOrientationEstimate,
    ReferenceOrientation,
    compare_optical_only_vs_fused,
)
from optical_imu_fusion import OpticalIMUFusionState

MAX_INPUT_BYTES = 16_000_000
MAX_OUTPUT_BYTES = 16_000_000
_ROOT_KEYS = {
    "schema_version", "clock_domain", "max_match_delta_ns",
    "optical_only", "fused", "references",
}
_OPTICAL_KEYS = {
    "body_id", "timestamp_ns", "clock_domain", "quaternion_world_from_body_xyzw",
    "status", "pattern_sha256", "optical_calibration_sha256",
}
_FUSED_KEYS = {
    "body_id", "imu_id", "timestamp_ns", "clock_domain",
    "quaternion_world_from_body_xyzw", "attitude_variance_rad2", "status", "source",
    "last_optical_timestamp_ns", "last_optical_frame_id", "optical_age_ns",
    "clock_mapping_sha256", "imu_calibration_sha256", "optical_pattern_sha256",
    "optical_calibration_sha256", "global_position_m", "global_velocity_m_s",
    "global_position_observable",
}
_REFERENCE_KEYS = {
    "timestamp_ns", "clock_domain", "quaternion_world_from_body_xyzw",
    "reference_sha256",
}


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch (missing={missing}, unknown={unknown})")


def _record(cls, raw: object, expected: set[str], name: str):
    item = _object(raw, name)
    _exact_keys(item, expected, name)
    try:
        return cls(**item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def parse_input_bytes(raw: bytes) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, bytes):
        raise ValueError("input must be bytes")
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("input byte capacity exceeded")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input must be valid UTF-8 JSON") from exc
    return _object(document, "document"), hashlib.sha256(raw).hexdigest()


def evaluate_document(document: object) -> dict[str, Any]:
    root = _object(document, "document")
    _exact_keys(root, _ROOT_KEYS, "document")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("schema_version must equal integer 1")
    clock_domain = root["clock_domain"]
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise ValueError("clock_domain must be nonempty")
    if type(root["max_match_delta_ns"]) is not int or not 0 <= root["max_match_delta_ns"] <= (1 << 63) - 1:
        raise ValueError("max_match_delta_ns must be a nonnegative int64")

    optical = tuple(
        _record(OpticalOnlyOrientationEstimate, item, _OPTICAL_KEYS, "optical-only record")
        for item in _array(root["optical_only"], "optical_only")
    )
    fused = tuple(
        _record(OpticalIMUFusionState, item, _FUSED_KEYS, "fused state")
        for item in _array(root["fused"], "fused")
    )
    references = tuple(
        _record(ReferenceOrientation, item, _REFERENCE_KEYS, "reference record")
        for item in _array(root["references"], "references")
    )
    report = compare_optical_only_vs_fused(
        optical, fused, references,
        clock_domain=clock_domain,
        max_match_delta_ns=root["max_match_delta_ns"],
        max_report_bytes=MAX_INPUT_BYTES,
    )
    return {
        "schema_version": 1,
        "report_kind": "optical_vs_fused_attitude",
        "qualification": "software_replay_only",
        "hardware_ground_truth_verified": False,
        "reference_sha256": report.reference_sha256,
        "input_sha256": report.input_sha256,
        "report_sha256": report.report_sha256,
        "clock_domain": report.clock_domain,
        "match_tolerance_ns": report.match_tolerance_ns,
        "optical_only": asdict(report.optical_only),
        "fused": asdict(report.fused),
        "paired_reference_count": report.paired_reference_count,
        "paired_optical_rmse_rad": report.paired_optical_rmse_rad,
        "paired_fused_rmse_rad": report.paired_fused_rmse_rad,
        "paired_rmse_improvement_rad": report.paired_rmse_improvement_rad,
    }


def evaluate_bytes(raw: bytes) -> dict[str, Any]:
    document, raw_sha = parse_input_bytes(raw)
    report = evaluate_document(document)
    report["recording_sha256"] = raw_sha
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Version 1 optical-vs-fused JSON recording")
    args = parser.parse_args(argv)
    try:
        raw = args.recording.read_bytes()
        report = evaluate_bytes(raw)
        output = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("output byte capacity exceeded")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"fusion A/B report rejected: {exc}", file=sys.stderr)
        return 2
    # Emit only after all parsing, validation and serialization succeeds.
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
