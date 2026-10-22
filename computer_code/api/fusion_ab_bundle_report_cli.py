"""Bundle-gated optical-vs-fused attitude report CLI.

Example:
    python fusion_ab_bundle_report_cli.py /path/to/capture-bundle > report.json

The CLI verifies the manifest and every declared artifact before reading the
fusion_ab_input role. This verifies bytes against a user-supplied manifest; it
does not authenticate hardware or prove that reference labels are independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from capture_bundle_manifest import validate_capture_bundle_and_read_role
from fusion_ab_report_cli import evaluate_bytes, parse_input_bytes, MAX_OUTPUT_BYTES


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate reference artifact JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite reference artifact JSON constant: {value}")


def evaluate_bundle(bundle_root: str | Path) -> dict:
    bundle, manifest_file_sha256, artifact, raw = validate_capture_bundle_and_read_role(
        bundle_root, role="fusion_ab_input", max_role_bytes=16_000_000,
    )
    ref_bundle, ref_manifest_file_sha, ref_artifact, ref_raw = validate_capture_bundle_and_read_role(
        bundle_root, role="reference_orientations", max_role_bytes=16_000_000,
    )
    imu_bundle, imu_manifest_file_sha, imu_artifact, imu_raw = validate_capture_bundle_and_read_role(
        bundle_root, role="imu_calibration", max_role_bytes=16_000_000,
    )
    # Each role read independently revalidates the full bundle. Reject any
    # manifest swap between those reads before combining evidence.
    if (
        len({bundle["manifest_sha256"], ref_bundle["manifest_sha256"], imu_bundle["manifest_sha256"]}) != 1
        or len({manifest_file_sha256, ref_manifest_file_sha, imu_manifest_file_sha}) != 1
    ):
        raise ValueError("capture bundle manifest changed during report evaluation")

    document, _raw_input_sha256 = parse_input_bytes(raw)
    reference_digest = hashlib.sha256(ref_raw).hexdigest()
    if not isinstance(document.get("references"), list) or not document["references"]:
        raise ValueError("A/B input must contain reference orientations")
    if any(not isinstance(item, dict) for item in document["references"]):
        raise ValueError("A/B references must be objects")
    if any(item.get("reference_sha256") != reference_digest for item in document["references"]):
        raise ValueError("A/B reference digest does not match reference_orientations artifact")

    try:
        reference_document = json.loads(
            ref_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference_orientations artifact must be UTF-8 JSON") from exc
    if not isinstance(reference_document, dict) or set(reference_document) != {"schema_version", "clock_domain", "poses"}:
        raise ValueError("reference_orientations artifact fields mismatch")
    if type(reference_document["schema_version"]) is not int or reference_document["schema_version"] != 1:
        raise ValueError("reference_orientations schema_version must equal integer 1")
    if reference_document["clock_domain"] != document.get("clock_domain"):
        raise ValueError("reference artifact clock domain disagrees with A/B input")
    poses = reference_document["poses"]
    if not isinstance(poses, list):
        raise ValueError("reference artifact poses must be an array")
    input_poses = [
        {
            "timestamp_ns": item.get("timestamp_ns"),
            "clock_domain": item.get("clock_domain"),
            "quaternion_world_from_body_xyzw": item.get("quaternion_world_from_body_xyzw"),
        }
        for item in document["references"]
    ]
    if poses != input_poses:
        raise ValueError("A/B reference records do not match bundled reference_orientations artifact")

    revisions = bundle["declared_revisions"]
    pattern_sha = revisions["pattern_sha256"]
    optical_calibration_sha = revisions["calibration_sha256"]
    imu_calibration_sha = imu_artifact["sha256"]
    for item in document.get("optical_only", []):
        if not isinstance(item, dict) or item.get("pattern_sha256") != pattern_sha:
            raise ValueError("optical-only pattern revision disagrees with capture bundle")
        if item.get("optical_calibration_sha256") != optical_calibration_sha:
            raise ValueError("optical-only calibration revision disagrees with capture bundle")
    for item in document.get("fused", []):
        if not isinstance(item, dict) or item.get("optical_pattern_sha256") != pattern_sha:
            raise ValueError("fused pattern revision disagrees with capture bundle")
        if item.get("optical_calibration_sha256") != optical_calibration_sha:
            raise ValueError("fused optical calibration revision disagrees with capture bundle")
        if item.get("imu_calibration_sha256") != imu_calibration_sha:
            raise ValueError("fused IMU calibration digest disagrees with imu_calibration artifact")

    raw_imu_entries = [
        item for item in bundle["artifact_checks"]
        if item["role"] == "raw_imu_samples" and item["status"] == "ok"
    ]
    if len(raw_imu_entries) != 1:
        raise ValueError("capture bundle must contain exactly one valid raw_imu_samples artifact")

    report = evaluate_bytes(raw)
    report.update({
        "qualification": "bundle_integrity_and_software_replay_only",
        "bundle_id": bundle["bundle_id"],
        "manifest_sha256": bundle["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "bundle_integrity_valid": True,
        "fusion_ab_artifact_path": artifact["path"],
        "fusion_ab_artifact_sha256": artifact["sha256"],
        "reference_artifact_path": ref_artifact["path"],
        "reference_artifact_sha256": ref_artifact["sha256"],
        "imu_calibration_artifact_path": imu_artifact["path"],
        "imu_calibration_artifact_sha256": imu_artifact["sha256"],
        "raw_imu_samples_artifact_sha256": raw_imu_entries[0]["actual_sha256"],
        "hardware_provenance_verified": False,
        "hardware_ground_truth_verified": False,
        "device_authenticity_verified": False,
        "independent_reference_verified": False,
        "timestamp_semantics_verified": False,
        "physical_metric_scale_verified": False,
    })
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["bundle_bound_report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path, help="capture bundle containing capture-bundle.json")
    args = parser.parse_args(argv)
    try:
        report = evaluate_bundle(args.bundle_root)
        output = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("output byte capacity exceeded")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"bundle-bound fusion report rejected: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
