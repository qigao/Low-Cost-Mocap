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

from capture_bundle_manifest import validate_capture_bundle_and_read_role
from fusion_ab_report_cli import evaluate_bytes, MAX_OUTPUT_BYTES


def evaluate_bundle(bundle_root: str | Path) -> dict:
    bundle, manifest_file_sha256, artifact, raw = validate_capture_bundle_and_read_role(
        bundle_root,
        role="fusion_ab_input",
        max_role_bytes=16_000_000,
    )
    report = evaluate_bytes(raw)
    report.update({
        "qualification": "bundle_integrity_and_software_replay_only",
        "bundle_id": bundle["bundle_id"],
        "manifest_sha256": bundle["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "bundle_integrity_valid": True,
        "fusion_ab_artifact_path": artifact["path"],
        "fusion_ab_artifact_sha256": artifact["sha256"],
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
