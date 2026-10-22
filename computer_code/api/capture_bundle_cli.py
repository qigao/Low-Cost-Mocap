"""Validate a versioned capture-bundle directory and print a machine-readable report.

Example:
    python capture_bundle_cli.py ./capture-bundle > bundle-report.json

Exit 0 means schema and listed artifact bytes match their declarations only.
It is not a physical hardware or sensor-exposure certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from capture_bundle_manifest import (
    DEFAULT_MAX_ARTIFACTS,
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    validate_capture_bundle_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--max-artifacts", type=int, default=DEFAULT_MAX_ARTIFACTS)
    parser.add_argument(
        "--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES
    )
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    args = parser.parse_args(argv)
    try:
        report = validate_capture_bundle_directory(
            args.bundle_root,
            max_artifacts=args.max_artifacts,
            max_artifact_bytes=args.max_artifact_bytes,
            max_total_bytes=args.max_total_bytes,
        )
    except (ValueError, OSError, TypeError) as exc:
        report = {
            "schema_version": 1,
            "qualification": "capture_bundle_integrity_only",
            "bundle_integrity_valid": False,
            "failures": [str(exc)],
            "declarations_not_verified": {
                "hardware_provenance_verified": False,
                "device_authenticity_verified": False,
                "independent_reference_verified": False,
                "timestamp_semantics_verified": False,
                "physical_metric_scale_verified": False,
            },
        }
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["bundle_integrity_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
