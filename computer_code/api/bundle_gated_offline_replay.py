"""Gate offline replay on capture-bundle integrity and emit one combined report.

This orchestration does not replace offline_acceptance_cli. It verifies that
the exact replay input is a hashed artifact in a valid bundle, then invokes
the existing evaluator. All output remains software-replay-only and never
claims hardware, sensor-exposure or metric-scale qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from capture_bundle_manifest import (
    DEFAULT_MAX_ARTIFACTS,
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    _open_directory_path_no_follow,
    _read_regular_file_at,
    _validate_capture_bundle_at,
    load_manifest_bytes,
    validate_capture_bundle_directory,
)
from offline_acceptance_cli import evaluate_document


DEFAULT_MAX_REPLAY_INPUT_BYTES = 64 * 1024 * 1024
_REPLAY_ROOT_FIELDS = {"schema_version", "trigger", "temporal"}


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _base_report(bundle_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "qualification": "bundle_integrity_and_software_replay_only",
        "bundle_integrity_valid": bundle_report.get("bundle_integrity_valid") is True,
        "bundle_manifest_sha256": bundle_report.get("manifest_sha256"),
        "bundle_report": bundle_report,
        "offline_replay_status": "not_started",
        "offline_replay_input_sha256": None,
        "offline_replay_report": None,
        "failures": [],
        "declarations_not_verified": {
            "hardware_provenance_verified": False,
            "device_authenticity_verified": False,
            "independent_reference_verified": False,
            "timestamp_semantics_verified": False,
            "physical_metric_scale_verified": False,
        },
    }


def run_bundle_gated_offline_replay(
    bundle_root: str | Path,
    *,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_replay_input_bytes: int = DEFAULT_MAX_REPLAY_INPUT_BYTES,
) -> dict[str, Any]:
    """Validate and replay from the same anchored bundle-root descriptor."""
    if type(max_replay_input_bytes) is not int or not 1 <= max_replay_input_bytes <= 1_000_000_000:
        raise ValueError("max_replay_input_bytes must be within 1..1000000000")
    root_input = Path(bundle_root)
    root_fd = None
    try:
        if root_input.is_symlink():
            raise ValueError("bundle root must not itself be a symlink")
        root = root_input.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("bundle root must be a directory")
        root_fd = _open_directory_path_no_follow(root)
        manifest_raw = _read_regular_file_at(
            root_fd,
            MANIFEST_NAME,
            max_bytes=MAX_MANIFEST_BYTES,
            cap_error="manifest exceeds byte cap",
        )
        manifest = load_manifest_bytes(manifest_raw)
        bundle_report = _validate_capture_bundle_at(
            manifest,
            root,
            root_fd,
            max_artifacts=max_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            max_total_bytes=max_total_bytes,
        )
        bundle_report["manifest_file_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    except (ValueError, OSError, TypeError) as exc:
        if root_fd is not None:
            os.close(root_fd)
            root_fd = None
        report = _base_report({
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
        })
        report["offline_replay_status"] = "invalid_bundle_manifest"
        report["failures"].append(str(exc))
        return report

    report = _base_report(bundle_report)
    if not bundle_report["bundle_integrity_valid"]:
        report["offline_replay_status"] = "skipped_bundle_integrity_invalid"
        report["failures"].extend(bundle_report["failures"])
        os.close(root_fd)
        root_fd = None
        return report

    try:
        artifacts = manifest["artifacts"]
        replay_artifacts = [
            artifact for artifact in artifacts
            if artifact.get("role") == "offline_replay_input"
        ]
        if len(replay_artifacts) != 1:
            report["offline_replay_status"] = "missing_or_duplicate_replay_input"
            report["failures"].append(
                "bundle must contain exactly one offline_replay_input artifact"
            )
            return report
        artifact = replay_artifacts[0]
        replay_raw = _read_regular_file_at(
            root_fd,
            artifact["path"],
            max_bytes=max_replay_input_bytes,
            cap_error="input exceeds replay byte cap",
        )
        replay_sha = hashlib.sha256(replay_raw).hexdigest()
        if (
            len(replay_raw) != artifact["size_bytes"]
            or replay_sha != artifact["sha256"]
        ):
            raise ValueError("offline replay input changed after bundle validation")
        replay_document = json.loads(
            replay_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if not isinstance(replay_document, dict):
            raise ValueError("offline replay input must be a JSON object")
        extra = set(replay_document) - _REPLAY_ROOT_FIELDS
        if extra:
            raise ValueError(
                "offline replay input has unknown fields: " + ", ".join(sorted(extra))
            )
        if (
            type(replay_document.get("schema_version")) is not int
            or replay_document["schema_version"] != 1
            or not ({"trigger", "temporal"} & set(replay_document))
        ):
            raise ValueError("offline replay input needs schema_version 1 and trigger or temporal section")
        replay_report = evaluate_document(replay_document)

        # Revalidate by pathname after evaluation to detect replacement or
        # mutation of the bundle root, manifest, or sibling evidence artifacts.
        try:
            final_bundle = validate_capture_bundle_directory(
                root_input,
                max_artifacts=max_artifacts,
                max_artifact_bytes=max_artifact_bytes,
                max_total_bytes=max_total_bytes,
            )
        except (OSError, ValueError, TypeError) as exc:
            report["bundle_integrity_valid"] = False
            report["offline_replay_status"] = "bundle_changed_during_replay"
            report["failures"].append(
                "final bundle revalidation failed: " + str(exc)
            )
            return report
        if (
            not final_bundle["bundle_integrity_valid"]
            or final_bundle.get("manifest_file_sha256")
            != bundle_report.get("manifest_file_sha256")
            or final_bundle.get("manifest_sha256")
            != bundle_report.get("manifest_sha256")
        ):
            report["bundle_integrity_valid"] = False
            report["bundle_report"] = final_bundle
            report["offline_replay_status"] = "bundle_changed_during_replay"
            report["failures"].extend(final_bundle.get("failures", []))
            report["failures"].append("bundle integrity changed while replay was running")
            return report
    except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        report["offline_replay_status"] = "invalid_replay_input"
        report["failures"].append(str(exc))
        return report
    finally:
        if root_fd is not None:
            os.close(root_fd)

    report["offline_replay_status"] = "completed_software_replay"
    report["offline_replay_input_sha256"] = replay_sha
    report["offline_replay_report"] = replay_report
    report["declarations_not_verified"]["hardware_provenance_verified"] = False
    report["declarations_not_verified"]["independent_reference_verified"] = False
    report["declarations_not_verified"]["timestamp_semantics_verified"] = False
    report["declarations_not_verified"]["physical_metric_scale_verified"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--max-artifacts", type=int, default=DEFAULT_MAX_ARTIFACTS)
    parser.add_argument(
        "--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES
    )
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument(
        "--max-replay-input-bytes", type=int,
        default=DEFAULT_MAX_REPLAY_INPUT_BYTES,
    )
    args = parser.parse_args(argv)
    try:
        report = run_bundle_gated_offline_replay(
            args.bundle_root,
            max_artifacts=args.max_artifacts,
            max_artifact_bytes=args.max_artifact_bytes,
            max_total_bytes=args.max_total_bytes,
            max_replay_input_bytes=args.max_replay_input_bytes,
        )
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({
            "schema_version": 1,
            "qualification": "bundle_integrity_and_software_replay_only",
            "bundle_integrity_valid": False,
            "offline_replay_status": "invalid_input",
            "failures": [str(exc)],
            "declarations_not_verified": {
                "hardware_provenance_verified": False,
                "device_authenticity_verified": False,
                "independent_reference_verified": False,
                "timestamp_semantics_verified": False,
                "physical_metric_scale_verified": False,
            },
        }, sort_keys=True, indent=2, allow_nan=False))
        return 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    if report["offline_replay_status"] == "completed_software_replay":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
