"""Offline JSON capture report CLI: statistics, not hardware certification.

Example:
    python offline_acceptance_cli.py recordings.json > acceptance-report.json

The SHA-256 identifies the exact recording bytes evaluated. A self-reported
clock domain or trigger reference cannot authenticate hardware provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from offline_capture_acceptance import (
    LabeledTemporalTrial,
    RecordedFrameBatch,
    replay_labeled_temporal_trials,
    replay_trigger_batches,
)
from temporal_pattern import (
    FrameStamp,
    LightSample,
    TemporalCodebook,
    _nonnegative_int,
)

_CANONICAL_MARKER_ID = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _dictionary(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _rows(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _record(cls, value: object, name: str):
    item = _dictionary(value, name)
    try:
        return cls(**item)
    except TypeError as exc:
        raise ValueError(f"{name} fields are incomplete or unknown") from exc


def evaluate_document(document: object) -> dict[str, Any]:
    """Parse strict versioned replay records into a JSON-safe summary."""
    root = _dictionary(document, "document")
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise ValueError("schema_version must equal integer 1")
    if "trigger" not in root and "temporal" not in root:
        raise ValueError("at least one trigger or temporal section is required")

    report: dict[str, Any] = {
        "schema_version": 1,
        "qualification": "software_replay_only",
        "hardware_provenance_verified": False,
        "independent_recording_provenance_required": True,
    }
    if "trigger" in root:
        config = _dictionary(root["trigger"], "trigger")
        batches = [
            RecordedFrameBatch(
                reference_trigger_id=_dictionary(item, "recording")["reference_trigger_id"],
                camera_frames=tuple(
                    _record(FrameStamp, frame, "camera frame")
                    for frame in _rows(
                        _dictionary(item, "recording")["camera_frames"],
                        "camera_frames",
                    )
                ),
            )
            for item in _rows(config["recordings"], "recordings")
        ]
        summary = replay_trigger_batches(
            batches,
            expected_camera_ids=_rows(config["expected_camera_ids"], "expected_camera_ids"),
            max_exposure_skew_ns=config["max_exposure_skew_ns"],
            allow_reference_gaps=config.get("allow_reference_gaps", False),
        )
        report["trigger"] = {
            "total_batches": summary.total_batches,
            "admitted_batches": summary.admitted_batches,
            "rejected_batches": summary.rejected_batches,
            "reference_gaps": summary.reference_gaps,
            "maximum_declared_skew_ns": summary.maximum_declared_skew_ns,
            "rejection_reasons": dict(summary.rejection_reasons),
            "hardware_provenance_verified": False,
        }

    if "temporal" in root:
        config = _dictionary(root["temporal"], "temporal")
        raw_codebook = _dictionary(config["codebook"], "codebook")
        pattern_map = {}
        for raw_id, bits in raw_codebook.items():
            if not isinstance(raw_id, str) or not _CANONICAL_MARKER_ID.fullmatch(raw_id):
                raise ValueError("codebook marker keys must be canonical nonnegative integers")
            marker = _nonnegative_int(int(raw_id), "marker_id")
            pattern_map[marker] = _rows(bits, "marker pattern")
        codebook = TemporalCodebook(
            pattern_map, max_bit_errors=config.get("max_bit_errors", 1)
        )
        trials = []
        for item in _rows(config["trials"], "trials"):
            entry = _dictionary(item, "temporal trial")
            trials.append(LabeledTemporalTrial(
                reference_marker_id=entry["reference_marker_id"],
                trigger_epoch=entry["trigger_epoch"],
                samples=tuple(
                    _record(LightSample, sample, "light sample")
                    for sample in _rows(entry["samples"], "light samples")
                ),
            ))
        summary = replay_labeled_temporal_trials(
            trials, codebook,
            min_valid_samples=config.get("min_valid_samples", 4),
        )
        report["temporal"] = {
            "trial_count": summary.trial_count,
            "correct_accepts": summary.correct_accepts,
            "wrong_accepts": summary.wrong_accepts,
            "rejected": summary.rejected,
            "acceptance_fraction": summary.acceptance_fraction,
            "false_accept_fraction": summary.false_accept_fraction,
            "conditional_false_accept_fraction": summary.conditional_false_accept_fraction,
            "wrong_accept_one_sided_upper_95": summary.wrong_accept_one_sided_upper_95,
            "rejection_reasons": dict(summary.rejection_reasons),
            "hardware_provenance_verified": False,
            "bound_assumes_independent_trials": True,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Version 1 JSON replay document")
    args = parser.parse_args(argv)
    try:
        content = args.recording.read_bytes()
        source = json.loads(
            content,
            parse_constant=lambda name: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {name}")
            ),
        )
        report = evaluate_document(source)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(f"invalid offline capture recording: {exc}")
    report["input_sha256"] = hashlib.sha256(content).hexdigest()
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
