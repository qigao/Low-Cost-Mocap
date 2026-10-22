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
MAX_RECORDING_BYTES = 64 * 1024 * 1024


def _dictionary(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_keys(
    value: dict, *, required: set[str], optional: set[str] = frozenset(), name: str
) -> None:
    actual = set(value)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise ValueError(f"{name} fields invalid: {'; '.join(details)}")


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


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
    _exact_keys(
        root, required={"schema_version"}, optional={"trigger", "temporal"}, name="document"
    )
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
        _exact_keys(
            config,
            required={"expected_camera_ids", "max_exposure_skew_ns", "recordings"},
            optional={"allow_reference_gaps"},
            name="trigger",
        )
        recordings = _rows(config["recordings"], "recordings")
        for item in recordings:
            _exact_keys(
                _dictionary(item, "recording"),
                required={"reference_trigger_id", "camera_frames"},
                name="recording",
            )
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
            "admitted_exposure_skew_ns_percentiles": dict(
                summary.admitted_exposure_skew_ns_percentiles
            ),
            "comparable_exposure_skew_batch_count": summary.comparable_exposure_skew_batch_count,
            "comparable_exposure_skew_ns_percentiles": dict(
                summary.comparable_exposure_skew_ns_percentiles
            ),
            "rejection_reasons": dict(summary.rejection_reasons),
            "hardware_provenance_verified": False,
        }

    if "temporal" in root:
        config = _dictionary(root["temporal"], "temporal")
        _exact_keys(
            config,
            required={"codebook", "trials"},
            optional={"max_bit_errors", "min_valid_samples"},
            name="temporal",
        )
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
            _exact_keys(
                entry,
                required={"reference_marker_id", "trigger_epoch", "samples"},
                name="temporal trial",
            )
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
            "trial_independence_verified": False,
            "statistical_limitations": [
                "wrong_accept_one_sided_upper_95 is a one-sided Wilson score bound that assumes independent Bernoulli trials.",
                "Capture-session, trajectory, marker and code-cycle correlation is not modeled; nominal coverage is not justified until trial independence is supported or a cluster-aware analysis is performed.",
            ],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Version 1 JSON replay document")
    args = parser.parse_args(argv)
    try:
        with args.recording.open("rb") as recording_file:
            content = recording_file.read(MAX_RECORDING_BYTES + 1)
        if len(content) > MAX_RECORDING_BYTES:
            raise ValueError(f"recording exceeds {MAX_RECORDING_BYTES} byte limit")
        source = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        report = evaluate_document(source)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(f"invalid offline capture recording: {exc}")
    report["input_sha256"] = hashlib.sha256(content).hexdigest()
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
