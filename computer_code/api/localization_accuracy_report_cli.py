"""Strict JSON CLI for replaying localization events against reference positions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from localization_accuracy_report import (
    PositionReference,
    ProcessingLatencySample,
    evaluate_localization_accuracy,
)
from localization_replay import LocalizationReplayConfig, replay_localization_events
from localization_replay_cli import (
    DEFAULT_MAX_BYTES,
    _build,
    _constant,
    _event,
    _pairs,
    _require_list,
)
from imu_calibration import CorrectedIMUVectors, IMUCalibration
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import PositionSourceContract
from translational_eskf import TranslationalSeed
from translational_propagation import GravityModel

SCHEMA_VERSION = 1


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("report must be finite canonical JSON") from exc


def evaluate_document(document: object, *, max_output_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    if not isinstance(document, dict):
        raise ValueError("document must be a JSON object")
    required = {"schema_version", "config", "events", "references", "max_match_delta_ns"}
    optional = {"latency_samples"}
    if set(document) - required - optional or required - set(document):
        raise ValueError("document fields must be schema_version, config, events, references and max_match_delta_ns; latency_samples is optional")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must equal integer 1")

    config = _build(LocalizationReplayConfig, document["config"], nested={
        "seed": lambda value: _build(TranslationalSeed, value),
        "imu_calibration": lambda value: _build(IMUCalibration, value),
        "gravity_model": lambda value: _build(GravityModel, value),
        "position_source_contracts": lambda value: tuple(
            _build(PositionSourceContract, item, enums={})
            for item in _require_list(value, "position_source_contracts")
        ),
    })
    events = tuple(_event(item) for item in _require_list(document["events"], "events"))
    references = tuple(
        _build(PositionReference, item)
        for item in _require_list(document["references"], "references")
    )
    latency_samples = tuple(
        _build(ProcessingLatencySample, item)
        for item in _require_list(document.get("latency_samples", []), "latency_samples")
    )
    max_delta = document["max_match_delta_ns"]
    if type(max_delta) is not int or max_delta < 0:
        raise ValueError("max_match_delta_ns must be a nonnegative integer")

    replay = replay_localization_events(config, events)
    metrics = evaluate_localization_accuracy(
        replay.states,
        references,
        max_match_delta_ns=max_delta,
        latency_samples=latency_samples,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "qualification": "software_replay_and_reference_metrics_only",
        "replay": {
            "config_sha256": replay.config_sha256,
            "input_sha256": replay.input_sha256,
            "output_sha256": replay.output_sha256,
            "event_count": replay.event_count,
        },
        "accuracy": metrics,
        "reference_independence_verified": False,
        "hardware_ground_truth_verified": False,
        "hardware_provenance_verified": False,
    }
    encoded = _canonical(report).encode("utf-8")
    if len(encoded) > max_output_bytes:
        raise ValueError("accuracy report output byte capacity exceeded")
    report["acceptance_report_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def load_document_bytes(data: bytes, *, max_input_bytes: int = DEFAULT_MAX_BYTES,
                        max_output_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    if not isinstance(data, bytes):
        raise ValueError("input must be bytes")
    if len(data) > max_input_bytes:
        raise ValueError("accuracy report input byte capacity exceeded")
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input must be valid UTF-8 JSON") from exc
    return evaluate_document(document, max_output_bytes=max_output_bytes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="versioned replay/reference JSON")
    parser.add_argument("--max-input-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    try:
        if args.max_input_bytes <= 0 or args.max_output_bytes <= 0:
            raise ValueError("byte limits must be positive")
        with args.input.open("rb") as stream:
            data = stream.read(args.max_input_bytes + 1)
        report = load_document_bytes(
            data,
            max_input_bytes=args.max_input_bytes,
            max_output_bytes=args.max_output_bytes,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"localization accuracy report rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
