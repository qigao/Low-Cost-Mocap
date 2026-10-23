"""Strict JSON-file front end for deterministic localization replay."""
from __future__ import annotations

from dataclasses import MISSING, asdict, fields
import argparse
import json
from pathlib import Path
import sys

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from localization_contract import LocalizationQuality
from localization_imu_fusion import LocalizationIMUAttitudeState
from localization_replay import (
    LocalizationReplayConfig, LocalizationReplayEvent, replay_localization_events,
)
from position_measurement import (
    MetricPositionMeasurement, PositionMeasurementSource, PositionSourceContract,
)
from translational_eskf import TranslationalSeed
from translational_propagation import GravityModel

SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 64_000_000


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze(item) for key, item in value.items()}
    return value


def _build(cls, raw, nested=None, enums=None):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    nested, enums = nested or {}, enums or {}
    all_fields = {f.name for f in fields(cls)}
    init_fields = {f.name for f in fields(cls) if f.init}
    derived_fields = all_fields - init_fields
    unknown = set(raw) - all_fields
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    required = {f.name for f in fields(cls)
                if f.init and f.default is MISSING and f.default_factory is MISSING}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{cls.__name__} is missing fields: {sorted(missing)}")
    values = {}
    for name, value in raw.items():
        if name in derived_fields:
            continue
        if name in nested:
            values[name] = None if value is None else nested[name](value)
        elif name in enums:
            try:
                values[name] = enums[name](value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported {cls.__name__}.{name}") from exc
        else:
            values[name] = _freeze(value)
    instance = cls(**values)
    for name in derived_fields & set(raw):
        if raw[name] != getattr(instance, name):
            raise ValueError(f"{cls.__name__}.{name} does not match the derived value")
    return instance


def _contract(raw):
    return _build(PositionSourceContract, raw,
                  enums={"source": PositionMeasurementSource})


def _measurement(raw):
    return _build(MetricPositionMeasurement, raw, enums={
        "source": PositionMeasurementSource, "quality": LocalizationQuality,
    })


def _event(raw):
    if not isinstance(raw, dict):
        raise ValueError("each event must be a JSON object")
    if raw.get("kind") == "imu":
        return _build(LocalizationReplayEvent, raw, nested={
            "sample": lambda v: _build(CorrectedIMUVectors, v),
            "attitude": lambda v: _build(LocalizationIMUAttitudeState, v),
        })
    if raw.get("kind") == "position":
        return _build(LocalizationReplayEvent, raw, nested={
            "measurement": _measurement,
        })
    raise ValueError("event kind must be imu or position")


def load_replay_document(data: bytes, *, max_input_bytes: int = DEFAULT_MAX_BYTES,
                         max_output_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    if not isinstance(data, bytes):
        raise ValueError("replay document must be bytes")
    if len(data) > max_input_bytes:
        raise ValueError("replay document byte capacity exceeded")
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("replay document must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "config", "events"}:
        raise ValueError("replay root must contain only schema_version, config and events")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported localization replay schema_version")
    config = _build(LocalizationReplayConfig, document["config"], nested={
        "seed": lambda v: _build(TranslationalSeed, v),
        "imu_calibration": lambda v: _build(IMUCalibration, v),
        "gravity_model": lambda v: _build(GravityModel, v),
        "position_source_contracts": lambda v: tuple(_contract(x) for x in _require_list(v, "position_source_contracts")),
    })
    events_raw = _require_list(document["events"], "events")
    events = tuple(_event(item) for item in events_raw)
    result = replay_localization_events(
        config, events, max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": result.config_sha256,
        "input_sha256": result.input_sha256,
        "output_sha256": result.output_sha256,
        "event_count": result.event_count,
        "final_state": asdict(result.states[-1]),
        "hardware_provenance_verified": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > max_output_bytes:
        raise ValueError("replay report byte capacity exceeded")
    return report


def _require_list(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="versioned localization replay JSON")
    parser.add_argument("--max-input-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    try:
        if args.max_input_bytes <= 0 or args.max_output_bytes <= 0:
            raise ValueError("byte limits must be positive")
        with args.input.open("rb") as stream:
            data = stream.read(args.max_input_bytes + 1)
        report = load_replay_document(data, max_input_bytes=args.max_input_bytes,
                                      max_output_bytes=args.max_output_bytes)
        print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"localization replay rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
