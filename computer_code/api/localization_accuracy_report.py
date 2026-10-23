"""Deterministic localization accuracy metrics against supplied reference data.

This module reports software-computed errors and monotonic-clock latency. It
does not authenticate reference independence, sensor provenance or hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Sequence

from translational_eskf import TranslationalErrorState

_MAX_SAMPLES = 1_000_000


def _timestamp(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value < (1 << 63):
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("accuracy report input must be canonical finite JSON") from exc


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _error_metrics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "rmse": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (
        ordered[middle - 1] + ordered[middle]
    ) / 2.0
    return {
        "count": len(ordered),
        "rmse": math.sqrt(sum(value * value for value in ordered) / len(ordered)),
        "median": median,
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class PositionReference:
    timestamp_ns: int
    clock_domain: str
    position_world_from_body_m: tuple[float, float, float]
    dataset_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        object.__setattr__(
            self, "position_world_from_body_m",
            _vector3(self.position_world_from_body_m, "position_world_from_body_m"),
        )
        object.__setattr__(self, "dataset_sha256", _digest(self.dataset_sha256, "dataset_sha256"))


@dataclass(frozen=True)
class ProcessingLatencySample:
    event_id: str
    clock_domain: str
    ingress_monotonic_ns: int
    egress_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id", 128))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        object.__setattr__(
            self, "ingress_monotonic_ns",
            _timestamp(self.ingress_monotonic_ns, "ingress_monotonic_ns"),
        )
        object.__setattr__(
            self, "egress_monotonic_ns",
            _timestamp(self.egress_monotonic_ns, "egress_monotonic_ns"),
        )
        if self.egress_monotonic_ns < self.ingress_monotonic_ns:
            raise ValueError("latency egress must not precede ingress")


def evaluate_localization_accuracy(
    states: Sequence[TranslationalErrorState],
    references: Sequence[PositionReference],
    *,
    max_match_delta_ns: int,
    latency_samples: Sequence[ProcessingLatencySample] = (),
    max_samples: int = 100_000,
) -> dict[str, object]:
    """Compute one-to-one position errors, dropout drift, reacquisition and latency.

    P95/P99 use nearest-rank empirical percentiles. LOST/invalid position states
    remain in coverage counts; no sample is silently discarded to improve error.
    """
    if not isinstance(states, (tuple, list)) or not states:
        raise ValueError("states must be a nonempty ordered sequence")
    if not isinstance(references, (tuple, list)) or not references:
        raise ValueError("references must be a nonempty ordered sequence")
    if not isinstance(latency_samples, (tuple, list)):
        raise ValueError("latency_samples must be an ordered sequence")
    if type(max_samples) is not int or not 1 <= max_samples <= _MAX_SAMPLES:
        raise ValueError("max_samples must be in 1..1000000")
    if len(states) > max_samples or len(references) > max_samples or len(latency_samples) > max_samples:
        raise ValueError("localization accuracy sample capacity exceeded")
    if type(max_match_delta_ns) is not int or max_match_delta_ns < 0:
        raise ValueError("max_match_delta_ns must be a nonnegative integer")
    if any(not isinstance(state, TranslationalErrorState) for state in states):
        raise ValueError("states must contain TranslationalErrorState")
    if any(not isinstance(ref, PositionReference) for ref in references):
        raise ValueError("references must contain PositionReference")
    if any(not isinstance(sample, ProcessingLatencySample) for sample in latency_samples):
        raise ValueError("latency_samples must contain ProcessingLatencySample")

    state_identity = {
        (s.body_id, s.imu_id, s.map_frame_id, s.clock_domain,
         s.localization_source_revision_sha256, s.localization_calibration_sha256,
         s.imu_calibration_sha256, s.gravity_model_sha256, s.clock_mapping_sha256)
        for s in states
    }
    if len(state_identity) != 1:
        raise ValueError("states mix body, frame, clock or calibration revisions")
    identity = next(iter(state_identity))
    state_clock = identity[3]
    if any(ref.clock_domain != state_clock for ref in references):
        raise ValueError("reference clock domain does not match localization state")
    dataset_digests = {ref.dataset_sha256 for ref in references}
    if len(dataset_digests) != 1:
        raise ValueError("references mix dataset digests")
    if any(a.timestamp_ns >= b.timestamp_ns for a, b in zip(states, states[1:])):
        raise ValueError("state timestamps must be strictly increasing")
    if any(a.timestamp_ns >= b.timestamp_ns for a, b in zip(references, references[1:])):
        raise ValueError("reference timestamps must be strictly increasing")
    if len({sample.event_id for sample in latency_samples}) != len(latency_samples):
        raise ValueError("latency event IDs must be unique")
    if len({sample.clock_domain for sample in latency_samples}) > 1:
        raise ValueError("latency samples must share one monotonic clock domain")

    used: set[int] = set()
    per_reference: list[float | None] = [None] * len(references)
    per_state: list[float | None] = [None] * len(states)
    state_errors: list[float] = []
    dead_reckoning_errors: list[float] = []
    reacquisition_errors: list[float] = []
    unmatched_state_count = 0
    cursor = 0
    status_counts = {"TRACKING": 0, "DEGRADED": 0, "LOST": 0}
    for state_index, state in enumerate(states):
        status_counts[state.status] += 1
        position = state.position_world_from_body_m
        if position is None or state.status == "LOST":
            unmatched_state_count += 1
            continue
        lower = state.timestamp_ns - max_match_delta_ns
        upper = state.timestamp_ns + max_match_delta_ns
        while cursor < len(references) and references[cursor].timestamp_ns < lower:
            cursor += 1
        candidates = []
        index = cursor
        while index < len(references) and references[index].timestamp_ns <= upper:
            if index not in used:
                candidates.append((abs(references[index].timestamp_ns - state.timestamp_ns), index))
            index += 1
        if not candidates:
            unmatched_state_count += 1
            continue
        _, ref_index = min(candidates, key=lambda item: (item[0], item[1]))
        used.add(ref_index)
        reference = references[ref_index].position_world_from_body_m
        error = math.sqrt(sum((position[axis] - reference[axis]) ** 2 for axis in range(3)))
        per_state[state_index] = error
        per_reference[ref_index] = error
        state_errors.append(error)
        if state.source == "IMU_PROPAGATED":
            dead_reckoning_errors.append(error)
        if state_index > 0 and states[state_index - 1].status == "LOST":
            reacquisition_errors.append(error)
        cursor = min(cursor, ref_index)

    latency_ns = [
        sample.egress_monotonic_ns - sample.ingress_monotonic_ns
        for sample in latency_samples
    ]
    input_payload = {
        "states": [asdict(state) for state in states],
        "references": [asdict(ref) for ref in references],
        "latency_samples": [asdict(sample) for sample in latency_samples],
        "max_match_delta_ns": max_match_delta_ns,
    }
    input_sha = sha256(_canonical(input_payload).encode("utf-8")).hexdigest()
    report: dict[str, object] = {
        "schema_version": 1,
        "qualification": "reference_metrics_only",
        "input_sha256": input_sha,
        "reference_dataset_sha256": next(iter(dataset_digests)),
        "body_id": identity[0],
        "map_frame_id": identity[2],
        "clock_domain": state_clock,
        "max_match_delta_ns": max_match_delta_ns,
        "sample_count": len(states),
        "reference_count": len(references),
        "matched_count": len(state_errors),
        "unmatched_state_count": unmatched_state_count,
        "unmatched_reference_count": len(references) - len(used),
        "status_counts": status_counts,
        "position_error_m": _error_metrics(state_errors),
        "imu_propagated_position_error_m": _error_metrics(dead_reckoning_errors),
        "reacquisition_position_error_m": _error_metrics(reacquisition_errors),
        "per_reference_position_error_m": per_reference,
        "latency": {
            "sample_count": len(latency_ns),
            "clock_domain": latency_samples[0].clock_domain if latency_samples else None,
            "unit": "nanoseconds",
            "p50_ns": _percentile(latency_ns, 0.50),
            "p95_ns": _percentile(latency_ns, 0.95),
            "p99_ns": _percentile(latency_ns, 0.99),
            "max_ns": max(latency_ns) if latency_ns else None,
        },
        "reference_independence_verified": False,
        "hardware_ground_truth_verified": False,
        "hardware_provenance_verified": False,
    }
    report["report_sha256"] = sha256(_canonical(report).encode("utf-8")).hexdigest()
    return report
