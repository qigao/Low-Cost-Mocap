"""Deterministic bounded replay and independent-reference attitude metrics.

This module consumes already timestamped software observations/states. A replay
digest proves reproducibility of these inputs and outputs only; it is not sensor
authenticity, hardware synchronization, or ground-truth certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Sequence

from fused_attitude_stream import fused_attitude_state_to_dict
from imu_contract import ClockMapping
from imu_propagation import AttitudePropagationState
from optical_imu_fusion import (
    OpticalAttitudeMeasurement,
    OpticalIMUAttitudeFusion,
    OpticalIMUFusionState,
)

_GENESIS = "0" * 64
_MAX_EVENTS = 1_000_000
_MAX_BYTES = 256_000_000
_MAX_I64 = (1 << 63) - 1


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("replay data must be finite canonical JSON") from exc


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric")
    return number


def _quaternion(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain four XYZW values")
    q = tuple(_finite(x, name) for x in value)
    norm = math.sqrt(sum(x*x for x in q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must have finite nonzero norm")
    return tuple(x/norm for x in q)


@dataclass(frozen=True)
class FusionReplayConfig:
    body_id: str
    imu_id: str
    imu_clock_domain: str
    imu_calibration_revision: str
    imu_calibration_sha256: str
    optical_pattern_sha256: str
    optical_calibration_sha256: str
    clock_mapping: ClockMapping
    gyro_noise_density_rad_s_sqrt_hz: tuple[float, float, float]
    optical_variance_rad2: float = 0.0025
    max_dropout_ns: int = 100_000_000
    max_innovation_rad: float = 1.2
    max_alignment_angular_rate_rad_s: float = 20.0
    history_capacity: int = 256

    def __post_init__(self) -> None:
        for name in ("body_id", "imu_id", "imu_clock_domain", "imu_calibration_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        for name in ("imu_calibration_sha256", "optical_pattern_sha256", "optical_calibration_sha256"):
            _digest(getattr(self, name), name)
        if not isinstance(self.clock_mapping, ClockMapping):
            raise ValueError("clock_mapping must be ClockMapping")
        if self.clock_mapping.target_clock_domain != self.imu_clock_domain:
            raise ValueError("clock mapping target must match IMU clock domain")
        if not isinstance(self.gyro_noise_density_rad_s_sqrt_hz, (tuple, list)) or len(self.gyro_noise_density_rad_s_sqrt_hz) != 3:
            raise ValueError("gyro noise density must contain three values")
        noise = tuple(_finite(x, "gyro noise density") for x in self.gyro_noise_density_rad_s_sqrt_hz)
        if any(x < 0 for x in noise):
            raise ValueError("gyro noise density must be nonnegative")
        object.__setattr__(self, "gyro_noise_density_rad_s_sqrt_hz", noise)
        if type(self.max_dropout_ns) is not int or not 1 <= self.max_dropout_ns <= _MAX_I64:
            raise ValueError("max_dropout_ns must be a positive int64")
        if type(self.history_capacity) is not int or not 3 <= self.history_capacity <= 4096:
            raise ValueError("history_capacity must be in 3..4096")
        if _finite(self.optical_variance_rad2, "optical_variance_rad2") <= 0:
            raise ValueError("optical variance must be positive")
        if not 0 < _finite(self.max_innovation_rad, "max_innovation_rad") <= math.pi:
            raise ValueError("max innovation must be in (0, pi]")
        if _finite(self.max_alignment_angular_rate_rad_s, "max_alignment_angular_rate_rad_s") < 0:
            raise ValueError("alignment angular-rate limit must be nonnegative")

    def create_fusion(self) -> OpticalIMUAttitudeFusion:
        return OpticalIMUAttitudeFusion(
            body_id=self.body_id,
            imu_id=self.imu_id,
            imu_clock_domain=self.imu_clock_domain,
            imu_calibration_revision=self.imu_calibration_revision,
            imu_calibration_sha256=self.imu_calibration_sha256,
            optical_pattern_sha256=self.optical_pattern_sha256,
            optical_calibration_sha256=self.optical_calibration_sha256,
            clock_mapping=self.clock_mapping,
            gyro_noise_density_rad_s_sqrt_hz=self.gyro_noise_density_rad_s_sqrt_hz,
            optical_variance_rad2=self.optical_variance_rad2,
            max_dropout_ns=self.max_dropout_ns,
            max_innovation_rad=self.max_innovation_rad,
            max_alignment_angular_rate_rad_s=self.max_alignment_angular_rate_rad_s,
            history_capacity=self.history_capacity,
        )

    @property
    def config_sha256(self) -> str:
        data = asdict(self)
        return sha256(_canonical(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FusionReplayEvent:
    event_index: int
    kind: str
    value: AttitudePropagationState | OpticalAttitudeMeasurement

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be a nonnegative integer")
        if self.kind == "imu":
            if not isinstance(self.value, AttitudePropagationState):
                raise ValueError("imu event requires AttitudePropagationState")
        elif self.kind == "optical":
            if not isinstance(self.value, OpticalAttitudeMeasurement):
                raise ValueError("optical event requires OpticalAttitudeMeasurement")
        else:
            raise ValueError("unsupported fusion replay event kind")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "kind": self.kind,
            "value": asdict(self.value),
        }


@dataclass(frozen=True)
class FusionReplayResult:
    config_sha256: str
    input_sha256: str
    output_sha256: str
    event_count: int
    output_count: int
    states: tuple[OpticalIMUFusionState, ...]
    hardware_provenance_verified: bool = False


def replay_fusion_events(
    config: FusionReplayConfig,
    events: Sequence[FusionReplayEvent],
    *,
    max_events: int = 100_000,
    max_input_bytes: int = 64_000_000,
    max_output_bytes: int = 64_000_000,
) -> FusionReplayResult:
    """Replay ordered events transactionally; any error raises with no result."""
    if not isinstance(config, FusionReplayConfig):
        raise ValueError("config must be FusionReplayConfig")
    if not isinstance(events, (tuple, list)):
        raise ValueError("events must be an ordered list or tuple")
    if type(max_events) is not int or not 1 <= max_events <= _MAX_EVENTS:
        raise ValueError("max_events must be in 1..1000000")
    if type(max_input_bytes) is not int or not 1 <= max_input_bytes <= _MAX_BYTES:
        raise ValueError("max_input_bytes must be in 1..256000000")
    if not events:
        raise ValueError("fusion replay requires at least one event")
    if len(events) > max_events:
        raise ValueError("fusion replay event capacity exceeded")
    if type(max_input_bytes) is not int or not 1 <= max_input_bytes <= _MAX_BYTES:
        raise ValueError("max_input_bytes must be in 1..256000000")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= _MAX_BYTES:
        raise ValueError("max_output_bytes must be in 1..256000000")
    normalized = []
    for expected_index, event in enumerate(events):
        if not isinstance(event, FusionReplayEvent):
            raise ValueError("events must contain FusionReplayEvent values")
        if event.event_index != expected_index:
            raise ValueError("replay event order/index is not contiguous")
        normalized.append(event.to_dict())
    input_packet = {"config_sha256": config.config_sha256, "events": normalized}
    input_json = _canonical(input_packet)
    if len(input_json.encode("utf-8")) > max_input_bytes:
        raise ValueError("fusion replay input byte capacity exceeded")
    fusion = config.create_fusion()
    states: list[OpticalIMUFusionState] = []
    for event in events:
        if event.kind == "imu":
            state = fusion.ingest_imu(event.value)
        else:
            state = fusion.ingest_optical(event.value)
        if not isinstance(state, OpticalIMUFusionState):
            raise ValueError("fusion returned an invalid state")
        states.append(state)
    output_packet = [fused_attitude_state_to_dict(item) for item in states]
    output_json = _canonical(output_packet)
    if len(output_json.encode("utf-8")) > max_output_bytes:
        raise ValueError("fusion replay output byte capacity exceeded")
    return FusionReplayResult(
        config_sha256=config.config_sha256,
        input_sha256=sha256(input_json.encode("utf-8")).hexdigest(),
        output_sha256=sha256(output_json.encode("utf-8")).hexdigest(),
        event_count=len(events),
        output_count=len(states),
        states=tuple(states),
        hardware_provenance_verified=False,
    )


@dataclass(frozen=True)
class ReferenceOrientation:
    timestamp_ns: int
    clock_domain: str
    quaternion_world_from_body_xyzw: tuple[float, float, float, float]
    reference_sha256: str

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int or not 0 <= self.timestamp_ns <= _MAX_I64:
            raise ValueError("reference timestamp must be a nonnegative int64")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("reference clock domain must be nonempty")
        object.__setattr__(self, "quaternion_world_from_body_xyzw", _quaternion(self.quaternion_world_from_body_xyzw, "reference quaternion"))
        _digest(self.reference_sha256, "reference_sha256")


@dataclass(frozen=True)
class OrientationErrorReport:
    reference_sha256: str
    clock_domain: str
    matched_count: int
    unmatched_state_count: int
    unmatched_reference_count: int
    tracking_count: int
    degraded_count: int
    lost_count: int
    rmse_rad: float
    median_rad: float
    p95_rad: float
    max_rad: float
    rmse_deg: float
    median_deg: float
    p95_deg: float
    max_deg: float
    errors_rad: tuple[float, ...]
    hardware_ground_truth_verified: bool = False


def evaluate_orientation_error(
    states: Sequence[OpticalIMUFusionState],
    references: Sequence[ReferenceOrientation],
    *,
    clock_domain: str,
    max_match_delta_ns: int = 2_000_000,
) -> OrientationErrorReport:
    """Match valid output states to unique references and compute geodesic error."""
    if not isinstance(states, (tuple, list)) or not isinstance(references, (tuple, list)):
        raise ValueError("states and references must be sequences")
    if not states or not references:
        raise ValueError("states and references must be nonempty")
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise ValueError("clock_domain must be nonempty")
    if type(max_match_delta_ns) is not int or not 0 <= max_match_delta_ns <= _MAX_I64:
        raise ValueError("max_match_delta_ns must be a nonnegative int64")
    if any(not isinstance(state, OpticalIMUFusionState) for state in states):
        raise ValueError("states must contain OpticalIMUFusionState")
    if any(not isinstance(ref, ReferenceOrientation) for ref in references):
        raise ValueError("references must contain ReferenceOrientation")
    if any(state.clock_domain != clock_domain for state in states):
        raise ValueError("fusion states contain a mixed clock domain")
    identity = (
        states[0].body_id, states[0].imu_id, states[0].imu_calibration_sha256,
        states[0].clock_mapping_sha256, states[0].optical_pattern_sha256,
        states[0].optical_calibration_sha256,
    )
    for state in states[1:]:
        if (
            state.body_id, state.imu_id, state.imu_calibration_sha256,
            state.clock_mapping_sha256, state.optical_pattern_sha256,
            state.optical_calibration_sha256,
        ) != identity:
            raise ValueError("fusion states contain mixed body/IMU/calibration revisions")
    if any(ref.clock_domain != clock_domain for ref in references):
        raise ValueError("reference poses contain a mixed clock domain")
    ref_times = [ref.timestamp_ns for ref in references]
    if any(b <= a for a, b in zip(ref_times, ref_times[1:])):
        raise ValueError("reference timestamps must be strictly increasing")
    if any(b.timestamp_ns < a.timestamp_ns for a, b in zip(states, states[1:])):
        raise ValueError("fusion states must be timestamp ordered")
    reference_digest = references[0].reference_sha256
    if any(ref.reference_sha256 != reference_digest for ref in references):
        raise ValueError("reference dataset digest changed within report")

    errors: list[float] = []
    used_refs: set[int] = set()
    unmatched_states = 0
    for state in states:
        if state.status == "LOST" or state.quaternion_world_from_body_xyzw is None:
            unmatched_states += 1
            continue
        # Deterministic nearest-neighbor match, one reference can be used once.
        candidates = [
            (abs(ref.timestamp_ns - state.timestamp_ns), index)
            for index, ref in enumerate(references)
            if index not in used_refs and abs(ref.timestamp_ns - state.timestamp_ns) <= max_match_delta_ns
        ]
        if not candidates:
            unmatched_states += 1
            continue
        _, index = min(candidates, key=lambda item: (item[0], item[1]))
        used_refs.add(index)
        q = _quaternion(state.quaternion_world_from_body_xyzw, "fusion quaternion")
        r = references[index].quaternion_world_from_body_xyzw
        dot = abs(sum(q[i] * r[i] for i in range(4)))
        angle = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        errors.append(angle)
    if not errors:
        raise ValueError("no publishable fusion states matched the reference dataset")
    ordered = sorted(errors)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
    return OrientationErrorReport(
        reference_sha256=reference_digest,
        clock_domain=clock_domain,
        matched_count=len(errors),
        unmatched_state_count=unmatched_states,
        unmatched_reference_count=len(references) - len(used_refs),
        tracking_count=sum(s.status == "TRACKING" for s in states),
        degraded_count=sum(s.status == "DEGRADED" for s in states),
        lost_count=sum(s.status == "LOST" for s in states),
        rmse_rad=rmse,
        median_rad=(ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2.0,
        p95_rad=ordered[p95_index],
        max_rad=max(ordered),
        rmse_deg=math.degrees(rmse),
        median_deg=math.degrees((ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2.0),
        p95_deg=math.degrees(ordered[p95_index]),
        max_deg=math.degrees(max(ordered)),
        errors_rad=tuple(errors),
        hardware_ground_truth_verified=False,
    )
