"""Deterministic bounded replay for timestamped IMU and position events.

Replay digests prove reproducibility of declared software inputs/outputs only;
they do not certify sensor authenticity, physical synchronization or truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Sequence

from delayed_position_replay import BoundedDelayedPositionReplay
from imu_calibration import CorrectedIMUVectors, IMUCalibration
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import MetricPositionMeasurement, PositionSourceContract
from translational_eskf import TranslationalErrorState, TranslationalSeed
from translational_propagation import GravityModel

_MAX_EVENTS = 1_000_000
_MAX_BYTES = 256_000_000


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("replay payload must be finite canonical JSON") from exc


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LocalizationReplayConfig:
    seed: TranslationalSeed
    imu_calibration: IMUCalibration
    gravity_model: GravityModel
    position_source_contracts: tuple[PositionSourceContract, ...] = ()
    max_history_frames: int = 256
    max_history_ns: int = 500_000_000
    max_sample_gap_ns: int = 20_000_000
    max_step_ns: int = 5_000_000
    max_substeps: int = 64
    max_position_dead_reckoning_ns: int = 100_000_000
    max_attitude_variance_rad2: float = 0.05
    max_residual_accel_bias_m_s2: float = 2.0
    body_origin_tolerance_m: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.seed, TranslationalSeed):
            raise ValueError("seed must be TranslationalSeed")
        if not isinstance(self.imu_calibration, IMUCalibration):
            raise ValueError("imu_calibration must be IMUCalibration")
        if not isinstance(self.gravity_model, GravityModel):
            raise ValueError("gravity_model must be GravityModel")
        if not isinstance(self.position_source_contracts, tuple) or any(
            not isinstance(item, PositionSourceContract)
            for item in self.position_source_contracts
        ):
            raise ValueError("position_source_contracts must be a tuple of contracts")
        for name in ("max_history_frames", "max_history_ns", "max_sample_gap_ns",
                     "max_step_ns", "max_substeps", "max_position_dead_reckoning_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_attitude_variance_rad2", "max_residual_accel_bias_m_s2",
                     "body_origin_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and nonnegative")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def config_sha256(self) -> str:
        return _sha(_json(asdict(self)))


@dataclass(frozen=True)
class LocalizationReplayEvent:
    event_index: int
    kind: str
    sample: CorrectedIMUVectors | None = None
    attitude: LocalizationIMUAttitudeState | None = None
    measurement: MetricPositionMeasurement | None = None
    max_position_nis: float = 16.0

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be a nonnegative integer")
        if self.kind == "imu":
            if not isinstance(self.sample, CorrectedIMUVectors) or not isinstance(
                self.attitude, LocalizationIMUAttitudeState
            ) or self.measurement is not None:
                raise ValueError("imu event requires sample and attitude only")
            if self.sample.sensor_timestamp_ns != self.attitude.timestamp_ns:
                raise ValueError("IMU sample/attitude timestamps must match")
        elif self.kind == "position":
            if not isinstance(self.measurement, MetricPositionMeasurement) or (
                self.sample is not None or self.attitude is not None
            ):
                raise ValueError("position event requires a measurement only")
            if isinstance(self.max_position_nis, bool) or not isinstance(
                self.max_position_nis, Real
            ) or not math.isfinite(float(self.max_position_nis)) or self.max_position_nis <= 0:
                raise ValueError("max_position_nis must be finite and positive")
        else:
            raise ValueError("unsupported localization replay event kind")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalizationReplayResult:
    config_sha256: str
    input_sha256: str
    output_sha256: str
    event_count: int
    states: tuple[TranslationalErrorState, ...]
    hardware_provenance_verified: bool = False


def replay_localization_events(
    config: LocalizationReplayConfig,
    events: Sequence[LocalizationReplayEvent],
    *,
    max_events: int = 100_000,
    max_input_bytes: int = 64_000_000,
    max_output_bytes: int = 64_000_000,
) -> LocalizationReplayResult:
    """Replay events in arrival order; delayed measurements use acquisition time."""
    if not isinstance(config, LocalizationReplayConfig):
        raise ValueError("config must be LocalizationReplayConfig")
    if not isinstance(events, (tuple, list)) or not events:
        raise ValueError("events must be a nonempty ordered sequence")
    if type(max_events) is not int or not 1 <= max_events <= _MAX_EVENTS:
        raise ValueError("max_events must be in 1..1000000")
    if len(events) > max_events:
        raise ValueError("localization replay event capacity exceeded")
    for name, value in (("max_input_bytes", max_input_bytes), ("max_output_bytes", max_output_bytes)):
        if type(value) is not int or not 1 <= value <= _MAX_BYTES:
            raise ValueError(f"{name} must be in 1..256000000")
    if any(not isinstance(event, LocalizationReplayEvent) for event in events):
        raise ValueError("events must contain LocalizationReplayEvent")
    if any(event.event_index != index for index, event in enumerate(events)):
        raise ValueError("replay event indices must be contiguous and ordered")

    input_json = _json({"config_sha256": config.config_sha256,
                        "events": [event.to_dict() for event in events]})
    if len(input_json.encode("utf-8")) > max_input_bytes:
        raise ValueError("localization replay input byte capacity exceeded")

    owner = BoundedDelayedPositionReplay(
        config.seed, config.imu_calibration, config.gravity_model,
        max_history_frames=config.max_history_frames,
        max_history_ns=config.max_history_ns,
        max_sample_gap_ns=config.max_sample_gap_ns,
        max_step_ns=config.max_step_ns,
        max_substeps=config.max_substeps,
        max_position_dead_reckoning_ns=config.max_position_dead_reckoning_ns,
        max_attitude_variance_rad2=config.max_attitude_variance_rad2,
        max_residual_accel_bias_m_s2=config.max_residual_accel_bias_m_s2,
        body_origin_tolerance_m=config.body_origin_tolerance_m,
        position_source_contracts=config.position_source_contracts,
    )
    states = []
    for event in events:
        if event.kind == "imu":
            state = owner.ingest_imu(event.sample, event.attitude)
        else:
            state = owner.ingest_position_measurement(
                event.measurement, max_position_nis=event.max_position_nis
            )
        states.append(state)
    output_json = _json({"states": [asdict(state) for state in states]})
    if len(output_json.encode("utf-8")) > max_output_bytes:
        raise ValueError("localization replay output byte capacity exceeded")
    return LocalizationReplayResult(
        config_sha256=config.config_sha256,
        input_sha256=_sha(input_json),
        output_sha256=_sha(output_json),
        event_count=len(events),
        states=tuple(states),
    )
