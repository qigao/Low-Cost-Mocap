"""Clock-aligned provider-neutral metric position measurements.

This module defines a common position-update boundary for localization today
and RTK later. It carries only metric position, diagonal uncertainty, source
identity/revisions, acquisition sequence, and a clock-alignment revision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from imu_contract import ClockMapping
from localization_contract import LocalizationQuality, MapPoseObservation


_MAX_I64 = (1 << 63) - 1


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _i64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vec3(value: object, name: str, *, nonnegative: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(_real(item, f"{name}[{index}]") for index, item in enumerate(value))
    if nonnegative and any(item < 0.0 for item in result):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


class PositionMeasurementSource(str, Enum):
    LOCALIZATION = "LOCALIZATION"
    RTK = "RTK"


@dataclass(frozen=True)
class PositionSourceContract:
    """Configuration-pinned provenance for one absolute-position provider."""

    source: PositionMeasurementSource
    source_revision_sha256: str
    measurement_calibration_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self, "source", PositionMeasurementSource(self.source)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported position source contract") from exc
        for name in (
            "source_revision_sha256",
            "measurement_calibration_sha256",
            "clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)


@dataclass(frozen=True)
class MetricPositionMeasurement:
    source: PositionMeasurementSource
    body_id: str
    map_frame_id: str
    timestamp_ns: int
    source_sequence_number: int
    clock_domain: str
    position_world_from_body_m: tuple[float, float, float]
    position_stddev_m: tuple[float, float, float]
    quality: LocalizationQuality
    source_revision_sha256: str
    measurement_calibration_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "source", PositionMeasurementSource(self.source))
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported position measurement source") from exc
        try:
            quality = LocalizationQuality(self.quality)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported position measurement quality") from exc
        if quality not in {LocalizationQuality.TRACKING, LocalizationQuality.DEGRADED}:
            raise ValueError("position measurement quality must be TRACKING or DEGRADED")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "body_id", _text(self.body_id, "body_id"))
        object.__setattr__(self, "map_frame_id", _text(self.map_frame_id, "map_frame_id", 128))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        _i64(self.timestamp_ns, "timestamp_ns")
        _i64(self.source_sequence_number, "source_sequence_number")
        object.__setattr__(
            self,
            "position_world_from_body_m",
            _vec3(self.position_world_from_body_m, "position_world_from_body_m"),
        )
        stddev = _vec3(self.position_stddev_m, "position_stddev_m", nonnegative=True)
        if any(value <= 0.0 for value in stddev):
            raise ValueError("position_stddev_m values must be positive")
        object.__setattr__(self, "position_stddev_m", stddev)
        for name in (
            "source_revision_sha256",
            "measurement_calibration_sha256",
            "clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)


def position_measurement_from_localization(
    observation: MapPoseObservation,
    mapping: ClockMapping,
    *,
    expected_target_clock_domain: str,
    allow_degraded: bool = False,
    degraded_variance_scale: float = 4.0,
    stddev_floor_m: float = 1e-6,
) -> MetricPositionMeasurement:
    """Adapt one current metric localization pose into the position boundary."""
    if not isinstance(observation, MapPoseObservation):
        raise ValueError("observation must be MapPoseObservation")
    if observation.position_unit != "meters" or not observation.metric_scale_qualified:
        raise ValueError("position correction requires qualified metric localization")
    if not isinstance(mapping, ClockMapping):
        raise ValueError("mapping must be ClockMapping")
    if mapping.source_clock_domain != observation.clock_domain:
        raise ValueError("clock mapping source does not match localization clock")
    target = _text(expected_target_clock_domain, "expected_target_clock_domain")
    if mapping.target_clock_domain != target:
        raise ValueError("clock mapping target does not match position filter clock")
    if type(allow_degraded) is not bool:
        raise ValueError("allow_degraded must be boolean")
    degraded_variance_scale = _real(
        degraded_variance_scale,
        "degraded_variance_scale",
        minimum=1.0,
    )
    stddev_floor_m = _real(stddev_floor_m, "stddev_floor_m", minimum=0.0)
    if stddev_floor_m <= 0.0:
        raise ValueError("stddev_floor_m must be positive")

    if observation.quality is LocalizationQuality.TRACKING:
        variance_scale = 1.0
    elif observation.quality is LocalizationQuality.DEGRADED:
        if not allow_degraded:
            raise ValueError("DEGRADED localization position correction is not enabled")
        variance_scale = degraded_variance_scale
    else:
        raise ValueError("localization quality is not usable for position correction")

    stddev_scale = math.sqrt(variance_scale)
    stddev = tuple(
        max(stddev_floor_m, value * stddev_scale)
        for value in observation.position_stddev
    )
    return MetricPositionMeasurement(
        source=PositionMeasurementSource.LOCALIZATION,
        body_id=observation.tracked_frame_id,
        map_frame_id=observation.map_frame_id,
        timestamp_ns=mapping.map_timestamp(observation.timestamp_ns),
        source_sequence_number=observation.source_sequence_number,
        clock_domain=target,
        position_world_from_body_m=observation.position_xyz,
        position_stddev_m=stddev,
        quality=observation.quality,
        source_revision_sha256=observation.map_revision_sha256,
        measurement_calibration_sha256=observation.calibration_sha256,
        clock_mapping_sha256=mapping.mapping_sha256,
    )
