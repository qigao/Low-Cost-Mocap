"""Provider-neutral localization/IMU attitude fusion using the existing bounded core.

MapPoseObservation corrections are adapted into the proven
OpticalIMUAttitudeFusion state machine without exposing optical-pattern naming
at this API boundary. This vertical slice fuses orientation only; position
remains outside the state until a separately qualified translational estimator
is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from imu_contract import ClockMapping
from imu_propagation import AttitudePropagationState
from localization_contract import LocalizationQuality, MapPoseObservation
from optical_imu_fusion import (
    OpticalAttitudeMeasurement,
    OpticalIMUAttitudeFusion,
    OpticalIMUFusionState,
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite real")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class LocalizationIMUAttitudeState:
    body_id: str
    imu_id: str
    timestamp_ns: int
    clock_domain: str
    quaternion_world_from_body_xyzw: tuple[float, float, float, float] | None
    attitude_variance_rad2: float | None
    status: str
    source: str
    last_localization_timestamp_ns: int | None
    last_source_sequence_number: int | None
    localization_age_ns: int | None
    clock_mapping_sha256: str
    imu_calibration_sha256: str
    localization_source_revision_sha256: str
    localization_calibration_sha256: str
    global_position_m: None = None
    global_velocity_m_s: None = None
    global_position_observable: bool = False

    def __post_init__(self) -> None:
        _text(self.body_id, "body_id")
        _text(self.imu_id, "imu_id")
        _text(self.clock_domain, "clock_domain")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative int64")
        if self.status not in {"TRACKING", "DEGRADED", "LOST"}:
            raise ValueError("invalid fusion status")
        if self.source not in {
            "LOCALIZATION_CORRECTED",
            "LOCALIZATION_CORRECTED_WITH_ALIGNMENT",
            "IMU_PROPAGATED",
            "LOST",
        }:
            raise ValueError("invalid fusion source")
        if self.status == "LOST":
            if (
                self.quaternion_world_from_body_xyzw is not None
                or self.attitude_variance_rad2 is not None
                or self.source != "LOST"
            ):
                raise ValueError("LOST state must hide attitude and use LOST source")
        else:
            if (
                self.quaternion_world_from_body_xyzw is None
                or self.attitude_variance_rad2 is None
            ):
                raise ValueError("publishable state requires attitude and variance")
            _real(self.attitude_variance_rad2, "attitude_variance_rad2", minimum=0.0)
        if (
            self.global_position_m is not None
            or self.global_velocity_m_s is not None
            or self.global_position_observable is not False
        ):
            raise ValueError("attitude-only fusion cannot publish global position")
        for name in (
            "clock_mapping_sha256",
            "imu_calibration_sha256",
            "localization_source_revision_sha256",
            "localization_calibration_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in (
            "last_localization_timestamp_ns",
            "last_source_sequence_number",
            "localization_age_ns",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative int64 or None")


def localization_attitude_measurement(
    observation: MapPoseObservation,
    *,
    allow_degraded: bool = False,
    degraded_variance_scale: float = 4.0,
    variance_floor_rad2: float = 1e-8,
) -> OpticalAttitudeMeasurement:
    """Adapt one current localization pose into the existing attitude core."""
    if not isinstance(observation, MapPoseObservation):
        raise ValueError("observation must be MapPoseObservation")
    if not isinstance(allow_degraded, bool):
        raise ValueError("allow_degraded must be boolean")
    degraded_variance_scale = _real(
        degraded_variance_scale,
        "degraded_variance_scale",
        minimum=1.0,
    )
    variance_floor_rad2 = _real(
        variance_floor_rad2,
        "variance_floor_rad2",
        minimum=0.0,
    )
    if variance_floor_rad2 <= 0.0:
        raise ValueError("variance_floor_rad2 must be positive")

    if observation.quality is LocalizationQuality.DEGRADED:
        if not allow_degraded:
            raise ValueError("DEGRADED localization correction is not enabled")
        scale = degraded_variance_scale
    elif observation.quality is LocalizationQuality.TRACKING:
        scale = 1.0
    else:
        raise ValueError("localization pose quality is not usable for correction")

    maximum_stddev = max(observation.orientation_stddev_rad)
    variance = max(
        variance_floor_rad2,
        maximum_stddev * maximum_stddev * scale,
    )
    return OpticalAttitudeMeasurement(
        body_id=observation.tracked_frame_id,
        timestamp_ns=observation.timestamp_ns,
        clock_domain=observation.clock_domain,
        quaternion_world_from_body_xyzw=observation.orientation_xyzw,
        frame_id=observation.source_sequence_number,
        pattern_sha256=observation.map_revision_sha256,
        calibration_sha256=observation.calibration_sha256,
        quality="MEASURED",
        attitude_variance_rad2=variance,
    )


class LocalizationIMUAttitudeFusion:
    """Reuse the bounded optical/IMU attitude core for MapPoseObservation input."""

    _SOURCE_MAP = {
        "OPTICAL_CORRECTED": "LOCALIZATION_CORRECTED",
        "OPTICAL_CORRECTED_WITH_ALIGNMENT": "LOCALIZATION_CORRECTED_WITH_ALIGNMENT",
        "IMU_PROPAGATED": "IMU_PROPAGATED",
        "LOST": "LOST",
    }

    def __init__(
        self,
        *,
        body_id: str,
        imu_id: str,
        imu_clock_domain: str,
        imu_calibration_revision: str,
        imu_calibration_sha256: str,
        localization_source_revision_sha256: str,
        localization_calibration_sha256: str,
        clock_mapping: ClockMapping,
        gyro_noise_density_rad_s_sqrt_hz,
        default_attitude_variance_rad2: float = 0.0025,
        max_dropout_ns: int = 100_000_000,
        max_innovation_rad: float = 1.2,
        max_alignment_angular_rate_rad_s: float = 20.0,
        history_capacity: int = 256,
    ) -> None:
        self.body_id = _text(body_id, "body_id")
        self.imu_id = _text(imu_id, "imu_id")
        self.source_revision = _digest(
            localization_source_revision_sha256,
            "localization_source_revision_sha256",
        )
        self.localization_calibration = _digest(
            localization_calibration_sha256,
            "localization_calibration_sha256",
        )
        self._core = OpticalIMUAttitudeFusion(
            body_id=self.body_id,
            imu_id=self.imu_id,
            imu_clock_domain=imu_clock_domain,
            imu_calibration_revision=imu_calibration_revision,
            imu_calibration_sha256=imu_calibration_sha256,
            optical_pattern_sha256=self.source_revision,
            optical_calibration_sha256=self.localization_calibration,
            clock_mapping=clock_mapping,
            gyro_noise_density_rad_s_sqrt_hz=gyro_noise_density_rad_s_sqrt_hz,
            optical_variance_rad2=default_attitude_variance_rad2,
            max_dropout_ns=max_dropout_ns,
            max_innovation_rad=max_innovation_rad,
            max_alignment_angular_rate_rad_s=max_alignment_angular_rate_rad_s,
            history_capacity=history_capacity,
        )
        self.state: LocalizationIMUAttitudeState | None = None

    def _wrap(self, state: OpticalIMUFusionState) -> LocalizationIMUAttitudeState:
        if not isinstance(state, OpticalIMUFusionState):
            raise ValueError("attitude core returned invalid state")
        if (
            state.optical_pattern_sha256 != self.source_revision
            or state.optical_calibration_sha256
            != self.localization_calibration
        ):
            raise ValueError("attitude core localization revision mismatch")
        wrapped = LocalizationIMUAttitudeState(
            body_id=state.body_id,
            imu_id=state.imu_id,
            timestamp_ns=state.timestamp_ns,
            clock_domain=state.clock_domain,
            quaternion_world_from_body_xyzw=state.quaternion_world_from_body_xyzw,
            attitude_variance_rad2=state.attitude_variance_rad2,
            status=state.status,
            source=self._SOURCE_MAP[state.source],
            last_localization_timestamp_ns=state.last_optical_timestamp_ns,
            last_source_sequence_number=state.last_optical_frame_id,
            localization_age_ns=state.optical_age_ns,
            clock_mapping_sha256=state.clock_mapping_sha256,
            imu_calibration_sha256=state.imu_calibration_sha256,
            localization_source_revision_sha256=self.source_revision,
            localization_calibration_sha256=self.localization_calibration,
        )
        self.state = wrapped
        return wrapped

    def ingest_imu(
        self, state: AttitudePropagationState
    ) -> LocalizationIMUAttitudeState:
        return self._wrap(self._core.ingest_imu(state))

    def ingest_localization(
        self,
        observation: MapPoseObservation,
        *,
        allow_reseed_after_lost: bool = False,
        allow_degraded: bool = False,
        degraded_variance_scale: float = 4.0,
        variance_floor_rad2: float = 1e-8,
    ) -> LocalizationIMUAttitudeState:
        if not isinstance(allow_reseed_after_lost, bool):
            raise ValueError("allow_reseed_after_lost must be boolean")
        measurement = localization_attitude_measurement(
            observation,
            allow_degraded=allow_degraded,
            degraded_variance_scale=degraded_variance_scale,
            variance_floor_rad2=variance_floor_rad2,
        )
        return self._wrap(
            self._core.ingest_optical(
                measurement,
                allow_reseed_after_lost=allow_reseed_after_lost,
            )
        )
