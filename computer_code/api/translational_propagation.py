"""Bounded accelerometer translation propagation in an explicitly defined map frame.

This is a deterministic dead-reckoning primitive, not a full INS/ESKF. It
requires:
- a metric localization seed with explicit initial velocity uncertainty,
- a gravity vector bound to the same map frame,
- corrected specific force in body axes,
- a same-timestamp provider-neutral fused attitude state,
- an IMU calibration whose sensor origin is the body origin.

A nonzero IMU lever arm is rejected because rotational acceleration
compensation requires angular acceleration terms that are not yet modeled.
Position/velocity are hidden after the configured dead-reckoning horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from imu_contract import ClockMapping
from localization_contract import LocalizationQuality, MapPoseObservation
from localization_imu_fusion import LocalizationIMUAttitudeState


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
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vec3(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, float, float]:
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


def _canonical_digest(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GravityModel:
    """Gravity acceleration expressed in one localization/map frame."""

    frame_id: str
    gravity_world_m_s2: tuple[float, float, float]
    model_id: str
    revision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id", 128))
        gravity = _vec3(self.gravity_world_m_s2, "gravity_world_m_s2")
        magnitude = math.sqrt(sum(value * value for value in gravity))
        # This engine currently targets terrestrial indoor/outdoor localization.
        # Reject zero/implausible gravity rather than silently disabling it.
        if not 5.0 <= magnitude <= 15.0:
            raise ValueError("gravity magnitude must be within terrestrial range [5, 15] m/s^2")
        object.__setattr__(self, "gravity_world_m_s2", gravity)
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", 128))
        object.__setattr__(
            self,
            "revision_sha256",
            _canonical_digest({
                "frame_id": self.frame_id,
                "gravity_world_m_s2": list(self.gravity_world_m_s2),
                "model_id": self.model_id,
            }),
        )


@dataclass(frozen=True)
class TranslationalSeed:
    """Explicit initial translational state in the IMU/fusion clock domain."""

    body_id: str
    imu_id: str
    map_frame_id: str
    timestamp_ns: int
    source_sequence_number: int
    clock_domain: str
    position_world_from_body_m: tuple[float, float, float]
    velocity_world_from_body_m_s: tuple[float, float, float]
    residual_accel_bias_body_m_s2: tuple[float, float, float]
    position_stddev_m: tuple[float, float, float]
    velocity_stddev_m_s: tuple[float, float, float]
    residual_accel_bias_stddev_m_s2: tuple[float, float, float]
    localization_source_revision_sha256: str
    localization_calibration_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "body_id", _text(self.body_id, "body_id"))
        object.__setattr__(self, "imu_id", _text(self.imu_id, "imu_id"))
        object.__setattr__(self, "map_frame_id", _text(self.map_frame_id, "map_frame_id", 128))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        _i64(self.timestamp_ns, "timestamp_ns")
        _i64(self.source_sequence_number, "source_sequence_number")
        for name in (
            "position_world_from_body_m",
            "velocity_world_from_body_m_s",
            "residual_accel_bias_body_m_s2",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        for name in (
            "position_stddev_m",
            "velocity_stddev_m_s",
            "residual_accel_bias_stddev_m_s2",
        ):
            object.__setattr__(
                self,
                name,
                _vec3(getattr(self, name), name, nonnegative=True),
            )
        for name in (
            "localization_source_revision_sha256",
            "localization_calibration_sha256",
            "clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)


def translational_seed_from_localization(
    observation: MapPoseObservation,
    mapping: ClockMapping,
    *,
    imu_id: str,
    expected_imu_clock_domain: str,
    initial_velocity_world_m_s: tuple[float, float, float],
    velocity_stddev_m_s: tuple[float, float, float],
    residual_accel_bias_body_m_s2: tuple[float, float, float],
    residual_accel_bias_stddev_m_s2: tuple[float, float, float],
) -> TranslationalSeed:
    """Map one qualified metric localization pose into the IMU clock.

    Velocity and residual accelerometer bias are deliberately explicit caller
    inputs. A single position observation does not make either observable.
    """
    if not isinstance(observation, MapPoseObservation):
        raise ValueError("observation must be MapPoseObservation")
    if observation.quality not in {
        LocalizationQuality.TRACKING,
        LocalizationQuality.DEGRADED,
    }:
        raise ValueError("localization seed quality is not usable")
    if observation.position_unit != "meters" or not observation.metric_scale_qualified:
        raise ValueError("translation seed requires qualified metric localization")
    if not isinstance(mapping, ClockMapping):
        raise ValueError("mapping must be ClockMapping")
    if mapping.source_clock_domain != observation.clock_domain:
        raise ValueError("clock mapping source does not match localization clock")
    target_clock = _text(expected_imu_clock_domain, "expected_imu_clock_domain")
    if mapping.target_clock_domain != target_clock:
        raise ValueError("clock mapping target does not match IMU clock")
    return TranslationalSeed(
        body_id=observation.tracked_frame_id,
        imu_id=_text(imu_id, "imu_id"),
        map_frame_id=observation.map_frame_id,
        timestamp_ns=mapping.map_timestamp(observation.timestamp_ns),
        source_sequence_number=observation.source_sequence_number,
        clock_domain=target_clock,
        position_world_from_body_m=observation.position_xyz,
        velocity_world_from_body_m_s=_vec3(
            initial_velocity_world_m_s,
            "initial_velocity_world_m_s",
        ),
        residual_accel_bias_body_m_s2=_vec3(
            residual_accel_bias_body_m_s2,
            "residual_accel_bias_body_m_s2",
        ),
        position_stddev_m=observation.position_stddev,
        velocity_stddev_m_s=_vec3(
            velocity_stddev_m_s,
            "velocity_stddev_m_s",
            nonnegative=True,
        ),
        residual_accel_bias_stddev_m_s2=_vec3(
            residual_accel_bias_stddev_m_s2,
            "residual_accel_bias_stddev_m_s2",
            nonnegative=True,
        ),
        localization_source_revision_sha256=observation.map_revision_sha256,
        localization_calibration_sha256=observation.calibration_sha256,
        clock_mapping_sha256=mapping.mapping_sha256,
    )


@dataclass(frozen=True)
class TranslationalPropagationState:
    body_id: str
    imu_id: str
    map_frame_id: str
    timestamp_ns: int
    imu_sequence_number: int
    clock_domain: str
    position_world_from_body_m: tuple[float, float, float] | None
    velocity_world_from_body_m_s: tuple[float, float, float] | None
    residual_accel_bias_body_m_s2: tuple[float, float, float]
    position_variance_m2: tuple[float, float, float] | None
    velocity_variance_m2_s2: tuple[float, float, float] | None
    residual_accel_bias_variance_m2_s4: tuple[float, float, float]
    status: str
    source: str
    last_position_update_timestamp_ns: int
    position_age_ns: int
    localization_source_revision_sha256: str
    localization_calibration_sha256: str
    imu_calibration_sha256: str
    gravity_model_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        for name in ("body_id", "imu_id", "map_frame_id", "clock_domain"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "timestamp_ns",
            "imu_sequence_number",
            "last_position_update_timestamp_ns",
            "position_age_ns",
        ):
            _i64(getattr(self, name), name)
        if self.last_position_update_timestamp_ns > self.timestamp_ns:
            raise ValueError("last position update cannot be newer than state timestamp")
        if self.position_age_ns != self.timestamp_ns - self.last_position_update_timestamp_ns:
            raise ValueError("position_age_ns is inconsistent with timestamps")
        object.__setattr__(
            self,
            "residual_accel_bias_body_m_s2",
            _vec3(
                self.residual_accel_bias_body_m_s2,
                "residual_accel_bias_body_m_s2",
            ),
        )
        object.__setattr__(
            self,
            "residual_accel_bias_variance_m2_s4",
            _vec3(
                self.residual_accel_bias_variance_m2_s4,
                "residual_accel_bias_variance_m2_s4",
                nonnegative=True,
            ),
        )
        if self.status not in {"TRACKING", "DEGRADED", "LOST"}:
            raise ValueError("invalid translational propagation status")
        if self.source not in {"POSITION_SEEDED", "IMU_PROPAGATED", "LOST"}:
            raise ValueError("invalid translational propagation source")
        if self.status == "LOST":
            if (
                self.position_world_from_body_m is not None
                or self.velocity_world_from_body_m_s is not None
                or self.position_variance_m2 is not None
                or self.velocity_variance_m2_s2 is not None
                or self.source != "LOST"
            ):
                raise ValueError("LOST translation state must hide position/velocity")
        else:
            if (
                self.position_world_from_body_m is None
                or self.velocity_world_from_body_m_s is None
                or self.position_variance_m2 is None
                or self.velocity_variance_m2_s2 is None
            ):
                raise ValueError("publishable translation state requires position/velocity and variance")
            object.__setattr__(
                self,
                "position_world_from_body_m",
                _vec3(self.position_world_from_body_m, "position_world_from_body_m"),
            )
            object.__setattr__(
                self,
                "velocity_world_from_body_m_s",
                _vec3(self.velocity_world_from_body_m_s, "velocity_world_from_body_m_s"),
            )
            object.__setattr__(
                self,
                "position_variance_m2",
                _vec3(self.position_variance_m2, "position_variance_m2", nonnegative=True),
            )
            object.__setattr__(
                self,
                "velocity_variance_m2_s2",
                _vec3(self.velocity_variance_m2_s2, "velocity_variance_m2_s2", nonnegative=True),
            )
        for name in (
            "localization_source_revision_sha256",
            "localization_calibration_sha256",
            "imu_calibration_sha256",
            "gravity_model_sha256",
            "clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)


class BoundedIMUTranslationPropagator:
    """Trapezoidal specific-force propagation with bounded dead reckoning."""

    def __init__(
        self,
        seed: TranslationalSeed,
        imu_calibration: IMUCalibration,
        gravity_model: GravityModel,
        *,
        max_sample_gap_ns: int = 20_000_000,
        max_step_ns: int = 5_000_000,
        max_substeps: int = 64,
        max_position_dead_reckoning_ns: int = 100_000_000,
        max_attitude_variance_rad2: float = 0.05,
        body_origin_tolerance_m: float = 1e-9,
    ) -> None:
        if not isinstance(seed, TranslationalSeed):
            raise ValueError("seed must be TranslationalSeed")
        if not isinstance(imu_calibration, IMUCalibration):
            raise ValueError("imu_calibration must be IMUCalibration")
        if not isinstance(gravity_model, GravityModel):
            raise ValueError("gravity_model must be GravityModel")
        if seed.imu_id != imu_calibration.imu_id:
            raise ValueError("translation seed IMU ID does not match calibration")
        if seed.clock_domain != imu_calibration.imu_clock_domain:
            raise ValueError("translation seed clock does not match IMU calibration")
        if seed.map_frame_id != gravity_model.frame_id:
            raise ValueError("gravity frame does not match translation map frame")
        tolerance = _real(body_origin_tolerance_m, "body_origin_tolerance_m", minimum=0.0)
        lever = np.asarray(imu_calibration.translation_body_from_imu_m, dtype=np.float64)
        if float(np.linalg.norm(lever)) > tolerance:
            raise ValueError(
                "nonzero IMU lever arm requires rotational acceleration compensation"
            )
        self.max_gap = _i64(max_sample_gap_ns, "max_sample_gap_ns")
        self.max_step = _i64(max_step_ns, "max_step_ns")
        self.max_substeps = _i64(max_substeps, "max_substeps")
        self.max_dead_reckoning = _i64(
            max_position_dead_reckoning_ns,
            "max_position_dead_reckoning_ns",
        )
        if min(self.max_gap, self.max_step, self.max_substeps, self.max_dead_reckoning) < 1:
            raise ValueError("gap/step/substep/dead-reckoning limits must be positive")
        if self.max_step > self.max_gap:
            raise ValueError("max_step_ns must not exceed max_sample_gap_ns")
        self.max_attitude_variance = _real(
            max_attitude_variance_rad2,
            "max_attitude_variance_rad2",
            minimum=0.0,
        )
        if self.max_attitude_variance <= 0.0:
            raise ValueError("max_attitude_variance_rad2 must be positive")

        self.seed = seed
        self.calibration = imu_calibration
        self.gravity = gravity_model
        self._position = np.asarray(seed.position_world_from_body_m, dtype=np.float64)
        self._velocity = np.asarray(seed.velocity_world_from_body_m_s, dtype=np.float64)
        self._bias = np.asarray(seed.residual_accel_bias_body_m_s2, dtype=np.float64)
        self._position_var = np.square(np.asarray(seed.position_stddev_m, dtype=np.float64))
        self._velocity_var = np.square(np.asarray(seed.velocity_stddev_m_s, dtype=np.float64))
        self._bias_var = np.square(np.asarray(seed.residual_accel_bias_stddev_m_s2, dtype=np.float64))
        self._last_sample: CorrectedIMUVectors | None = None
        self._last_attitude: LocalizationIMUAttitudeState | None = None
        self._state: TranslationalPropagationState | None = None
        self._expired = False

    @property
    def state(self) -> TranslationalPropagationState | None:
        return self._state

    def _validate_pair(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> None:
        if not isinstance(sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if not isinstance(attitude, LocalizationIMUAttitudeState):
            raise ValueError("attitude must be LocalizationIMUAttitudeState")
        if sample.imu_id != self.seed.imu_id or attitude.imu_id != self.seed.imu_id:
            raise ValueError("IMU identity mismatch")
        if attitude.body_id != self.seed.body_id:
            raise ValueError("attitude body does not match translation seed")
        if (
            sample.clock_domain != self.seed.clock_domain
            or attitude.clock_domain != self.seed.clock_domain
        ):
            raise ValueError("sample/attitude clock does not match translation seed")
        if sample.sensor_timestamp_ns != attitude.timestamp_ns:
            raise ValueError("specific force and attitude timestamps must match exactly")
        if sample.calibration_sha256 != self.calibration.calibration_sha256:
            raise ValueError("corrected IMU sample calibration mismatch")
        if attitude.imu_calibration_sha256 != self.calibration.calibration_sha256:
            raise ValueError("attitude IMU calibration mismatch")
        if attitude.localization_source_revision_sha256 != self.seed.localization_source_revision_sha256:
            raise ValueError("attitude localization source revision mismatch")
        if attitude.localization_calibration_sha256 != self.seed.localization_calibration_sha256:
            raise ValueError("attitude localization calibration mismatch")
        if attitude.clock_mapping_sha256 != self.seed.clock_mapping_sha256:
            raise ValueError("attitude clock mapping revision mismatch")
        if attitude.status == "LOST" or attitude.quaternion_world_from_body_xyzw is None:
            raise ValueError("translation propagation requires a publishable attitude")
        if (
            attitude.attitude_variance_rad2 is None
            or attitude.attitude_variance_rad2 > self.max_attitude_variance
        ):
            raise ValueError("attitude uncertainty exceeds translation propagation limit")

    def _world_acceleration(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = Rotation.from_quat(
            attitude.quaternion_world_from_body_xyzw
        ).as_matrix()
        specific_force_body = (
            np.asarray(sample.specific_force_body_m_s2, dtype=np.float64)
            - self._bias
        )
        acceleration = (
            rotation @ specific_force_body
            + np.asarray(self.gravity.gravity_world_m_s2, dtype=np.float64)
        )
        if not np.isfinite(acceleration).all():
            raise ValueError("world acceleration is non-finite")

        noise_body_var_density = np.square(
            np.asarray(self.calibration.accel_noise_density_m_s2_sqrt_hz, dtype=np.float64)
        )
        noise_world_density = np.diag(
            rotation @ np.diag(noise_body_var_density) @ rotation.T
        )
        # Attitude uncertainty is an instantaneous acceleration variance,
        # not a white-noise density. Keep it dimensionally separate from the
        # accelerometer noise density so the integration powers of dt remain
        # correct.
        attitude_accel_variance = np.full(
            3,
            float(np.dot(specific_force_body, specific_force_body))
            * attitude.attitude_variance_rad2,
            dtype=np.float64,
        )
        if not (
            np.isfinite(noise_world_density).all()
            and np.isfinite(attitude_accel_variance).all()
        ):
            raise ValueError("acceleration uncertainty is non-finite")
        return acceleration, noise_world_density, attitude_accel_variance

    def _publish(
        self,
        *,
        timestamp_ns: int,
        sequence_number: int,
        status: str,
        source: str,
    ) -> TranslationalPropagationState:
        age = timestamp_ns - self.seed.timestamp_ns
        visible = status != "LOST"
        state = TranslationalPropagationState(
            body_id=self.seed.body_id,
            imu_id=self.seed.imu_id,
            map_frame_id=self.seed.map_frame_id,
            timestamp_ns=timestamp_ns,
            imu_sequence_number=sequence_number,
            clock_domain=self.seed.clock_domain,
            position_world_from_body_m=(
                tuple(float(v) for v in self._position) if visible else None
            ),
            velocity_world_from_body_m_s=(
                tuple(float(v) for v in self._velocity) if visible else None
            ),
            residual_accel_bias_body_m_s2=tuple(float(v) for v in self._bias),
            position_variance_m2=(
                tuple(float(v) for v in self._position_var) if visible else None
            ),
            velocity_variance_m2_s2=(
                tuple(float(v) for v in self._velocity_var) if visible else None
            ),
            residual_accel_bias_variance_m2_s4=tuple(
                float(v) for v in self._bias_var
            ),
            status=status,
            source=source,
            last_position_update_timestamp_ns=self.seed.timestamp_ns,
            position_age_ns=age,
            localization_source_revision_sha256=self.seed.localization_source_revision_sha256,
            localization_calibration_sha256=self.seed.localization_calibration_sha256,
            imu_calibration_sha256=self.calibration.calibration_sha256,
            gravity_model_sha256=self.gravity.revision_sha256,
            clock_mapping_sha256=self.seed.clock_mapping_sha256,
        )
        self._state = state
        return state

    def advance(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> TranslationalPropagationState:
        if self._expired:
            raise ValueError("translation propagation expired; explicit position reseed required")
        self._validate_pair(sample, attitude)

        previous_sample = self._last_sample
        previous_attitude = self._last_attitude
        if previous_sample is None:
            if sample.sensor_timestamp_ns != self.seed.timestamp_ns:
                raise ValueError("first IMU sample must align exactly with translation seed")
            self._last_sample = sample
            self._last_attitude = attitude
            return self._publish(
                timestamp_ns=sample.sensor_timestamp_ns,
                sequence_number=sample.sequence_number,
                status="TRACKING",
                source="POSITION_SEEDED",
            )

        if sample.sequence_number <= previous_sample.sequence_number:
            raise ValueError("IMU sequence is duplicate or out of order")
        if sample.sensor_timestamp_ns <= previous_sample.sensor_timestamp_ns:
            raise ValueError("IMU timestamp is duplicate or regressed")
        dt_ns = sample.sensor_timestamp_ns - previous_sample.sensor_timestamp_ns
        if dt_ns > self.max_gap:
            raise ValueError("IMU sample gap exceeds translation propagation limit")
        substeps = (dt_ns + self.max_step - 1) // self.max_step
        if substeps > self.max_substeps:
            raise ValueError("translation interval exceeds bounded substep capacity")

        (
            previous_acceleration,
            previous_noise,
            previous_attitude_accel_var,
        ) = self._world_acceleration(previous_sample, previous_attitude)
        (
            current_acceleration,
            current_noise,
            current_attitude_accel_var,
        ) = self._world_acceleration(sample, attitude)
        mean_acceleration = 0.5 * (previous_acceleration + current_acceleration)
        mean_noise_density = 0.5 * (previous_noise + current_noise)
        mean_attitude_accel_var = 0.5 * (
            previous_attitude_accel_var + current_attitude_accel_var
        )
        if not np.isfinite(mean_acceleration).all():
            raise ValueError("mean world acceleration is non-finite")

        candidate_position = self._position.copy()
        candidate_velocity = self._velocity.copy()
        candidate_position_var = self._position_var.copy()
        candidate_velocity_var = self._velocity_var.copy()
        candidate_bias_var = self._bias_var.copy()

        base_step_ns, remainder = divmod(dt_ns, substeps)
        bias_rw = np.asarray(
            self.calibration.accel_bias_random_walk_m_s2_per_sqrt_s,
            dtype=np.float64,
        )
        # Orientation and force are linearly interpolated only at the interval
        # level in this first bounded primitive. Substeps bound integration
        # arithmetic and process-noise accumulation; they do not invent new
        # sensor observations.
        for index in range(substeps):
            step_ns = base_step_ns + (1 if index < remainder else 0)
            dt = step_ns / 1_000_000_000.0
            candidate_position = (
                candidate_position
                + candidate_velocity * dt
                + 0.5 * mean_acceleration * dt * dt
            )
            candidate_velocity = candidate_velocity + mean_acceleration * dt

            # Diagonal conservative process model. We intentionally do not call
            # this a full covariance/ESKF because cross-covariances are absent.
            world_bias_var = np.max(candidate_bias_var)
            bounded_accel_state_var = (
                mean_attitude_accel_var + world_bias_var
            )
            candidate_position_var = (
                candidate_position_var
                + candidate_velocity_var * dt * dt
                + mean_noise_density * (dt ** 3) / 3.0
                + 0.25 * bounded_accel_state_var * (dt ** 4)
            )
            candidate_velocity_var = (
                candidate_velocity_var
                + mean_noise_density * dt
                + bounded_accel_state_var * dt * dt
            )
            candidate_bias_var = candidate_bias_var + np.square(bias_rw) * dt

        if not all(
            np.isfinite(value).all()
            for value in (
                candidate_position,
                candidate_velocity,
                candidate_position_var,
                candidate_velocity_var,
                candidate_bias_var,
            )
        ):
            raise ValueError("translation propagation produced non-finite state")

        # Commit only after the full interval validates.
        self._position = candidate_position
        self._velocity = candidate_velocity
        self._position_var = candidate_position_var
        self._velocity_var = candidate_velocity_var
        self._bias_var = candidate_bias_var
        self._last_sample = sample
        self._last_attitude = attitude

        age = sample.sensor_timestamp_ns - self.seed.timestamp_ns
        if age > self.max_dead_reckoning:
            self._expired = True
            return self._publish(
                timestamp_ns=sample.sensor_timestamp_ns,
                sequence_number=sample.sequence_number,
                status="LOST",
                source="LOST",
            )
        return self._publish(
            timestamp_ns=sample.sensor_timestamp_ns,
            sequence_number=sample.sequence_number,
            status="DEGRADED",
            source="IMU_PROPAGATED",
        )
