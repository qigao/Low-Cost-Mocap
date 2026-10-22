"""Bounded deterministic IMU gyro attitude propagation.

This is a propagation primitive, not an optical-IMU fusion estimator. It only
produces orientation relative to a supplied initial attitude; it never estimates
global position/velocity and does not claim drift-free orientation. Corrected
gyro samples are expressed in body axes, and quaternions use world-from-body
rotation with XYZW component order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from imu_calibration import CorrectedIMUVectors


_MAX_I64 = (1 << 63) - 1


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _nonnegative_i64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _normalize_quaternion(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("orientation_world_from_body_xyzw must contain four values")
    q = tuple(_finite_real(v, "quaternion component") for v in value)
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("initial quaternion must have a finite nonzero norm")
    return tuple(v / norm for v in q)


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _delta_quaternion(
    angular_rate_rad_s: tuple[float, float, float],
    dt_s: float,
) -> tuple[float, float, float, float]:
    wx, wy, wz = angular_rate_rad_s
    magnitude = math.sqrt(wx * wx + wy * wy + wz * wz)
    angle = magnitude * dt_s
    if not math.isfinite(angle):
        raise ValueError("gyro integration angle is nonfinite")
    if magnitude <= 1e-15 or abs(angle) <= 1e-12:
        return _normalize_quaternion(
            (0.5 * wx * dt_s, 0.5 * wy * dt_s, 0.5 * wz * dt_s, 1.0)
        )
    half_angle = 0.5 * angle
    scale = math.sin(half_angle) / magnitude
    return (
        wx * scale,
        wy * scale,
        wz * scale,
        math.cos(half_angle),
    )


@dataclass(frozen=True)
class AttitudePropagationState:
    imu_id: str
    sequence_number: int
    sensor_timestamp_ns: int
    clock_domain: str
    calibration_revision: str
    calibration_sha256: str
    orientation_world_from_body_xyzw: tuple[float, float, float, float]
    status: str
    integrated_interval_ns: int
    substeps: int
    source: str = "imu_gyro_propagation"
    global_position_m: None = None
    global_velocity_m_s: None = None
    global_position_observable: bool = False

    def __post_init__(self) -> None:
        _text(self.imu_id, "imu_id")
        _text(self.clock_domain, "clock_domain")
        _text(self.calibration_revision, "calibration_revision")
        _nonnegative_i64(self.sequence_number, "sequence_number")
        _nonnegative_i64(self.sensor_timestamp_ns, "sensor_timestamp_ns")
        _nonnegative_i64(self.integrated_interval_ns, "integrated_interval_ns")
        _nonnegative_i64(self.substeps, "substeps")
        if (
            not isinstance(self.calibration_sha256, str)
            or len(self.calibration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.calibration_sha256)
        ):
            raise ValueError("calibration_sha256 must be lowercase SHA-256")
        if self.status not in {"INITIALIZED", "IMU_PROPAGATED"}:
            raise ValueError("unsupported attitude propagation status")
        if self.source != "imu_gyro_propagation":
            raise ValueError("unexpected propagation source")
        if self.global_position_m is not None or self.global_velocity_m_s is not None:
            raise ValueError("gyro-only propagation must not publish global position or velocity")
        if self.global_position_observable is not False:
            raise ValueError("gyro-only propagation cannot claim global position observability")
        object.__setattr__(
            self,
            "orientation_world_from_body_xyzw",
            _normalize_quaternion(self.orientation_world_from_body_xyzw),
        )


class BoundedIMUAttitudePropagator:
    """Integrate corrected body-frame gyro rates with bounded time steps.

    The first sample establishes the timestamp/rate baseline but does not move
    the initial attitude. Later intervals use the trapezoidal mean of adjacent
    angular-rate samples, then exact quaternion increments for the bounded
    piecewise-constant rate. State commits only after the entire interval passes.
    """

    def __init__(
        self,
        *,
        imu_id: str,
        clock_domain: str,
        calibration_revision: str,
        calibration_sha256: str,
        initial_orientation_world_from_body_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        max_sample_gap_ns: int = 20_000_000,
        max_step_ns: int = 5_000_000,
        max_substeps: int = 64,
    ) -> None:
        self.imu_id = _text(imu_id, "imu_id")
        self.clock_domain = _text(clock_domain, "clock_domain")
        self.calibration_revision = _text(calibration_revision, "calibration_revision")
        if (
            not isinstance(calibration_sha256, str)
            or len(calibration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in calibration_sha256)
        ):
            raise ValueError("calibration_sha256 must be lowercase SHA-256")
        self.calibration_sha256 = calibration_sha256
        self._initial_orientation = _normalize_quaternion(initial_orientation_world_from_body_xyzw)
        self.max_sample_gap_ns = _nonnegative_i64(max_sample_gap_ns, "max_sample_gap_ns")
        self.max_step_ns = _nonnegative_i64(max_step_ns, "max_step_ns")
        self.max_substeps = _nonnegative_i64(max_substeps, "max_substeps")
        if self.max_sample_gap_ns < 1 or self.max_step_ns < 1 or self.max_substeps < 1:
            raise ValueError("sample gap, integration step and substep limits must be positive")
        if self.max_step_ns > self.max_sample_gap_ns:
            raise ValueError("max_step_ns must not exceed max_sample_gap_ns")
        self._last_sample: CorrectedIMUVectors | None = None
        self._orientation = self._initial_orientation
        self._state: AttitudePropagationState | None = None

    @property
    def state(self) -> AttitudePropagationState | None:
        return self._state

    def advance(self, sample: CorrectedIMUVectors) -> AttitudePropagationState:
        if not isinstance(sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if sample.imu_id != self.imu_id:
            raise ValueError("IMU device ID mismatch")
        if sample.clock_domain != self.clock_domain:
            raise ValueError("IMU clock-domain mismatch")
        if sample.calibration_revision != self.calibration_revision:
            raise ValueError("IMU calibration revision mismatch")
        if sample.calibration_sha256 != self.calibration_sha256:
            raise ValueError("IMU calibration digest mismatch")
        timestamp = _nonnegative_i64(sample.sensor_timestamp_ns, "sensor_timestamp_ns")
        sequence = _nonnegative_i64(sample.sequence_number, "sequence_number")
        rate = tuple(_finite_real(v, "gyro_body_rad_s") for v in sample.gyro_body_rad_s)

        previous = self._last_sample
        if previous is None:
            state = AttitudePropagationState(
                imu_id=sample.imu_id,
                sequence_number=sequence,
                sensor_timestamp_ns=timestamp,
                clock_domain=sample.clock_domain,
                calibration_revision=sample.calibration_revision,
                calibration_sha256=sample.calibration_sha256,
                orientation_world_from_body_xyzw=self._orientation,
                status="INITIALIZED",
                integrated_interval_ns=0,
                substeps=0,
            )
            self._last_sample = sample
            self._state = state
            return state

        if sequence <= previous.sequence_number:
            raise ValueError("IMU sequence is duplicate or out of order")
        if timestamp <= previous.sensor_timestamp_ns:
            raise ValueError("IMU acquisition timestamp is duplicate or regressed")
        dt_ns = timestamp - previous.sensor_timestamp_ns
        if dt_ns > self.max_sample_gap_ns:
            raise ValueError("IMU sample gap exceeds configured limit")
        substeps = (dt_ns + self.max_step_ns - 1) // self.max_step_ns
        if substeps > self.max_substeps:
            raise ValueError("IMU interval exceeds bounded integration substep capacity")

        mean_rate = tuple(
            0.5 * (previous.gyro_body_rad_s[i] + rate[i])
            for i in range(3)
        )
        candidate = self._orientation
        base_step_ns, remainder = divmod(dt_ns, substeps)
        for index in range(substeps):
            step_ns = base_step_ns + (1 if index < remainder else 0)
            step_s = step_ns / 1_000_000_000.0
            delta = _delta_quaternion(mean_rate, step_s)
            candidate = _normalize_quaternion(_quaternion_multiply(candidate, delta))
        state = AttitudePropagationState(
            imu_id=sample.imu_id,
            sequence_number=sequence,
            sensor_timestamp_ns=timestamp,
            clock_domain=sample.clock_domain,
            calibration_revision=sample.calibration_revision,
            calibration_sha256=sample.calibration_sha256,
            orientation_world_from_body_xyzw=candidate,
            status="IMU_PROPAGATED",
            integrated_interval_ns=dt_ns,
            substeps=substeps,
        )
        # Commit only after all validation and numerical operations succeeded.
        self._orientation = candidate
        self._last_sample = sample
        self._state = state
        return state

    def reset(self) -> None:
        """Clear the time baseline and return to the explicit initial attitude."""
        self._last_sample = None
        self._orientation = self._initial_orientation
        self._state = None
