"""Immutable IMU-to-body extrinsic and sensor calibration contract.

This module validates calibration data and rotates bias-corrected sensor vectors
into body axes. It does not estimate calibration from hardware, certify physical
accuracy, or compensate the IMU lever arm's rotational acceleration at the body
origin; that compensation belongs in a specified estimator model.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np

from imu_contract import IMUSample

_IDENTITY3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_SCHEMA_VERSION = 1


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _vector3(value: object, name: str, *, nonnegative: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(_real(item, name) for item in value)
    if nonnegative and any(item < 0 for item in result):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _rotation3(value: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("rotation_body_from_imu must be a 3x3 matrix")
    rows = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("rotation_body_from_imu must be a 3x3 matrix")
        rows.append(tuple(_real(item, "rotation_body_from_imu") for item in row))
    matrix = tuple(rows)
    tolerance = 1e-7
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tolerance:
                raise ValueError("rotation_body_from_imu must be orthonormal")
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError("rotation_body_from_imu must have determinant +1")
    return matrix


def _matrix3(value: object, name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    rows = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix")
        rows.append(tuple(_real(item, name) for item in row))
    matrix = np.asarray(rows, dtype=np.float64)
    condition = float(np.linalg.cond(matrix))
    if (
        not np.isfinite(matrix).all()
        or not math.isfinite(condition)
        or condition > 1_000.0
    ):
        raise ValueError(f"{name} must be finite and well-conditioned (condition <= 1000)")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _rotate(
    rotation: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


@dataclass(frozen=True)
class IMUCalibration:
    """Revisioned calibration; transforms IMU-frame vectors into body axes.

    translation_body_from_imu_m is the IMU origin expressed in body coordinates.
    Corrected vectors remain measured at the IMU origin; no lever-arm correction
    to the body's origin is applied here.
    """

    imu_id: str
    revision: str
    imu_clock_domain: str
    rotation_body_from_imu: tuple[tuple[float, float, float], ...]
    translation_body_from_imu_m: tuple[float, float, float]
    gyro_bias_rad_s: tuple[float, float, float]
    accel_bias_m_s2: tuple[float, float, float]
    gyro_noise_density_rad_s_sqrt_hz: tuple[float, float, float]
    accel_noise_density_m_s2_sqrt_hz: tuple[float, float, float]
    calibration_method: str
    gyro_correction_matrix: tuple[tuple[float, float, float], ...] = _IDENTITY3
    accel_correction_matrix: tuple[tuple[float, float, float], ...] = _IDENTITY3
    valid_from_timestamp_ns: int = 0
    valid_until_timestamp_ns: int | None = None
    calibration_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "imu_id", _text(self.imu_id, "imu_id"))
        object.__setattr__(self, "revision", _text(self.revision, "revision"))
        object.__setattr__(self, "imu_clock_domain", _text(self.imu_clock_domain, "imu_clock_domain"))
        object.__setattr__(self, "calibration_method", _text(self.calibration_method, "calibration_method"))
        object.__setattr__(self, "rotation_body_from_imu", _rotation3(self.rotation_body_from_imu))
        object.__setattr__(self, "gyro_correction_matrix", _matrix3(self.gyro_correction_matrix, "gyro_correction_matrix"))
        object.__setattr__(self, "accel_correction_matrix", _matrix3(self.accel_correction_matrix, "accel_correction_matrix"))
        object.__setattr__(self, "translation_body_from_imu_m", _vector3(self.translation_body_from_imu_m, "translation_body_from_imu_m"))
        object.__setattr__(self, "gyro_bias_rad_s", _vector3(self.gyro_bias_rad_s, "gyro_bias_rad_s"))
        object.__setattr__(self, "accel_bias_m_s2", _vector3(self.accel_bias_m_s2, "accel_bias_m_s2"))
        object.__setattr__(self, "gyro_noise_density_rad_s_sqrt_hz", _vector3(self.gyro_noise_density_rad_s_sqrt_hz, "gyro_noise_density_rad_s_sqrt_hz", nonnegative=True))
        object.__setattr__(self, "accel_noise_density_m_s2_sqrt_hz", _vector3(self.accel_noise_density_m_s2_sqrt_hz, "accel_noise_density_m_s2_sqrt_hz", nonnegative=True))
        if type(self.valid_from_timestamp_ns) is not int or not 0 <= self.valid_from_timestamp_ns <= (1 << 63) - 1:
            raise ValueError("valid_from_timestamp_ns must be a nonnegative signed-64-bit integer")
        if self.valid_until_timestamp_ns is not None:
            if (
                type(self.valid_until_timestamp_ns) is not int
                or not self.valid_from_timestamp_ns < self.valid_until_timestamp_ns <= (1 << 63) - 1
            ):
                raise ValueError("valid_until_timestamp_ns must be after valid_from and within signed-64-bit range")
        payload = self._digest_payload()
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if not isinstance(self.calibration_sha256, str):
            raise ValueError("calibration_sha256 must be a string")
        if self.calibration_sha256:
            if self.calibration_sha256 != digest:
                raise ValueError("calibration_sha256 does not match calibration contents")
        else:
            object.__setattr__(self, "calibration_sha256", digest)

    def _digest_payload(self) -> dict[str, object]:
        return {
            "imu_id": self.imu_id,
            "revision": self.revision,
            "imu_clock_domain": self.imu_clock_domain,
            "rotation_body_from_imu": self.rotation_body_from_imu,
            "translation_body_from_imu_m": self.translation_body_from_imu_m,
            "gyro_bias_rad_s": self.gyro_bias_rad_s,
            "accel_bias_m_s2": self.accel_bias_m_s2,
            "gyro_noise_density_rad_s_sqrt_hz": self.gyro_noise_density_rad_s_sqrt_hz,
            "accel_noise_density_m_s2_sqrt_hz": self.accel_noise_density_m_s2_sqrt_hz,
            "calibration_method": self.calibration_method,
            "gyro_correction_matrix": self.gyro_correction_matrix,
            "accel_correction_matrix": self.accel_correction_matrix,
            "schema_version": _SCHEMA_VERSION,
            "valid_from_timestamp_ns": self.valid_from_timestamp_ns,
            "valid_until_timestamp_ns": self.valid_until_timestamp_ns,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the canonical versioned calibration object including its digest."""
        return {**self._digest_payload(), "calibration_sha256": self.calibration_sha256}

    @classmethod
    def from_dict(cls, raw: object) -> "IMUCalibration":
        """Parse a strict JSON-like object; unknown and missing fields fail closed."""
        if not isinstance(raw, dict):
            raise ValueError("IMU calibration must be an object")
        required = {
            "schema_version", "imu_id", "revision", "imu_clock_domain",
            "rotation_body_from_imu", "translation_body_from_imu_m",
            "gyro_bias_rad_s", "accel_bias_m_s2",
            "gyro_noise_density_rad_s_sqrt_hz",
            "accel_noise_density_m_s2_sqrt_hz", "calibration_method",
            "valid_from_timestamp_ns", "valid_until_timestamp_ns",
            "calibration_sha256",
        }
        optional = {"gyro_correction_matrix", "accel_correction_matrix"}
        unknown = set(raw) - required - optional
        missing = required - set(raw)
        if unknown:
            raise ValueError("unknown IMU calibration fields: " + ", ".join(sorted(unknown)))
        if missing:
            raise ValueError("missing IMU calibration fields: " + ", ".join(sorted(missing)))
        if type(raw["schema_version"]) is not int or raw["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported IMU calibration schema_version")
        return cls(
            imu_id=raw["imu_id"],
            revision=raw["revision"],
            imu_clock_domain=raw["imu_clock_domain"],
            rotation_body_from_imu=raw["rotation_body_from_imu"],
            translation_body_from_imu_m=raw["translation_body_from_imu_m"],
            gyro_bias_rad_s=raw["gyro_bias_rad_s"],
            accel_bias_m_s2=raw["accel_bias_m_s2"],
            gyro_noise_density_rad_s_sqrt_hz=raw["gyro_noise_density_rad_s_sqrt_hz"],
            accel_noise_density_m_s2_sqrt_hz=raw["accel_noise_density_m_s2_sqrt_hz"],
            calibration_method=raw["calibration_method"],
            gyro_correction_matrix=raw.get("gyro_correction_matrix", _IDENTITY3),
            accel_correction_matrix=raw.get("accel_correction_matrix", _IDENTITY3),
            valid_from_timestamp_ns=raw["valid_from_timestamp_ns"],
            valid_until_timestamp_ns=raw["valid_until_timestamp_ns"],
            calibration_sha256=raw["calibration_sha256"],
        )

    def validate_sample(self, sample: IMUSample) -> None:
        if not isinstance(sample, IMUSample):
            raise ValueError("sample must be IMUSample")
        sample.require_usable()
        if sample.imu_id != self.imu_id:
            raise ValueError("IMU calibration device ID mismatch")
        if sample.clock_domain != self.imu_clock_domain:
            raise ValueError("IMU calibration clock-domain mismatch")
        if sample.calibration_revision != self.revision:
            raise ValueError("IMU calibration revision mismatch")
        if sample.sensor_timestamp_ns < self.valid_from_timestamp_ns:
            raise ValueError("IMU calibration is not yet valid for this sample")
        if (
            self.valid_until_timestamp_ns is not None
            and sample.sensor_timestamp_ns >= self.valid_until_timestamp_ns
        ):
            raise ValueError("IMU calibration has expired for this sample")


@dataclass(frozen=True)
class CorrectedIMUVectors:
    imu_id: str
    sequence_number: int
    sensor_timestamp_ns: int
    clock_domain: str
    calibration_revision: str
    calibration_sha256: str
    gyro_body_rad_s: tuple[float, float, float]
    specific_force_body_m_s2: tuple[float, float, float]
    measurement_reference_point: str = "imu_origin"


def correct_imu_sample(sample: IMUSample, calibration: IMUCalibration) -> CorrectedIMUVectors:
    """Bias-correct and rotate one sample; does not estimate motion or lever-arm terms."""
    if not isinstance(calibration, IMUCalibration):
        raise ValueError("calibration must be IMUCalibration")
    calibration.validate_sample(sample)
    gyro_unscaled = tuple(
        sample.gyro_rad_s[i] - calibration.gyro_bias_rad_s[i]
        for i in range(3)
    )
    force_unscaled = tuple(
        sample.accel_m_s2[i] - calibration.accel_bias_m_s2[i]
        for i in range(3)
    )
    gyro_sensor = _rotate(calibration.gyro_correction_matrix, gyro_unscaled)
    force_sensor = _rotate(calibration.accel_correction_matrix, force_unscaled)
    gyro_body = _rotate(calibration.rotation_body_from_imu, gyro_sensor)
    force_body = _rotate(calibration.rotation_body_from_imu, force_sensor)
    if not all(math.isfinite(value) for value in (*gyro_body, *force_body)):
        raise ValueError("corrected IMU vectors are nonfinite")
    return CorrectedIMUVectors(
        imu_id=sample.imu_id,
        sequence_number=sample.sequence_number,
        sensor_timestamp_ns=sample.sensor_timestamp_ns,
        clock_domain=sample.clock_domain,
        calibration_revision=calibration.revision,
        calibration_sha256=calibration.calibration_sha256,
        gyro_body_rad_s=gyro_body,
        specific_force_body_m_s2=force_body,
    )
