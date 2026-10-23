"""Exact-epoch interpolation between two real IMU/attitude samples.

This module creates an observation at a timestamp strictly inside a retained
sample interval without fabricating an IMU sequence number. The returned value
retains both bracketing source sequence numbers and all calibration/clock
provenance. It is an interpolation primitive only; replay integration must
still split the estimator interval and apply the measurement transactionally.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from imu_calibration import CorrectedIMUVectors
from localization_imu_fusion import LocalizationIMUAttitudeState


@dataclass(frozen=True)
class InterpolatedIMUObservationEpoch:
    timestamp_ns: int
    left_sequence_number: int
    right_sequence_number: int
    imu_id: str
    clock_domain: str
    calibration_sha256: str
    gyro_body_rad_s: tuple[float, float, float]
    specific_force_body_m_s2: tuple[float, float, float]
    body_id: str
    map_frame_id: str
    quaternion_world_from_body_xyzw: tuple[float, float, float, float]
    attitude_variance_rad2: float
    localization_source_revision_sha256: str
    localization_calibration_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if (
            type(self.left_sequence_number) is not int
            or type(self.right_sequence_number) is not int
            or self.left_sequence_number < 0
            or self.right_sequence_number <= self.left_sequence_number
        ):
            raise ValueError("bracketing source sequence numbers must be strictly increasing")
        for name in ("imu_id", "clock_domain", "body_id", "map_frame_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        for name in (
            "calibration_sha256",
            "localization_source_revision_sha256",
            "localization_calibration_sha256",
            "clock_mapping_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        for name in ("gyro_body_rad_s", "specific_force_body_m_s2"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        q = np.asarray(self.quaternion_world_from_body_xyzw, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("quaternion must contain four finite values")
        norm = float(np.linalg.norm(q))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError("interpolated quaternion must be normalized")
        object.__setattr__(self, "quaternion_world_from_body_xyzw", tuple(float(v) for v in q))
        if (
            isinstance(self.attitude_variance_rad2, bool)
            or not isinstance(self.attitude_variance_rad2, (int, float))
            or not math.isfinite(float(self.attitude_variance_rad2))
            or self.attitude_variance_rad2 < 0.0
        ):
            raise ValueError("attitude_variance_rad2 must be finite and nonnegative")


def interpolate_imu_observation_epoch(
    left_sample: CorrectedIMUVectors,
    left_attitude: LocalizationIMUAttitudeState,
    right_sample: CorrectedIMUVectors,
    right_attitude: LocalizationIMUAttitudeState,
    timestamp_ns: int,
) -> InterpolatedIMUObservationEpoch:
    """Interpolate one strict interior epoch; endpoints remain real samples."""
    pairs = ((left_sample, left_attitude), (right_sample, right_attitude))
    for sample, attitude in pairs:
        if not isinstance(sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if not isinstance(attitude, LocalizationIMUAttitudeState):
            raise ValueError("attitude must be LocalizationIMUAttitudeState")
        if sample.sensor_timestamp_ns != attitude.timestamp_ns:
            raise ValueError("sample/attitude timestamps must match")
        if attitude.status == "LOST" or attitude.quaternion_world_from_body_xyzw is None:
            raise ValueError("interpolation requires publishable endpoint attitudes")

    if right_sample.sensor_timestamp_ns <= left_sample.sensor_timestamp_ns:
        raise ValueError("sample timestamps must be strictly increasing")
    if right_sample.sequence_number <= left_sample.sequence_number:
        raise ValueError("sample sequence numbers must be strictly increasing")
    if type(timestamp_ns) is not int or not (
        left_sample.sensor_timestamp_ns < timestamp_ns < right_sample.sensor_timestamp_ns
    ):
        raise ValueError("interpolation timestamp must be strictly inside the sample interval")

    invariant_pairs = (
        (left_sample.imu_id, right_sample.imu_id, "IMU identity"),
        (left_sample.clock_domain, right_sample.clock_domain, "IMU clock domain"),
        (left_sample.calibration_sha256, right_sample.calibration_sha256, "IMU calibration"),
        (left_attitude.body_id, right_attitude.body_id, "body identity"),
        (left_attitude.map_frame_id, right_attitude.map_frame_id, "map frame"),
        (left_attitude.imu_id, right_attitude.imu_id, "attitude IMU identity"),
        (left_attitude.clock_domain, right_attitude.clock_domain, "attitude clock domain"),
        (left_attitude.imu_calibration_sha256, right_attitude.imu_calibration_sha256, "attitude IMU calibration"),
        (left_attitude.localization_source_revision_sha256, right_attitude.localization_source_revision_sha256, "localization source revision"),
        (left_attitude.localization_calibration_sha256, right_attitude.localization_calibration_sha256, "localization calibration"),
        (left_attitude.clock_mapping_sha256, right_attitude.clock_mapping_sha256, "localization clock mapping"),
    )
    for left, right, label in invariant_pairs:
        if left != right:
            raise ValueError(f"bracketing {label} mismatch")
    if left_sample.calibration_sha256 != left_attitude.imu_calibration_sha256:
        raise ValueError("left sample/attitude calibration mismatch")
    if right_sample.calibration_sha256 != right_attitude.imu_calibration_sha256:
        raise ValueError("right sample/attitude calibration mismatch")

    span = right_sample.sensor_timestamp_ns - left_sample.sensor_timestamp_ns
    fraction = (timestamp_ns - left_sample.sensor_timestamp_ns) / span

    def linear(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((1.0 - fraction) * a + fraction * b for a, b in zip(left, right))

    rotations = Rotation.from_quat([
        left_attitude.quaternion_world_from_body_xyzw,
        right_attitude.quaternion_world_from_body_xyzw,
    ])
    quaternion = Slerp([0.0, 1.0], rotations)([fraction]).as_quat()[0]
    variance_left = float(left_attitude.attitude_variance_rad2)
    variance_right = float(right_attitude.attitude_variance_rad2)
    return InterpolatedIMUObservationEpoch(
        timestamp_ns=timestamp_ns,
        left_sequence_number=left_sample.sequence_number,
        right_sequence_number=right_sample.sequence_number,
        imu_id=left_sample.imu_id,
        clock_domain=left_sample.clock_domain,
        calibration_sha256=left_sample.calibration_sha256,
        gyro_body_rad_s=linear(left_sample.gyro_body_rad_s, right_sample.gyro_body_rad_s),
        specific_force_body_m_s2=linear(
            left_sample.specific_force_body_m_s2,
            right_sample.specific_force_body_m_s2,
        ),
        body_id=left_attitude.body_id,
        map_frame_id=left_attitude.map_frame_id,
        quaternion_world_from_body_xyzw=tuple(float(value) for value in quaternion),
        attitude_variance_rad2=(1.0 - fraction) * variance_left + fraction * variance_right,
        localization_source_revision_sha256=left_attitude.localization_source_revision_sha256,
        localization_calibration_sha256=left_attitude.localization_calibration_sha256,
        clock_mapping_sha256=left_attitude.clock_mapping_sha256,
    )
