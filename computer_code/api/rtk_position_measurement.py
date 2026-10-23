"""RTK FIXED position adaptation into the common MetricPositionMeasurement boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from imu_contract import ClockMapping
from localization_contract import LocalizationQuality
from localization_imu_fusion import LocalizationIMUAttitudeState
from outdoor_sensor_contract import RTKSample
from position_measurement import MetricPositionMeasurement, PositionMeasurementSource
from rtk_coordinates import (
    ENUOrigin,
    antenna_enu_to_body_origin_enu,
    qualified_rtk_position_enu,
)


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite real")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(_real(v, f"{name}[{i}]") for i, v in enumerate(value))


def _canonical_digest(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RTKPositionCalibration:
    receiver_id: str
    body_id: str
    map_frame_id: str
    enu_origin: ENUOrigin
    antenna_lever_arm_body_m: tuple[float, float, float]
    horizontal_accuracy_to_axis_stddev_scale: float
    vertical_accuracy_to_stddev_scale: float
    max_correction_age_ms: float
    max_horizontal_accuracy_m: float
    max_vertical_accuracy_m: float
    min_satellites: int
    source_revision_sha256: str = field(init=False)
    calibration_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_id", _text(self.receiver_id, "receiver_id"))
        object.__setattr__(self, "body_id", _text(self.body_id, "body_id"))
        object.__setattr__(self, "map_frame_id", _text(self.map_frame_id, "map_frame_id", 128))
        if not isinstance(self.enu_origin, ENUOrigin):
            raise ValueError("enu_origin must be ENUOrigin")
        object.__setattr__(
            self, "antenna_lever_arm_body_m",
            _vec3(self.antenna_lever_arm_body_m, "antenna_lever_arm_body_m"),
        )
        for name in (
            "horizontal_accuracy_to_axis_stddev_scale",
            "vertical_accuracy_to_stddev_scale",
        ):
            value = _real(getattr(self, name), name, minimum=0.0)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "max_correction_age_ms",
            "max_horizontal_accuracy_m",
            "max_vertical_accuracy_m",
        ):
            object.__setattr__(self, name, _real(getattr(self, name), name, minimum=0.0))
        if type(self.min_satellites) is not int or self.min_satellites < 1:
            raise ValueError("min_satellites must be a positive integer")

        source_payload = {
            "map_frame_id": self.map_frame_id,
            "datum": self.enu_origin.datum,
            "latitude_deg": self.enu_origin.latitude_deg,
            "longitude_deg": self.enu_origin.longitude_deg,
            "ellipsoid_height_m": self.enu_origin.ellipsoid_height_m,
        }
        source_digest = _canonical_digest(source_payload)
        object.__setattr__(self, "source_revision_sha256", source_digest)
        object.__setattr__(
            self,
            "calibration_sha256",
            _canonical_digest({
                "source_revision_sha256": source_digest,
                "receiver_id": self.receiver_id,
                "body_id": self.body_id,
                "antenna_lever_arm_body_m": list(self.antenna_lever_arm_body_m),
                "horizontal_accuracy_to_axis_stddev_scale": self.horizontal_accuracy_to_axis_stddev_scale,
                "vertical_accuracy_to_stddev_scale": self.vertical_accuracy_to_stddev_scale,
                "max_correction_age_ms": self.max_correction_age_ms,
                "max_horizontal_accuracy_m": self.max_horizontal_accuracy_m,
                "max_vertical_accuracy_m": self.max_vertical_accuracy_m,
                "min_satellites": self.min_satellites,
            }),
        )


def position_measurement_from_rtk(
    sample: RTKSample,
    mapping: ClockMapping,
    calibration: RTKPositionCalibration,
    attitude: LocalizationIMUAttitudeState,
) -> MetricPositionMeasurement:
    """Qualify FIXED RTK, remove antenna lever arm, and bind uncertainty/provenance."""
    if not isinstance(sample, RTKSample):
        raise ValueError("sample must be RTKSample")
    if not isinstance(mapping, ClockMapping):
        raise ValueError("mapping must be ClockMapping")
    if not isinstance(calibration, RTKPositionCalibration):
        raise ValueError("calibration must be RTKPositionCalibration")
    if not isinstance(attitude, LocalizationIMUAttitudeState):
        raise ValueError("attitude must be LocalizationIMUAttitudeState")
    if sample.receiver_id != calibration.receiver_id:
        raise ValueError("RTK receiver identity does not match calibration")
    if sample.antenna_reference_point != "antenna_phase_center":
        raise ValueError("RTK sample must reference the antenna phase center")
    if mapping.source_clock_domain != sample.clock_domain:
        raise ValueError("RTK clock mapping source domain mismatch")
    if mapping.target_clock_domain != attitude.clock_domain:
        raise ValueError("RTK clock mapping target domain mismatch")
    if attitude.body_id != calibration.body_id:
        raise ValueError("attitude body does not match RTK calibration")
    if attitude.map_frame_id != calibration.map_frame_id:
        raise ValueError("attitude map frame does not match RTK ENU frame")
    if (
        attitude.status == "LOST"
        or attitude.quaternion_world_from_body_xyzw is None
        or attitude.attitude_variance_rad2 is None
    ):
        raise ValueError("RTK lever-arm correction requires publishable attitude")

    mapped_timestamp = mapping.map_timestamp(sample.solution_timestamp_ns)
    if mapped_timestamp != attitude.timestamp_ns:
        raise ValueError("RTK solution must align exactly with current attitude timestamp")

    antenna_position = qualified_rtk_position_enu(
        sample,
        calibration.enu_origin,
        max_correction_age_ms=calibration.max_correction_age_ms,
        max_horizontal_accuracy_m=calibration.max_horizontal_accuracy_m,
        max_vertical_accuracy_m=calibration.max_vertical_accuracy_m,
        min_satellites=calibration.min_satellites,
    )
    rotation_world_from_body = Rotation.from_quat(
        attitude.quaternion_world_from_body_xyzw
    ).as_matrix()
    body_position = antenna_enu_to_body_origin_enu(
        antenna_position,
        rotation_world_from_body,
        calibration.antenna_lever_arm_body_m,
    )

    lever_norm = float(np.linalg.norm(calibration.antenna_lever_arm_body_m))
    lever_variance = lever_norm * lever_norm * attitude.attitude_variance_rad2
    horizontal_stddev = math.sqrt(
        (sample.horizontal_accuracy_m * calibration.horizontal_accuracy_to_axis_stddev_scale) ** 2
        + lever_variance
    )
    vertical_stddev = math.sqrt(
        (sample.vertical_accuracy_m * calibration.vertical_accuracy_to_stddev_scale) ** 2
        + lever_variance
    )
    if not all(math.isfinite(v) and v > 0.0 for v in (horizontal_stddev, vertical_stddev)):
        raise ValueError("RTK position uncertainty is invalid")

    return MetricPositionMeasurement(
        source=PositionMeasurementSource.RTK,
        body_id=calibration.body_id,
        map_frame_id=calibration.map_frame_id,
        timestamp_ns=mapped_timestamp,
        source_sequence_number=sample.sequence_number,
        clock_domain=attitude.clock_domain,
        position_world_from_body_m=body_position,
        position_stddev_m=(horizontal_stddev, horizontal_stddev, vertical_stddev),
        quality=LocalizationQuality.TRACKING,
        source_revision_sha256=calibration.source_revision_sha256,
        measurement_calibration_sha256=calibration.calibration_sha256,
        clock_mapping_sha256=mapping.mapping_sha256,
    )
