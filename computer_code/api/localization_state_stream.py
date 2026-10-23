"""Versioned unified localization state for application consumers.

This is an additive provider-neutral output contract. It combines a same-time
LocalizationIMUAttitudeState and conditioned TranslationalErrorState only after
identity, frame, clock and revision checks agree. Existing optical PoseStream v1
and fused_attitude_state payloads are unchanged.

The translational covariance is 9x9 over p/v/residual-accelerometer-bias.
Attitude variance remains separate because the current estimator is conditioned
on attitude and does not carry a joint attitude-translation covariance.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

from localization_imu_fusion import LocalizationIMUAttitudeState
from translational_eskf import TranslationalErrorState


_PROTOCOL = "low_cost_mocap_localization"
_SCHEMA_VERSION = 1
_KIND = "localization_state"
_MAX_I64 = (1 << 63) - 1

_REQUIRED = {
    "protocol", "schema_version", "kind",
    "body_id", "imu_id", "map_frame_id",
    "timestamp_ns", "imu_sequence_number", "clock_domain",
    "status", "attitude_status", "translation_status",
    "attitude_source", "translation_source",
    "quaternion_world_from_body_xyzw",
    "position_world_from_body_m", "velocity_world_from_body_m_s",
    "residual_accel_bias_body_m_s2",
    "attitude_variance_rad2", "translational_covariance_p_v_bias",
    "last_localization_timestamp_ns", "last_localization_source_sequence_number",
    "localization_age_ns",
    "last_position_update_timestamp_ns", "last_position_measurement_source",
    "last_position_source_sequence_number", "position_age_ns",
    "clock_mapping_sha256", "imu_calibration_sha256",
    "localization_source_revision_sha256", "localization_calibration_sha256",
    "last_position_source_revision_sha256", "last_position_calibration_sha256",
    "last_position_clock_mapping_sha256", "gravity_model_sha256",
    "position_units", "metric_scale_qualified",
    "joint_attitude_translation_covariance_available",
    "hardware_provenance_verified",
}

_DIGEST_FIELDS = (
    "clock_mapping_sha256",
    "imu_calibration_sha256",
    "localization_source_revision_sha256",
    "localization_calibration_sha256",
    "last_position_source_revision_sha256",
    "last_position_calibration_sha256",
    "last_position_clock_mapping_sha256",
    "gravity_model_sha256",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _i64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _vector(value: object, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numeric values")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _optional_time(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _i64(value, name)


def _validate_covariance(value: object) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError("translational_covariance_p_v_bias must be a 9x9 array")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("translational_covariance_p_v_bias must be a 9x9 array")
        rows.append([
            _finite(item, f"translational_covariance_p_v_bias[{row_index}][{column}]")
            for column, item in enumerate(row)
        ])
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("translational covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-9:
        raise ValueError("translational covariance must be positive semidefinite")
    return rows


def _overall_status(attitude_status: str, translation_status: str) -> str:
    valid = {"TRACKING", "DEGRADED", "LOST"}
    if attitude_status not in valid or translation_status not in valid:
        raise ValueError("unsupported component localization status")
    if "LOST" in {attitude_status, translation_status}:
        return "LOST"
    if attitude_status == "TRACKING" and translation_status == "TRACKING":
        return "TRACKING"
    return "DEGRADED"


def localization_state_to_dict(
    attitude: LocalizationIMUAttitudeState,
    translation: TranslationalErrorState,
) -> dict[str, Any]:
    """Combine same-time attitude and translation states into one strict payload."""
    if not isinstance(attitude, LocalizationIMUAttitudeState):
        raise ValueError("attitude must be LocalizationIMUAttitudeState")
    if not isinstance(translation, TranslationalErrorState):
        raise ValueError("translation must be TranslationalErrorState")

    for label, left, right in (
        ("body", attitude.body_id, translation.body_id),
        ("IMU", attitude.imu_id, translation.imu_id),
        ("map frame", attitude.map_frame_id, translation.map_frame_id),
        ("clock domain", attitude.clock_domain, translation.clock_domain),
        ("IMU calibration", attitude.imu_calibration_sha256, translation.imu_calibration_sha256),
        (
            "localization source revision",
            attitude.localization_source_revision_sha256,
            translation.localization_source_revision_sha256,
        ),
        (
            "localization calibration",
            attitude.localization_calibration_sha256,
            translation.localization_calibration_sha256,
        ),
        ("localization clock mapping", attitude.clock_mapping_sha256, translation.clock_mapping_sha256),
    ):
        if left != right:
            raise ValueError(f"attitude/translation {label} mismatch")
    if attitude.timestamp_ns != translation.timestamp_ns:
        raise ValueError("attitude/translation timestamps must match exactly")

    status = _overall_status(attitude.status, translation.status)
    visible = status != "LOST"
    if visible:
        if (
            attitude.quaternion_world_from_body_xyzw is None
            or attitude.attitude_variance_rad2 is None
            or translation.position_world_from_body_m is None
            or translation.velocity_world_from_body_m_s is None
            or translation.covariance_p_v_bias is None
        ):
            raise ValueError("publishable unified localization requires both component states")

    payload = {
        "protocol": _PROTOCOL,
        "schema_version": _SCHEMA_VERSION,
        "kind": _KIND,
        "body_id": attitude.body_id,
        "imu_id": attitude.imu_id,
        "map_frame_id": attitude.map_frame_id,
        "timestamp_ns": attitude.timestamp_ns,
        "imu_sequence_number": translation.imu_sequence_number,
        "clock_domain": attitude.clock_domain,
        "status": status,
        "attitude_status": attitude.status,
        "translation_status": translation.status,
        "attitude_source": attitude.source,
        "translation_source": translation.source,
        "quaternion_world_from_body_xyzw": (
            list(attitude.quaternion_world_from_body_xyzw) if visible else None
        ),
        "position_world_from_body_m": (
            list(translation.position_world_from_body_m) if visible else None
        ),
        "velocity_world_from_body_m_s": (
            list(translation.velocity_world_from_body_m_s) if visible else None
        ),
        "residual_accel_bias_body_m_s2": (
            list(translation.residual_accel_bias_body_m_s2) if visible else None
        ),
        "attitude_variance_rad2": (
            attitude.attitude_variance_rad2 if visible else None
        ),
        "translational_covariance_p_v_bias": (
            [list(row) for row in translation.covariance_p_v_bias]
            if visible else None
        ),
        "last_localization_timestamp_ns": attitude.last_localization_timestamp_ns,
        "last_localization_source_sequence_number": attitude.last_source_sequence_number,
        "localization_age_ns": attitude.localization_age_ns,
        "last_position_update_timestamp_ns": translation.last_position_update_timestamp_ns,
        "last_position_measurement_source": translation.last_position_measurement_source,
        "last_position_source_sequence_number": translation.last_position_source_sequence_number,
        "position_age_ns": translation.position_age_ns,
        "clock_mapping_sha256": attitude.clock_mapping_sha256,
        "imu_calibration_sha256": attitude.imu_calibration_sha256,
        "localization_source_revision_sha256": attitude.localization_source_revision_sha256,
        "localization_calibration_sha256": attitude.localization_calibration_sha256,
        "last_position_source_revision_sha256": translation.last_position_source_revision_sha256,
        "last_position_calibration_sha256": translation.last_position_calibration_sha256,
        "last_position_clock_mapping_sha256": translation.last_position_clock_mapping_sha256,
        "gravity_model_sha256": translation.gravity_model_sha256,
        "position_units": "meters",
        "metric_scale_qualified": True,
        "joint_attitude_translation_covariance_available": False,
        "hardware_provenance_verified": False,
    }
    return validate_localization_state_payload(payload)


def validate_localization_state_payload(
    payload: object,
    *,
    previous_timestamp_ns: int | None = None,
    previous_imu_sequence_number: int | None = None,
) -> dict[str, Any]:
    """Validate detached unified localization state without mutating consumer state."""
    if not isinstance(payload, dict):
        raise ValueError("localization state payload must be an object")
    keys = set(payload)
    if keys != _REQUIRED:
        raise ValueError(
            "localization state fields mismatch "
            f"(missing={sorted(_REQUIRED-keys)}, unknown={sorted(keys-_REQUIRED)})"
        )
    if (
        payload["protocol"] != _PROTOCOL
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != _SCHEMA_VERSION
        or payload["kind"] != _KIND
    ):
        raise ValueError("unsupported localization state schema")
    if payload["hardware_provenance_verified"] is not False:
        raise ValueError("software localization cannot assert hardware provenance")
    if payload["position_units"] != "meters" or payload["metric_scale_qualified"] is not True:
        raise ValueError("unified localization state requires qualified metric position")
    if payload["joint_attitude_translation_covariance_available"] is not False:
        raise ValueError("conditioned estimator cannot claim joint attitude/translation covariance")

    for name in ("body_id", "imu_id", "map_frame_id", "clock_domain"):
        _text(payload[name], name)
    timestamp = _i64(payload["timestamp_ns"], "timestamp_ns")
    sequence = _i64(payload["imu_sequence_number"], "imu_sequence_number")
    if previous_timestamp_ns is not None and timestamp <= _i64(
        previous_timestamp_ns, "previous_timestamp_ns"
    ):
        raise ValueError("localization state timestamp is duplicate or regressed")
    if previous_imu_sequence_number is not None and sequence <= _i64(
        previous_imu_sequence_number, "previous_imu_sequence_number"
    ):
        raise ValueError("localization state IMU sequence is duplicate or out of order")

    for name in _DIGEST_FIELDS:
        _digest(payload[name], name)

    attitude_status = payload["attitude_status"]
    translation_status = payload["translation_status"]
    expected_status = _overall_status(attitude_status, translation_status)
    if payload["status"] != expected_status:
        raise ValueError("unified status disagrees with component statuses")
    if payload["attitude_source"] not in {
        "LOCALIZATION_CORRECTED",
        "LOCALIZATION_CORRECTED_WITH_ALIGNMENT",
        "IMU_PROPAGATED",
        "LOST",
    }:
        raise ValueError("unsupported unified attitude source")
    if payload["translation_source"] not in {
        "POSITION_SEEDED", "POSITION_CORRECTED", "IMU_PROPAGATED", "LOST",
    }:
        raise ValueError("unsupported unified translation source")

    localization_ts = _optional_time(
        payload["last_localization_timestamp_ns"],
        "last_localization_timestamp_ns",
    )
    localization_seq = _optional_time(
        payload["last_localization_source_sequence_number"],
        "last_localization_source_sequence_number",
    )
    localization_age = _optional_time(payload["localization_age_ns"], "localization_age_ns")
    if localization_ts is None:
        if localization_seq is not None or localization_age is not None:
            raise ValueError("localization sequence/age require a localization timestamp")
    else:
        if localization_seq is None or localization_age is None:
            raise ValueError("localization timestamp requires sequence and age")
        if localization_ts > timestamp or timestamp - localization_ts != localization_age:
            raise ValueError("localization age is inconsistent with state timestamp")

    position_ts = _i64(
        payload["last_position_update_timestamp_ns"],
        "last_position_update_timestamp_ns",
    )
    position_seq = _i64(
        payload["last_position_source_sequence_number"],
        "last_position_source_sequence_number",
    )
    position_age = _i64(payload["position_age_ns"], "position_age_ns")
    if position_ts > timestamp or timestamp - position_ts != position_age:
        raise ValueError("position age is inconsistent with state timestamp")
    if payload["last_position_measurement_source"] not in {"LOCALIZATION", "RTK"}:
        raise ValueError("unsupported last position measurement source")

    dynamic = (
        "quaternion_world_from_body_xyzw",
        "position_world_from_body_m",
        "velocity_world_from_body_m_s",
        "residual_accel_bias_body_m_s2",
        "attitude_variance_rad2",
        "translational_covariance_p_v_bias",
    )
    if expected_status == "LOST":
        if any(payload[name] is not None for name in dynamic):
            raise ValueError("LOST unified localization state must hide dynamic state")
    else:
        quaternion = _vector(payload["quaternion_world_from_body_xyzw"], 4, "quaternion")
        norm = math.sqrt(sum(value*value for value in quaternion))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("unified localization quaternion must be normalized")
        _vector(payload["position_world_from_body_m"], 3, "position_world_from_body_m")
        _vector(payload["velocity_world_from_body_m_s"], 3, "velocity_world_from_body_m_s")
        _vector(payload["residual_accel_bias_body_m_s2"], 3, "residual_accel_bias_body_m_s2")
        variance = _finite(payload["attitude_variance_rad2"], "attitude_variance_rad2")
        if variance < 0:
            raise ValueError("attitude_variance_rad2 must be nonnegative")
        _validate_covariance(payload["translational_covariance_p_v_bias"])

    return {
        key: (
            [list(row) for row in value]
            if key == "translational_covariance_p_v_bias" and value is not None
            else list(value)
            if key in {
                "quaternion_world_from_body_xyzw",
                "position_world_from_body_m",
                "velocity_world_from_body_m_s",
                "residual_accel_bias_body_m_s2",
            } and value is not None
            else value
        )
        for key, value in payload.items()
    }


class LocalizationStateConsumer:
    """Strict single-session consumer for unified localization state payloads."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.body_id: str | None = None
        self.imu_id: str | None = None
        self.map_frame_id: str | None = None
        self.clock_domain: str | None = None
        self.imu_calibration_sha256: str | None = None
        self.localization_source_revision_sha256: str | None = None
        self.localization_calibration_sha256: str | None = None
        self.gravity_model_sha256: str | None = None
        self.clock_mapping_sha256: str | None = None
        self.last_timestamp_ns: int | None = None
        self.last_imu_sequence_number: int | None = None
        self.state: dict[str, Any] | None = None

    def consume(self, payload: object) -> dict[str, Any]:
        checked = validate_localization_state_payload(
            payload,
            previous_timestamp_ns=self.last_timestamp_ns,
            previous_imu_sequence_number=self.last_imu_sequence_number,
        )
        fixed = {
            "body_id": checked["body_id"],
            "imu_id": checked["imu_id"],
            "map_frame_id": checked["map_frame_id"],
            "clock_domain": checked["clock_domain"],
            "imu_calibration_sha256": checked["imu_calibration_sha256"],
            "localization_source_revision_sha256": checked["localization_source_revision_sha256"],
            "localization_calibration_sha256": checked["localization_calibration_sha256"],
            "gravity_model_sha256": checked["gravity_model_sha256"],
            "clock_mapping_sha256": checked["clock_mapping_sha256"],
        }
        for name, value in fixed.items():
            previous = getattr(self, name)
            if previous is not None and previous != value:
                raise ValueError(f"localization state {name} changed without consumer reset")

        # Commit only after all invariants pass.
        for name, value in fixed.items():
            setattr(self, name, value)
        self.last_timestamp_ns = checked["timestamp_ns"]
        self.last_imu_sequence_number = checked["imu_sequence_number"]
        self.state = checked
        return dict(checked)
