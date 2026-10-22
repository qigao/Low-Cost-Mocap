"""Versioned wire contract for fused optical/IMU attitude states.

This is an additive payload kind inside the existing low_cost_mocap_stream
envelope. Existing optical asset_description and pose_frame payloads retain
their schema and semantics. Older consumers fail closed on this new kind.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any


_REQUIRED = {
    "protocol", "schema_version", "kind", "body_id", "imu_id",
    "timestamp_ns", "clock_domain", "quaternion_world_from_body_xyzw",
    "attitude_variance_rad2", "status", "source",
    "last_optical_timestamp_ns", "last_optical_frame_id", "optical_age_ns",
    "clock_mapping_sha256", "imu_calibration_sha256",
    "optical_pattern_sha256", "optical_calibration_sha256",
    "global_position_m", "global_velocity_m_s", "global_position_observable",
    "hardware_provenance_verified",
}
_DIGEST_FIELDS = (
    "clock_mapping_sha256", "imu_calibration_sha256",
    "optical_pattern_sha256", "optical_calibration_sha256",
)
_MAX_I64 = (1 << 63) - 1


def _time(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def validate_fused_attitude_payload(
    payload: object,
    *,
    known_body_ids: set[str],
    expected_pattern_sha256: str,
    expected_optical_calibration_sha256: str | None = None,
    previous_timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Validate a detached fused-attitude payload before consumer state advances."""
    if not isinstance(payload, dict):
        raise ValueError("fused attitude payload must be an object")
    keys = set(payload)
    if keys != _REQUIRED:
        missing = sorted(_REQUIRED - keys)
        unknown = sorted(keys - _REQUIRED)
        raise ValueError(f"fused attitude fields mismatch (missing={missing}, unknown={unknown})")
    if (
        payload["protocol"] != "low_cost_mocap_pose"
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "fused_attitude_state"
    ):
        raise ValueError("unsupported fused-attitude schema")
    if payload["hardware_provenance_verified"] is not False:
        raise ValueError("software fusion cannot assert hardware provenance")
    body_id = _text(payload["body_id"], "body_id")
    if body_id not in known_body_ids:
        raise ValueError("fused attitude references an unknown body")
    _text(payload["imu_id"], "imu_id")
    _text(payload["clock_domain"], "clock_domain")
    timestamp = _time(payload["timestamp_ns"], "timestamp_ns")
    if previous_timestamp_ns is not None and timestamp < _time(previous_timestamp_ns, "previous_timestamp_ns"):
        raise ValueError("fused attitude timestamp is stale or regressed")
    for name in _DIGEST_FIELDS:
        _digest(payload[name], name)
    if payload["optical_pattern_sha256"] != expected_pattern_sha256:
        raise ValueError("fused attitude pattern revision disagrees with asset")
    if (
        expected_optical_calibration_sha256 is not None
        and payload["optical_calibration_sha256"] != expected_optical_calibration_sha256
    ):
        raise ValueError("fused attitude optical calibration revision changed")
    status = payload["status"]
    source = payload["source"]
    if status not in {"TRACKING", "DEGRADED", "LOST"}:
        raise ValueError("unsupported fused-attitude status")
    if source not in {"OPTICAL_CORRECTED", "OPTICAL_CORRECTED_WITH_ALIGNMENT", "IMU_PROPAGATED", "LOST"}:
        raise ValueError("unsupported fused-attitude source")
    if (status == "LOST") != (source == "LOST"):
        raise ValueError("LOST status and source must agree")
    if status == "TRACKING" and source != "OPTICAL_CORRECTED":
        raise ValueError("TRACKING requires a current optical correction")
    if status == "DEGRADED" and source not in {"IMU_PROPAGATED", "OPTICAL_CORRECTED_WITH_ALIGNMENT"}:
        raise ValueError("DEGRADED requires bounded IMU propagation or aligned optical correction")
    q = payload["quaternion_world_from_body_xyzw"]
    variance = payload["attitude_variance_rad2"]
    if status == "LOST":
        if q is not None or variance is not None:
            raise ValueError("LOST fused attitude must not publish orientation or variance")
    else:
        if not isinstance(q, list) or len(q) != 4:
            raise ValueError("publishable fused attitude requires an XYZW quaternion")
        values = [_finite(x, "quaternion component") for x in q]
        norm = math.sqrt(sum(x*x for x in values))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError("fused attitude quaternion must be normalized")
        v = _finite(variance, "attitude_variance_rad2")
        if v < 0:
            raise ValueError("attitude variance must be nonnegative")
    optical_ts = payload["last_optical_timestamp_ns"]
    optical_frame = payload["last_optical_frame_id"]
    optical_age = payload["optical_age_ns"]
    if optical_ts is None:
        if status != "LOST":
            raise ValueError("publishable attitude requires an optical seed")
        if optical_frame is not None or optical_age is not None:
            raise ValueError("optical frame and age require an optical timestamp")
    else:
        optical_ts = _time(optical_ts, "last_optical_timestamp_ns")
        optical_frame = _time(optical_frame, "last_optical_frame_id")
        optical_age = _time(optical_age, "optical_age_ns")
        if optical_ts > timestamp or timestamp - optical_ts != optical_age:
            raise ValueError("fused optical age is inconsistent with state timestamp")
    if payload["global_position_m"] is not None or payload["global_velocity_m_s"] is not None:
        raise ValueError("attitude-only stream cannot publish global position or velocity")
    if payload["global_position_observable"] is not False:
        raise ValueError("attitude-only stream cannot claim global position observability")
    return dict(payload)


def fused_attitude_state_to_dict(state: object) -> dict[str, Any]:
    """Serialize an OpticalIMUFusionState without changing optical PoseStream v1."""
    from optical_imu_fusion import OpticalIMUFusionState

    if not isinstance(state, OpticalIMUFusionState):
        raise ValueError("state must be OpticalIMUFusionState")
    return {
        "protocol": "low_cost_mocap_pose",
        "schema_version": 1,
        "kind": "fused_attitude_state",
        "body_id": state.body_id,
        "imu_id": state.imu_id,
        "timestamp_ns": state.timestamp_ns,
        "clock_domain": state.clock_domain,
        "quaternion_world_from_body_xyzw": (
            None if state.quaternion_world_from_body_xyzw is None
            else list(state.quaternion_world_from_body_xyzw)
        ),
        "attitude_variance_rad2": state.attitude_variance_rad2,
        "status": state.status,
        "source": state.source,
        "last_optical_timestamp_ns": state.last_optical_timestamp_ns,
        "last_optical_frame_id": state.last_optical_frame_id,
        "optical_age_ns": state.optical_age_ns,
        "clock_mapping_sha256": state.clock_mapping_sha256,
        "imu_calibration_sha256": state.imu_calibration_sha256,
        "optical_pattern_sha256": state.optical_pattern_sha256,
        "optical_calibration_sha256": state.optical_calibration_sha256,
        "global_position_m": None,
        "global_velocity_m_s": None,
        "global_position_observable": False,
        "hardware_provenance_verified": False,
    }
