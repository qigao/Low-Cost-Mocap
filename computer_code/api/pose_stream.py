"""Versioned, explicit quality-gated software pose stream serialization.

A static asset description is separate from per-exposure pose frames, like
Motive/NatNet asset/frame separation. This is NOT a NatNet implementation.
Clock timestamps and camera metadata are declarations, not independent proof
of physical sensor exposure. This module never touches flight-control IO.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import LockedBoardFrame
from temporal_pattern import _nonnegative_int


def _valid_pose_rotation(raw: object, board_id: str) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{board_id} pose rotation must be finite 3x3")
    if (
        not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-7)
        or abs(float(np.linalg.det(matrix)) - 1.0) > 1e-7
    ):
        raise ValueError(f"{board_id} pose rotation must be a proper rotation")
    return matrix


def _pose_xyz(raw: object, board_id: str) -> list[float]:
    try:
        xyz = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{board_id} pose translation must be finite xyz") from exc
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"{board_id} pose translation must be finite xyz")
    return [float(v) for v in xyz]


def _quat_xyzw(rotation: np.ndarray) -> list[float]:
    values = Rotation.from_matrix(rotation).as_quat()
    if not np.isfinite(values).all():
        raise ValueError("rotation quaternion is nonfinite")
    if values[-1] < 0:
        values = -values
    return [float(x) for x in values]


def describe_board_assets(registry: BoardPatternRegistry) -> dict[str, Any]:
    """Return revision-keyed static board geometry and LED timecode metadata."""
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    boards = []
    for board_id, board in sorted(registry.boards.items()):
        boards.append({
            "board_id": board_id,
            "type": board.board_type,
            "min_visible_markers": board.min_visible_markers,
            "body_from_board": {
                "rotation": board.rotation_body_from_board.tolist(),
                "translation_m": board.translation_body_from_board_m.tolist(),
            },
            "markers": [
                {
                    "marker_id": marker,
                    "xyz_m": [float(v) for v in board.marker_xyz_m[marker]],
                    "code": "".join(str(bit) for bit in registry.codebook.patterns[marker]),
                }
                for marker in sorted(board.marker_xyz_m)
            ],
        })
    return {
        "protocol": "low_cost_mocap_pose",
        "schema_version": 1,
        "kind": "asset_description",
        "pattern_sha256": registry.config_sha256,
        "hardware_provenance_verified": False,
        "model_coordinates_are_declared_not_metrologically_verified": True,
        "period_triggers": registry.period_triggers,
        "max_bit_errors": registry.max_bit_errors,
        "pose_phase": registry.pose_phase,
        "boards": boards,
    }


def pose_frame_to_dict(
    frame: LockedBoardFrame,
    registry: BoardPatternRegistry,
) -> dict[str, Any]:
    """Serialize all current board outcomes without ever publishing stale pose.

    A result is MEASURED only when tracking state=TRACKING, pose is fresh at
    the same trigger and age is zero. INVALID records contain null position,
    quaternion and residuals; last-measured age can be reported as metadata
    only. No prediction or hardware-certification pathway is present.
    """
    if not isinstance(frame, LockedBoardFrame):
        raise ValueError("frame must be LockedBoardFrame")
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    if frame.pattern_sha256 != registry.config_sha256:
        raise ValueError("pattern revision mismatch in pose frame")
    if frame.hardware_provenance_verified:
        raise ValueError("software pose serialization cannot certify hardware provenance")
    trigger = _nonnegative_int(frame.trigger_id, "trigger_id")
    exposure_ns = _nonnegative_int(
        frame.latest_exposure_timestamp_ns, "latest_exposure_timestamp_ns"
    )
    skew_ns = _nonnegative_int(frame.exposure_skew_ns, "exposure_skew_ns")
    if (
        not isinstance(frame.clock_domain, str)
        or not frame.clock_domain.strip()
        or not isinstance(frame.calibration_sha256, str)
        or len(frame.calibration_sha256) != 64
        or any(c not in "0123456789abcdef" for c in frame.calibration_sha256)
    ):
        raise ValueError("pose frame has invalid clock domain or calibration revision")
    cameras = tuple(_nonnegative_int(x, "camera_id") for x in frame.camera_ids)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("pose frame camera IDs must be unique and nonempty")
    if set(frame.boards) != set(registry.boards):
        raise ValueError("pose frame must describe exactly the registered boards")
    quarantined = tuple(
        _nonnegative_int(camera, "quarantined camera_id")
        for camera in frame.quarantined_camera_ids
    )
    if (
        len(set(quarantined)) != len(quarantined)
        or not set(quarantined).issubset(set(cameras))
    ):
        raise ValueError("quarantined camera IDs must be unique and expected")

    bodies = []
    for board_id in sorted(registry.boards):
        record = frame.boards[board_id]
        result = record.outcome
        if record.board_id != board_id or result.board_id != board_id:
            raise ValueError("board status identity differs from registry")
        last_trigger = record.measurement_trigger_id
        age = record.measurement_age_triggers
        last_time = record.last_measurement_exposure_timestamp_ns
        age_ns = record.measurement_age_ns
        if last_trigger is not None:
            last_trigger = _nonnegative_int(last_trigger, "measurement_trigger_id")
            if last_trigger > trigger:
                raise ValueError("measurement cannot originate from future trigger")
        if age is not None:
            age = _nonnegative_int(age, "measurement_age_triggers")
        if last_time is not None:
            last_time = _nonnegative_int(
                last_time, "last_measurement_exposure_timestamp_ns"
            )
            if last_time > exposure_ns:
                raise ValueError("measurement cannot originate from future timestamp")
        if age_ns is not None:
            age_ns = _nonnegative_int(age_ns, "measurement_age_ns")
        if (last_trigger is None) != (age is None):
            raise ValueError("measurement age requires a reference trigger")
        if (last_time is None) != (age_ns is None):
            raise ValueError("measurement nanosecond age requires a reference time")
        if last_trigger is not None and age != trigger - last_trigger:
            raise ValueError("measurement trigger age is inconsistent")
        if last_time is not None and age_ns != exposure_ns - last_time:
            raise ValueError("measurement time age is inconsistent")

        seen = tuple(_nonnegative_int(x, "marker_id") for x in result.visible_marker_ids)
        excluded = tuple(
            _nonnegative_int(x, "excluded marker_id")
            for x in result.excluded_marker_ids
        )
        owned = set(registry.boards[board_id].marker_xyz_m)
        if (
            not set(seen).issubset(owned)
            or not set(excluded).issubset(owned)
            or set(seen) & set(excluded)
            or len(seen) != len(set(seen))
            or len(excluded) != len(set(excluded))
        ):
            raise ValueError("board marker IDs are missing or owned by another board")

        measured = record.evidence_kind == "MEASURED"
        if measured:
            pose = result.pose
            if (
                record.state != "TRACKING"
                or result.status != "recognized"
                or pose is None
                or pose.board_id != board_id
                or pose.trigger_id != trigger
                or last_trigger != trigger
                or last_time != exposure_ns
                or age != 0
                or age_ns != 0
            ):
                raise ValueError("MEASURED pose must be fresh at current exposure")
            if pose.metric_scale_validated:
                raise ValueError("software tracker has not validated physical metric scale")
            R_board = _valid_pose_rotation(
                pose.rotation_world_from_board, board_id
            )
            R_body = _valid_pose_rotation(
                pose.rotation_world_from_body, board_id
            )
            position = _pose_xyz(pose.translation_world_from_body, board_id)
            board_position = _pose_xyz(pose.translation_world_from_board, board_id)
            board_model = registry.boards[board_id]
            R_expected_body = R_board @ board_model.rotation_body_from_board.T
            t_expected_body = (
                np.asarray(board_position)
                - R_expected_body @ board_model.translation_body_from_board_m
            )
            if (
                not np.allclose(R_body, R_expected_body, atol=1e-7, rtol=0)
                or not np.allclose(position, t_expected_body, atol=1e-7, rtol=0)
            ):
                raise ValueError("body pose disagrees with registered board mounting transform")
            pose_marker_ids = tuple(
                _nonnegative_int(x, "pose marker_id") for x in pose.marker_ids
            )
            if (
                len(set(pose_marker_ids)) != len(pose_marker_ids)
                or not set(pose_marker_ids).issubset(set(seen))
            ):
                raise ValueError("pose includes duplicate or unobserved marker identity")
            quality = {}
            for label, number in (
                ("rigid_rms_world_units", pose.rigid_rms_error),
                ("reprojection_max_rms_px", pose.reprojection_max_rms_px),
            ):
                try:
                    numeric = float(number)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {label} metric") from exc
                if not np.isfinite(numeric) or numeric < 0:
                    raise ValueError(f"invalid {label} metric")
                quality[label] = numeric
            rotation = _quat_xyzw(R_body)
            board_rotation = _quat_xyzw(R_board)
        else:
            if record.evidence_kind != "INVALID" or result.pose is not None:
                raise ValueError("only MEASURED or INVALID output is supported")
            if record.state not in ("DISCOVER", "DEGRADED", "LOST"):
                raise ValueError("invalid board state")
            position = rotation = board_position = board_rotation = quality = None

        bodies.append({
            "board_id": board_id,
            "tracking_status": record.state,
            "evidence_kind": record.evidence_kind,
            "measurement_trigger_id": last_trigger,
            "measurement_age_triggers": age,
            "last_measurement_exposure_timestamp_ns": last_time,
            "measurement_age_ns": age_ns,
            "visible_marker_ids": sorted(seen),
            "excluded_marker_ids": sorted(excluded),
            "position_world_from_body": position,
            "quaternion_world_from_body_xyzw": rotation,
            "position_world_from_board": board_position,
            "quaternion_world_from_board_xyzw": board_rotation,
            "quality": quality,
            "metric_scale_validated": False,
        })
    return {
        "protocol": "low_cost_mocap_pose",
        "schema_version": 1,
        "kind": "pose_frame",
        "frame_id": trigger,
        "clock_domain": frame.clock_domain,
        "latest_exposure_timestamp_ns": exposure_ns,
        "exposure_skew_ns": skew_ns,
        "camera_ids": list(cameras),
        "quarantined_camera_ids": sorted(quarantined),
        "pattern_sha256": frame.pattern_sha256,
        "calibration_sha256": frame.calibration_sha256,
        "hardware_provenance_verified": False,
        "position_units": "uncalibrated_world",
        "rigid_bodies": bodies,
    }


def pose_frame_to_json(
    frame: LockedBoardFrame, registry: BoardPatternRegistry
) -> str:
    """Stable finite JSON suitable for independent software consumers."""
    return json.dumps(
        pose_frame_to_dict(frame, registry),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
