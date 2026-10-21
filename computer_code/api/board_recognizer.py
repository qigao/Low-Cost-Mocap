"""Multi-board identity and pose recognition from synchronized LED observations.

Invalid boards produce no pose and never reuse prior frames. Unknown global
marker IDs reject the whole scene. Hardware exposure provenance and real
flight-control integration remain separate acceptance gates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from board_patterns import BoardPatternRegistry
from marker_admission import AdmittedMarker
from spatiotemporal_reconstruction import (
    SynchronizedGeometry,
    reconstruct_synchronized_markers,
    fit_synchronized_rigid_pattern,
)
from temporal_pattern import (
    FrameStamp, _nonnegative_int, group_synchronized_detections,
)


@dataclass(frozen=True)
class BoardPose:
    board_id: str
    trigger_id: int
    marker_ids: tuple[int, ...]
    rotation_world_from_board: np.ndarray
    translation_world_from_board: np.ndarray
    rotation_world_from_body: np.ndarray
    translation_world_from_body: np.ndarray
    rigid_rms_error: float
    reprojection_max_rms_px: float
    metric_scale_validated: bool = False


@dataclass(frozen=True)
class BoardOutcome:
    board_id: str
    status: str
    visible_marker_ids: tuple[int, ...]
    excluded_marker_ids: tuple[int, ...]
    pose: BoardPose | None


@dataclass(frozen=True)
class BoardRecognitionResult:
    trigger_id: int
    camera_ids: tuple[int, ...]
    boards: Mapping[str, BoardOutcome]


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ValueError("board pose must contain finite coordinates")
    out.setflags(write=False)
    return out


def recognize_boards(
    reconstruction: SynchronizedGeometry,
    registry: BoardPatternRegistry,
    *,
    max_rigid_rms_world_units: float = 0.02,
) -> BoardRecognitionResult:
    """Report every board as recognized, not_visible, insufficient or rejected.

    Each board is fitted against *only its own* global marker IDs. A rejected
    board cannot leak a partial/stale pose. Another healthy board may still
    be recognized, but an unknown marker identity rejects the whole scene.
    """
    if not isinstance(reconstruction, SynchronizedGeometry):
        raise ValueError("reconstruction must be SynchronizedGeometry")
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    if (
        isinstance(max_rigid_rms_world_units, (bool, np.bool_))
        or not isinstance(
            max_rigid_rms_world_units, (int, float, np.integer, np.floating)
        )
    ):
        raise ValueError("max_rigid_rms_world_units must be a finite nonnegative scalar")
    try:
        limit = float(max_rigid_rms_world_units)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("max_rigid_rms_world_units must be a finite nonnegative scalar") from exc
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_rigid_rms_world_units must be a finite nonnegative scalar")

    trigger = _nonnegative_int(reconstruction.trigger_id, "trigger_id")
    if not reconstruction.camera_ids:
        raise ValueError("reconstruction has no synchronized camera IDs")
    excluded_ids = tuple(reconstruction.excluded_marker_ids)
    for raw_marker in (*reconstruction.markers, *excluded_ids):
        marker = _nonnegative_int(raw_marker, "marker_id")
        if marker not in registry.marker_to_board:
            raise ValueError(f"unregistered marker_id {marker} in synchronized scene")
    if set(reconstruction.markers) & set(excluded_ids):
        raise ValueError("excluded marker ID also appears in admitted geometry")

    outcomes: dict[str, BoardOutcome] = {}
    for board_id, board in registry.boards.items():
        ids = tuple(sorted(
            marker for marker in reconstruction.markers
            if registry.marker_to_board[marker] == board_id
        ))
        excluded = tuple(sorted(
            marker for marker in excluded_ids
            if registry.marker_to_board[marker] == board_id
        ))
        if len(ids) < board.min_visible_markers:
            status = "not_visible" if not ids and not excluded else "insufficient_markers"
            outcomes[board_id] = BoardOutcome(board_id, status, ids, excluded, None)
            continue
        board_scene = SynchronizedGeometry(
            trigger_id=trigger,
            camera_ids=reconstruction.camera_ids,
            markers=MappingProxyType({
                marker: reconstruction.markers[marker] for marker in ids
            }),
            excluded_marker_ids=(),
        )
        try:
            fit = fit_synchronized_rigid_pattern(
                board_scene, board.marker_xyz_m,
                min_markers=board.min_visible_markers, max_rms_error=limit,
            )
        except ValueError:
            outcomes[board_id] = BoardOutcome(
                board_id, "geometry_rejected", ids, excluded, None
            )
            continue

        R_wb, t_wb = fit.rotation, fit.translation
        # p_body = R_body_from_board @ p_board + t_body_from_board.
        # world_from_body = world_from_board @ inverse(body_from_board).
        R_wbody = R_wb @ board.rotation_body_from_board.T
        t_wbody = t_wb - R_wbody @ board.translation_body_from_board_m
        pose = BoardPose(
            board_id, trigger, fit.marker_ids,
            _readonly(R_wb), _readonly(t_wb),
            _readonly(R_wbody), _readonly(t_wbody),
            fit.rigid_rms_error, fit.reprojection_max_rms_px, False,
        )
        outcomes[board_id] = BoardOutcome(
            board_id, "recognized", ids, excluded, pose
        )
    return BoardRecognitionResult(
        trigger, tuple(reconstruction.camera_ids), MappingProxyType(outcomes)
    )


def recognize_admitted_boards(
    admissions: Sequence[AdmittedMarker],
    frames: Sequence[FrameStamp],
    expected_camera_ids: Sequence[int],
    camera_poses_by_id: Mapping,
    intrinsics_by_id: Mapping,
    registry: BoardPatternRegistry,
    *,
    trigger_epoch: int,
    max_exposure_skew_ns: int,
    previous_trigger_id: int | None = None,
    allow_trigger_gaps: bool = False,
    max_reprojection_rms_px: float = 1.0,
    max_rigid_rms_world_units: float = 0.02,
) -> BoardRecognitionResult:
    """Service entry: local-trace admitted markers -> current multi-board poses.

    Input stamps still require *external hardware* exposure/clock provenance.
    This function does not infer synchronization from host timestamps.
    If called only at a configured pose_phase, allow_trigger_gaps=True must
    be explicitly requested, and the caller must independently account for
    all intervening hardware triggers. It is not an inferred sync fallback.
    """
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    epoch = _nonnegative_int(trigger_epoch, "trigger_epoch")
    if not frames:
        raise ValueError("frame stamps are required")
    seen_local: set[tuple[int, int]] = set()
    detections = []
    for admission in admissions:
        if not isinstance(admission, AdmittedMarker):
            raise ValueError("board recognition requires AdmittedMarker records")
        stamp = admission.detection.frame_stamp
        if not isinstance(stamp, FrameStamp):
            raise ValueError("admitted marker has no originating FrameStamp")
        trigger = _nonnegative_int(stamp.trigger_id, "trigger_id")
        if trigger < epoch or (
            admission.decoded_cycle_index
            != (trigger - epoch) // registry.period_triggers
        ):
            raise ValueError("admitted marker is from another temporal code cycle")
        if admission.camera_id != stamp.camera_id:
            raise ValueError("admitted marker camera does not match frame")
        local = _nonnegative_int(admission.local_track_id, "local_track_id")
        local_key = (admission.camera_id, local)
        if local_key in seen_local:
            raise ValueError("local light track reused for multiple marker IDs")
        seen_local.add(local_key)
        if admission.detection.marker_id not in registry.marker_to_board:
            raise ValueError("admitted marker_id is not registered to a board")
        detections.append(admission.detection)

    batch, grouped = group_synchronized_detections(
        detections, frames, expected_camera_ids,
        max_exposure_skew_ns=max_exposure_skew_ns,
        previous_trigger_id=previous_trigger_id,
        allow_trigger_gaps=allow_trigger_gaps,
        allowed_marker_ids=tuple(registry.marker_to_board),
    )
    if batch.trigger_id < epoch:
        raise ValueError("batch trigger precedes temporal code epoch")
    if registry.pose_phase is not None and (
        (batch.trigger_id - epoch) % registry.period_triggers
        != registry.pose_phase
    ):
        raise ValueError("current frame is outside configured shared pose_phase")

    if grouped.observations_by_marker_id:
        reconstruction = reconstruct_synchronized_markers(
            batch, grouped, camera_poses_by_id, intrinsics_by_id,
            max_reprojection_rms_px=max_reprojection_rms_px,
        )
    else:
        reconstruction = SynchronizedGeometry(
            trigger_id=batch.trigger_id,
            camera_ids=batch.camera_ids,
            markers=MappingProxyType({}),
            excluded_marker_ids=grouped.excluded_marker_ids,
        )
    return recognize_boards(
        reconstruction, registry,
        max_rigid_rms_world_units=max_rigid_rms_world_units,
    )
