"""Bounded temporal-ID locking and current-frame multi-board pose measurements.

Software-only: this does NOT create hardware exposure triggers, authenticate a
controller, or switch the live drone serial path. A lock comes from a
a locally re-decoded code-history trace using the active registry, not a guessed marker ID.
Later frames use the active codebook phase and *actually observed* ON centroids.
No predicted/dark pixel is ever labeled as a measured 2D or 6DoF observation.

State is bounded to marker IDs, per-local-track confirmation metadata,
monotonic frame stamps and each board's last-measurement *time only*.
Raw images, 3D runtime states and old poses are not persisted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import (
    BoardOutcome, BoardRecognitionResult, LocalTrackCycle, recognize_boards,
)
from geometry import _camera_arrays, _observation_array
from light_intensity import BinaryIntensityCalibration, ROIContrast, StampedTrackROI
from light_tracking import TrackEvidence
from marker_admission import admit_track_marker_at_frame
from spatiotemporal_reconstruction import (
    SynchronizedGeometry, reconstruct_synchronized_markers,
)
from temporal_pattern import (
    FrameStamp, GroupedMarkers, StampedIdentifiedDetection,
    _nonnegative_int, group_synchronized_detections,
    validate_trigger_batch,
)


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise ValueError(f"{name} must be finite nonnegative real")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite nonnegative real") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite nonnegative real")
    return result


@dataclass(frozen=True)
class MarkerIdentityLock:
    camera_id: int
    local_track_id: int
    marker_id: int
    confirmed_trigger_id: int
    config_sha256: str


@dataclass(frozen=True)
class BoardFrameStatus:
    board_id: str
    state: str  # DISCOVER, TRACKING, DEGRADED or LOST
    evidence_kind: str  # MEASURED or INVALID (prediction is not implemented)
    measurement_trigger_id: int | None
    measurement_age_triggers: int | None
    last_measurement_exposure_timestamp_ns: int | None
    measurement_age_ns: int | None
    outcome: BoardOutcome


@dataclass(frozen=True)
class LockedBoardFrame:
    trigger_id: int
    camera_ids: tuple[int, ...]
    latest_exposure_timestamp_ns: int
    pattern_sha256: str
    calibration_sha256: str
    boards: Mapping[str, BoardFrameStatus]
    active_locks: tuple[MarkerIdentityLock, ...]
    newly_confirmed_marker_ids: tuple[int, ...]
    invalidated_marker_ids: tuple[int, ...]
    measured_marker_ids: tuple[int, ...]
    hardware_provenance_verified: bool = False


class IdentityLockedBoardTracker:
    """Streaming software state machine for known hardware camera exposures.

    Caller must supply physically verified current-trigger frame stamps and
    camera-local track evidence. This service still validates the declared
    stamps, temporal code and per-camera calibration contracts. All global
    failures leave internal state unchanged. Unknown/outdated identities
    fail closed. Board-local geometry errors are isolated by board ID.
    """

    def __init__(
        self,
        registry: BoardPatternRegistry,
        camera_ids: Sequence[int],
        camera_poses_by_id: Mapping,
        intrinsics_by_id: Mapping,
        brightness_calibration_by_camera_id: Mapping[int, BinaryIntensityCalibration],
        *,
        trigger_epoch: int,
        max_exposure_skew_ns: int,
        max_identity_age_triggers: int | None = None,
        max_degraded_triggers: int = 2,
        max_reprojection_rms_px: float = 1.0,
        max_rigid_rms_world_units: float = 0.02,
        allow_trigger_gaps: bool = False,
    ) -> None:
        if not isinstance(registry, BoardPatternRegistry):
            raise ValueError("registry must be BoardPatternRegistry")
        self.registry = registry
        self.trigger_epoch = _nonnegative_int(trigger_epoch, "trigger_epoch")
        self.camera_ids = tuple(
            _nonnegative_int(i, "camera_id") for i in camera_ids
        )
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique and nonempty")
        if (
            not isinstance(camera_poses_by_id, Mapping)
            or set(camera_poses_by_id) != set(self.camera_ids)
            or not isinstance(intrinsics_by_id, Mapping)
            or set(intrinsics_by_id) != set(self.camera_ids)
        ):
            raise ValueError("camera poses and intrinsics must cover exactly physical camera IDs")
        if (
            not isinstance(brightness_calibration_by_camera_id, Mapping)
            or set(brightness_calibration_by_camera_id) != set(self.camera_ids)
            or not all(
                isinstance(brightness_calibration_by_camera_id[c], BinaryIntensityCalibration)
                for c in self.camera_ids
            )
        ):
            raise ValueError("brightness calibration must cover exactly physical camera IDs")
        # Freeze valid calibration with the tracker. Mutating a caller's
        # NumPy array after initialization must not silently change an
        # accepted 2D->3D model mid-session.
        frozen_poses = {}
        frozen_intrinsics = {}
        calibration_manifest = []
        for camera in self.camera_ids:
            K, R, t = _camera_arrays(
                camera_poses_by_id[camera], intrinsics_by_id[camera]
            )
            K, R, t = K.copy(), R.copy(), t.copy()
            for item in (K, R, t):
                item.setflags(write=False)
            frozen_poses[camera] = MappingProxyType({"R": R, "t": t})
            frozen_intrinsics[camera] = K
            calibration_manifest.append({
                "camera_id": camera, "K": K.tolist(),
                "R": R.tolist(), "t": t.tolist(),
            })
        self.camera_poses_by_id = MappingProxyType(frozen_poses)
        self.intrinsics_by_id = MappingProxyType(frozen_intrinsics)
        self.calibration_sha256 = hashlib.sha256(
            json.dumps(
                calibration_manifest, sort_keys=True,
                separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        self.brightness_calibration_by_camera_id = MappingProxyType(
            dict(brightness_calibration_by_camera_id)
        )
        self.max_exposure_skew_ns = _nonnegative_int(
            max_exposure_skew_ns, "max_exposure_skew_ns"
        )
        if max_identity_age_triggers is None:
            max_identity_age_triggers = 2 * registry.period_triggers
        self.max_identity_age_triggers = _nonnegative_int(
            max_identity_age_triggers, "max_identity_age_triggers"
        )
        if self.max_identity_age_triggers < 1:
            raise ValueError("max_identity_age_triggers must be positive")
        self.max_degraded_triggers = _nonnegative_int(
            max_degraded_triggers, "max_degraded_triggers"
        )
        if not isinstance(allow_trigger_gaps, (bool, np.bool_)):
            raise ValueError("allow_trigger_gaps must be boolean")
        self.allow_trigger_gaps = bool(allow_trigger_gaps)
        # Numeric policy validation must reject unrepresentable Python ints
        # and NaN before they enter expensive reconstruction.
        self.max_reprojection_rms_px = _finite_nonnegative(
            max_reprojection_rms_px, "max_reprojection_rms_px"
        )
        self.max_rigid_rms_world_units = _finite_nonnegative(
            max_rigid_rms_world_units, "max_rigid_rms_world_units"
        )

        self._locks: dict[tuple[int, int], MarkerIdentityLock] = {}
        self._last_trigger_id: int | None = None
        self._clock_domain: str | None = None
        self._last_exposure_by_camera_id: dict[int, int] = {}
        self._last_measured_trigger: dict[str, int] = {}
        self._last_measured_exposure_ns: dict[str, int] = {}

    @property
    def last_trigger_id(self) -> int | None:
        return self._last_trigger_id

    @property
    def active_identity_locks(self) -> tuple[MarkerIdentityLock, ...]:
        return tuple(self._locks[key] for key in sorted(self._locks))

    def _isolated_geometry(
        self, batch, grouped: GroupedMarkers
    ) -> tuple[BoardRecognitionResult, set[str]]:
        """Triangulation failure in board B must not invalidate board A."""
        accepted = {}
        failed: set[str] = set()
        for board_id in self.registry.boards:
            members = {
                marker: observations
                for marker, observations in grouped.observations_by_marker_id.items()
                if self.registry.marker_to_board[marker] == board_id
            }
            if not members:
                continue
            board_group = GroupedMarkers(
                MappingProxyType(members), ()
            )
            try:
                local = reconstruct_synchronized_markers(
                    batch, board_group,
                    self.camera_poses_by_id,
                    self.intrinsics_by_id,
                    max_reprojection_rms_px=self.max_reprojection_rms_px,
                )
            except (ValueError, np.linalg.LinAlgError):
                failed.add(board_id)
                continue
            accepted.update(local.markers)
        scene = SynchronizedGeometry(
            trigger_id=batch.trigger_id,
            camera_ids=batch.camera_ids,
            markers=MappingProxyType(accepted),
            excluded_marker_ids=grouped.excluded_marker_ids,
        )
        result = recognize_boards(
            scene, self.registry,
            max_rigid_rms_world_units=self.max_rigid_rms_world_units,
        )
        if failed:
            outcomes = dict(result.boards)
            for board_id in failed:
                member_ids = tuple(sorted(
                    marker for marker in grouped.observations_by_marker_id
                    if self.registry.marker_to_board[marker] == board_id
                ))
                old = outcomes[board_id]
                outcomes[board_id] = BoardOutcome(
                    board_id, "geometry_rejected", member_ids,
                    old.excluded_marker_ids, None,
                )
            result = replace(result, boards=MappingProxyType(outcomes))
        return result, failed

    def advance(
        self,
        frames: Sequence[FrameStamp],
        tracked_roi: Sequence[StampedTrackROI],
        *,
        revalidation_traces: Sequence[LocalTrackCycle] = (),
        controller_config_sha256: str | None = None,
    ) -> LockedBoardFrame:
        """Process one real exposure batch, atomically, with no pose caching.

        After the first code-cycle admission, subsequent frames use locked
        IDs with *actual ON detections*, not predicted or OFF positions.
        Revalidation decodes the *raw track cycle* under the active registry,
        and must refer to the same current frame and local track.
        The current code phase and calibrated observed light bit must agree.
        Locks expire after max_identity_age_triggers without new code evidence.
        """
        if (
            controller_config_sha256 is not None
            and (
                not isinstance(controller_config_sha256, str)
                or controller_config_sha256 != self.registry.config_sha256
            )
        ):
            raise ValueError("controller pattern SHA-256 differs from active registry")
        batch = validate_trigger_batch(
            frames, self.camera_ids,
            max_exposure_skew_ns=self.max_exposure_skew_ns,
            previous_trigger_id=self._last_trigger_id,
            allow_trigger_gaps=self.allow_trigger_gaps,
        )
        if batch.trigger_id < self.trigger_epoch:
            raise ValueError("batch trigger precedes LED code epoch")
        if self._clock_domain is not None and batch.clock_domain != self._clock_domain:
            raise ValueError("camera clock domain changed across frame batches")
        stamps = {f.camera_id: f for f in frames}
        for stamp in frames:
            previous = self._last_exposure_by_camera_id.get(stamp.camera_id)
            if previous is not None and stamp.exposure_timestamp_ns <= previous:
                raise ValueError("camera exposure timestamp did not advance")

        phase = (batch.trigger_id - self.trigger_epoch) % self.registry.period_triggers
        # Only declared pose phases may publish measured 3D positions when
        # the registry constrains frame-level pose measurement to one phase.
        pose_phase_allowed = (
            self.registry.pose_phase is None
            or phase == self.registry.pose_phase
        )

        seen_roi: dict[tuple[int, int], StampedTrackROI] = {}
        if len(tracked_roi) > len(self.registry.marker_to_board) * len(self.camera_ids):
            raise ValueError("tracked ROI sample capacity exceeded")
        for entry in tracked_roi:
            if not isinstance(entry, StampedTrackROI):
                raise ValueError("tracked_roi must contain StampedTrackROI")
            stamp = entry.frame_stamp
            evidence = entry.track_evidence
            if not isinstance(evidence, TrackEvidence):
                raise ValueError("ROI evidence must contain TrackEvidence")
            if entry.roi is not None and not isinstance(entry.roi, ROIContrast):
                raise ValueError("ROI evidence must be ROIContrast or None")
            if not isinstance(stamp, FrameStamp) or stamp.camera_id not in stamps:
                raise ValueError("ROI stamp has unexpected camera")
            if stamp != stamps[stamp.camera_id]:
                raise ValueError("ROI stamp does not belong to current trigger batch")
            camera = _nonnegative_int(evidence.camera_id, "track camera_id")
            local_id = _nonnegative_int(evidence.track_id, "local_track_id")
            track_trigger = _nonnegative_int(evidence.trigger_id, "track trigger_id")
            if camera != stamp.camera_id or track_trigger != batch.trigger_id:
                raise ValueError("track evidence camera/trigger does not match frame")
            if not isinstance(entry.predicted_roi_qualified, (bool, np.bool_)):
                raise ValueError("predicted_roi_qualified must be boolean")
            if (
                evidence.observed_xy is None
                and entry.roi is not None
                and not entry.predicted_roi_qualified
            ):
                raise ValueError("predicted dark ROI requires independent qualification")
            key = (camera, local_id)
            if key in seen_roi:
                raise ValueError("duplicate camera-local track in current frame")
            seen_roi[key] = entry

        stage = {
            key: lock for key, lock in self._locks.items()
            if batch.trigger_id - lock.confirmed_trigger_id
            <= self.max_identity_age_triggers
        }
        invalidated = [
            lock.marker_id for key, lock in self._locks.items()
            if key not in stage
        ]
        newly_confirmed = []
        seen_revalidations = set()
        for candidate in revalidation_traces:
            if not isinstance(candidate, LocalTrackCycle):
                raise ValueError("revalidation_traces must contain LocalTrackCycle")
            camera = _nonnegative_int(candidate.camera_id, "revalidation camera_id")
            local_id = _nonnegative_int(
                candidate.local_track_id, "revalidation local_track_id"
            )
            key = (camera, local_id)
            if key in seen_revalidations:
                raise ValueError("duplicate code revalidation for a local track")
            seen_revalidations.add(key)
            roi = seen_roi.get(key)
            if roi is None or candidate.target_frame != roi.frame_stamp:
                raise ValueError("revalidated identity lacks current tracked frame evidence")
            # Critical trust boundary: there is no caller-supplied marker ID.
            # Decode from the full stamped local trace using *this* active
            # registry's codebook and the registered camera calibration.
            admission = admit_track_marker_at_frame(
                candidate.trace,
                self.registry.codebook,
                self.brightness_calibration_by_camera_id[camera],
                camera_id=camera, local_track_id=local_id,
                target_frame=roi.frame_stamp,
                trigger_epoch=self.trigger_epoch,
            )
            marker = _nonnegative_int(
                admission.detection.marker_id, "admitted marker_id"
            )
            if marker not in self.registry.marker_to_board:
                raise ValueError("revalidated marker_id absent from active codebook")
            if (
                admission.decoded_cycle_index
                != (batch.trigger_id - self.trigger_epoch)
                // self.registry.period_triggers
            ):
                raise ValueError("revalidated marker belongs to another code cycle")
            if roi.track_evidence.observed_xy is None:
                raise ValueError("revalidated identity lacks current observed centroid")
            actual_xy = _observation_array(
                roi.track_evidence.observed_xy, camera
            )
            admission_xy = _observation_array(admission.detection.xy, camera)
            if not np.array_equal(actual_xy, admission_xy):
                raise ValueError("revalidated marker centroid differs from current track")
            bit = None if roi.roi is None else self.brightness_calibration_by_camera_id[
                camera
            ].classify(roi.roi.contrast)
            if bit != 1 or self.registry.codebook.patterns[marker][phase] != 1:
                raise ValueError("revalidation requires calibrated ON code phase")
            old = stage.get(key)
            if old is not None and old.marker_id != marker:
                raise ValueError("local track has conflicting revalidated marker identity")
            for other_key, existing in stage.items():
                if other_key != key and other_key[0] == camera and existing.marker_id == marker:
                    raise ValueError("two local tracks claim the same global marker")
            stage[key] = MarkerIdentityLock(
                camera, local_id, marker, batch.trigger_id, self.registry.config_sha256
            )
            newly_confirmed.append(marker)

        detections = []
        for key, entry in seen_roi.items():
            lock = stage.get(key)
            if lock is None:
                continue  # Undecoded/unlocked lights must not become 3D markers.
            if lock.config_sha256 != self.registry.config_sha256:
                raise ValueError("local marker lock has stale configuration revision")
            expected = self.registry.codebook.patterns[lock.marker_id][phase]
            measured_bit = (
                None if entry.roi is None else
                self.brightness_calibration_by_camera_id[key[0]].classify(
                    entry.roi.contrast
                )
            )
            if measured_bit is not None and measured_bit != expected:
                invalidated.append(lock.marker_id)
                del stage[key]
                continue
            # Even if a predicted dark ROI is externally qualified, it never
            # supplies actual ON evidence or a triangulable 2D observation.
            observed_xy = entry.track_evidence.observed_xy
            if (
                pose_phase_allowed
                and expected == 1
                and measured_bit == 1
                and observed_xy is not None
            ):
                xy = _observation_array(observed_xy, key[0])
                detections.append(StampedIdentifiedDetection(
                    entry.frame_stamp, lock.marker_id, tuple(xy.tolist())
                ))

        # One locked global identity may only be observed once in each
        # physical camera per exposure. group_synchronized_detections
        # applies this check, plus the current batch's provenance.
        validated_batch, grouped = group_synchronized_detections(
            detections, frames, self.camera_ids,
            max_exposure_skew_ns=self.max_exposure_skew_ns,
            allowed_marker_ids=tuple(self.registry.marker_to_board),
        )
        board_result, geometry_failures = self._isolated_geometry(
            validated_batch, grouped
        )

        states = {}
        latest_exposure_ns = max(f.exposure_timestamp_ns for f in frames)
        staged_last = dict(self._last_measured_trigger)
        staged_times = dict(self._last_measured_exposure_ns)
        for board_id, outcome in board_result.boards.items():
            if outcome.pose is not None:
                staged_last[board_id] = batch.trigger_id
                staged_times[board_id] = latest_exposure_ns
                state = "TRACKING"
                kind = "MEASURED"
                age = 0
                age_ns = 0
                measurement_trigger = batch.trigger_id
                measurement_time = latest_exposure_ns
            else:
                last_measure = staged_last.get(board_id)
                age = (
                    batch.trigger_id - last_measure
                    if last_measure is not None else None
                )
                measurement_time = staged_times.get(board_id)
                age_ns = (
                    latest_exposure_ns - measurement_time
                    if measurement_time is not None else None
                )
                if age is None:
                    state = "DISCOVER"
                elif age <= self.max_degraded_triggers:
                    state = "DEGRADED"
                else:
                    state = "LOST"
                kind = "INVALID"  # never relabel old pose as a new one
                measurement_trigger = last_measure
            states[board_id] = BoardFrameStatus(
                board_id, state, kind, measurement_trigger, age,
                measurement_time, age_ns, outcome,
            )
        frame = LockedBoardFrame(
            trigger_id=batch.trigger_id,
            camera_ids=batch.camera_ids,
            latest_exposure_timestamp_ns=latest_exposure_ns,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256=self.calibration_sha256,
            boards=MappingProxyType(states),
            active_locks=tuple(stage[key] for key in sorted(stage)),
            newly_confirmed_marker_ids=tuple(sorted(set(newly_confirmed))),
            invalidated_marker_ids=tuple(sorted(set(invalidated))),
            measured_marker_ids=tuple(sorted(
                marker for marker in grouped.observations_by_marker_id
                if self.registry.marker_to_board[marker] not in geometry_failures
            )),
            hardware_provenance_verified=False,
        )

        # Publish state only after all global validations and pose fitting.
        self._locks = stage
        self._last_trigger_id = batch.trigger_id
        self._clock_domain = batch.clock_domain
        self._last_exposure_by_camera_id = {
            stamp.camera_id: stamp.exposure_timestamp_ns for stamp in frames
        }
        self._last_measured_trigger = staged_last
        self._last_measured_exposure_ns = staged_times
        return frame
