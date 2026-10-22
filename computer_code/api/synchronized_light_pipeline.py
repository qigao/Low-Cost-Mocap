"""Atomic multi-camera 2D-local-tracking -> timecode -> current-pose adapter.

The caller supplies externally verified FrameStamp batches, one raw grayscale
image and one independently detected bright-centroid list per physical camera.
This module neither generates trigger IDs nor connects a PS3Eye/serial output.
No raw frames are retained. Local-tracker state is staged across *all*
cameras; none is committed unless the 6DoF service accepts the same batch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from board_recognizer import LocalTrackCycle
from identity_locked_board_tracker import (
    IdentityLockedBoardTracker, LockedBoardFrame,
)
from light_intensity import (
    StampedTrackROI, sample_tracked_light_roi,
)
from light_tracking import BoundedLightTracker
from marker_admission import admit_track_marker_at_frame
from temporal_pattern import (
    FrameStamp, _nonnegative_int, validate_trigger_batch,
)


class SynchronizedLocalLightPipeline:
    """Bounded non-hardware adapter with explicit failure/rollback semantics.

    The local identity of an undecoded bright contour is not a board marker ID.
    At the end of each global code period a complete bounded local history
    can be decoded against the current immutable registry. Once confirmed,
    subsequent true ON observations may contribute to every-trigger rigid
    pose fitting until bounded identity expiry or code mismatch.
    """

    def __init__(
        self,
        pose_tracker: IdentityLockedBoardTracker,
        *,
        max_tracks_per_camera: int | None = None,
        max_motion_px_per_trigger: float = 20.0,
        max_missing_triggers: int = 3,
        minimum_detection_separation_px: float = 0.5,
        ambiguity_margin_fraction: float = 0.05,
        min_decode_observations: int = 8,
    ) -> None:
        if not isinstance(pose_tracker, IdentityLockedBoardTracker):
            raise ValueError("pose_tracker must be IdentityLockedBoardTracker")
        self.pose_tracker = pose_tracker
        registry = pose_tracker.registry
        if max_tracks_per_camera is None:
            max_tracks_per_camera = len(registry.marker_to_board)
        limit = _nonnegative_int(max_tracks_per_camera, "max_tracks_per_camera")
        if limit < 1 or limit > 128:
            raise ValueError("max_tracks_per_camera must be 1..128")
        min_obs = _nonnegative_int(min_decode_observations, "min_decode_observations")
        if min_obs < 4 or min_obs > registry.period_triggers:
            raise ValueError("min_decode_observations must be 4..code period")
        self._min_decode_observations = min_obs
        self._local_trackers = {
            camera: BoundedLightTracker(
                camera_id=camera,
                max_tracks=limit,
                max_motion_px_per_trigger=max_motion_px_per_trigger,
                max_missing_triggers=max_missing_triggers,
                ambiguity_margin_fraction=ambiguity_margin_fraction,
                minimum_detection_separation_px=minimum_detection_separation_px,
                allow_trigger_gaps=pose_tracker.allow_trigger_gaps,
            )
            for camera in pose_tracker.camera_ids
        }
        # Bounded per-track history of structured ROI evidence; never images.
        self._histories: dict[tuple[int, int], tuple[StampedTrackROI, ...]] = {}

    def active_local_track_ids(self, camera_id: int) -> tuple[int, ...]:
        camera = _nonnegative_int(camera_id, "camera_id")
        if camera not in self._local_trackers:
            raise ValueError("unknown camera_id")
        return self._local_trackers[camera].active_track_ids

    @property
    def retained_roi_sample_count(self) -> int:
        """Bounded by cameras * max_tracks * code_period."""
        return sum(len(trace) for trace in self._histories.values())

    def process(
        self,
        frames: Sequence[FrameStamp],
        grayscale_by_camera: Mapping[int, object],
        detections_by_camera: Mapping[int, Sequence[object]],
        *,
        independently_qualified_dark_track_ids: Mapping[
            int, Sequence[int]
        ] | None = None,
        controller_config_sha256: str | None = None,
    ) -> LockedBoardFrame:
        """Transactionally ingest one complete, declared trigger batch.

        The qualification map is optional: absent entries are *erasures*, not
        OFF measurements. An external motion-quality verifier, never the
        local position predictor alone, must authorize dark predicted ROIs.
        The controller SHA only checks software configuration equality.
        """
        tracker = self.pose_tracker
        camera_ids = tuple(tracker.camera_ids)
        if (
            not isinstance(grayscale_by_camera, Mapping)
            or set(grayscale_by_camera) != set(camera_ids)
            or not isinstance(detections_by_camera, Mapping)
            or set(detections_by_camera) != set(camera_ids)
        ):
            raise ValueError("grayscale and detection maps must cover exactly physical camera IDs")
        if independently_qualified_dark_track_ids is None:
            allowed_dark = {}
        else:
            if (
                not isinstance(independently_qualified_dark_track_ids, Mapping)
                or not set(independently_qualified_dark_track_ids).issubset(set(camera_ids))
            ):
                raise ValueError("dark ROI qualification has unknown camera IDs")
            allowed_dark = {
                camera: frozenset(
                    _nonnegative_int(x, "qualified dark local track ID")
                    for x in ids
                )
                for camera, ids in independently_qualified_dark_track_ids.items()
            }

        batch = validate_trigger_batch(
            frames, camera_ids,
            max_exposure_skew_ns=tracker.max_exposure_skew_ns,
            previous_trigger_id=tracker.last_trigger_id,
            allow_trigger_gaps=tracker.allow_trigger_gaps,
        )
        if batch.trigger_id < tracker.trigger_epoch:
            raise ValueError("batch trigger precedes active LED code epoch")
        stamps = {stamp.camera_id: stamp for stamp in frames}
        period = tracker.registry.period_triggers
        phase = (batch.trigger_id - tracker.trigger_epoch) % period
        cycle_index = (batch.trigger_id - tracker.trigger_epoch) // period

        # Make a shallow copy of immutable per-track evidence tuples and a
        # deep numeric-state clone of each bounded 2D track table.
        stage_local = {
            camera: state.staged_copy()
            for camera, state in self._local_trackers.items()
        }
        stage_histories = dict(self._histories)
        all_stamped = []
        candidates = []
        for camera in camera_ids:
            image = np.asarray(grayscale_by_camera[camera])
            if image.ndim != 2 or image.dtype.kind not in "iuf":
                raise ValueError("raw grayscale frame must be a numeric 2D array")
            if not np.isfinite(image).all():
                raise ValueError("raw grayscale frame contains nonfinite samples")

            # A locked LED whose expected phase is OFF must never take a
            # neighbor's actual bright detection in this 2D assignment.
            forbidden = tuple(
                lock.local_track_id
                for lock in tracker.active_identity_locks
                if lock.camera_id == camera
                and batch.trigger_id - lock.confirmed_trigger_id <= tracker.max_identity_age_triggers
                and tracker.registry.codebook.patterns[lock.marker_id][phase] == 0
            )
            evidence = stage_local[camera].update(
                batch.trigger_id,
                detections_by_camera[camera],
                forbidden_track_ids=forbidden,
            )
            active = set(stage_local[camera].active_track_ids)
            for key in tuple(stage_histories):
                if key[0] == camera and key[1] not in active:
                    del stage_histories[key]
            qualified = allowed_dark.get(camera, frozenset())
            if not qualified.issubset(active):
                raise ValueError("qualified dark local track is not active")
            for point in evidence:
                key = (camera, point.track_id)
                motion_qualified = point.track_id in qualified
                roi = sample_tracked_light_roi(
                    image, point,
                    predicted_roi_qualified=motion_qualified,
                )
                stamp = stamps[camera]
                item = StampedTrackROI(
                    frame_stamp=stamp,
                    track_evidence=point,
                    roi=roi,
                    predicted_roi_qualified=motion_qualified,
                )
                all_stamped.append(item)
                previous = stage_histories.get(key, ())
                if previous and (
                    (previous[-1].frame_stamp.trigger_id - tracker.trigger_epoch) // period
                    != cycle_index
                ):
                    previous = ()
                history = (*previous[-period + 1:], item)
                stage_histories[key] = history

                # Final phase provides an opportunity to decode the *entire*
                # current epoch cycle with erasures. Other phases never seed
                # IDs just from one bright frame or spatial proximity.
                if (
                    phase == period - 1
                    and roi is not None
                    and point.observed_xy is not None
                    and len(history) >= self._min_decode_observations
                    and tracker.brightness_calibration_by_camera_id[camera].classify(roi.contrast) == 1
                ):
                    try:
                        # Prequalify here to skip undecodable local histories
                        # without aborting healthy board IDs. The downstream
                        # tracker MUST independently re-decode this trace.
                        admit_track_marker_at_frame(
                            history, tracker.registry.codebook,
                            tracker.brightness_calibration_by_camera_id[camera],
                            camera_id=camera, local_track_id=point.track_id,
                            target_frame=stamp,
                            trigger_epoch=tracker.trigger_epoch,
                            min_valid_samples=self._min_decode_observations,
                        )
                    except ValueError:
                        continue
                    candidates.append(LocalTrackCycle(
                        camera_id=camera,
                        local_track_id=point.track_id,
                        target_frame=stamp,
                        trace=tuple(history),
                    ))

        # Actual BoundedLightTracker retirement (not a temporary OFF phase)
        # invalidates the old lock. A new local track can only inherit the
        # same global marker after its own full-cycle code admission.
        retired_local_tracks = tuple(
            (lock.camera_id, lock.local_track_id)
            for lock in tracker.active_identity_locks
            if lock.local_track_id
            not in stage_local[lock.camera_id].active_track_ids
        )
        result = tracker.advance(
            frames, all_stamped,
            revalidation_traces=candidates,
            retired_local_tracks=retired_local_tracks,
            controller_config_sha256=controller_config_sha256,
        )
        # Only publish local tracker and bounded history after the global
        # provenance, current-code and geometry path has accepted the batch.
        self._local_trackers = stage_local
        self._histories = stage_histories
        return result
