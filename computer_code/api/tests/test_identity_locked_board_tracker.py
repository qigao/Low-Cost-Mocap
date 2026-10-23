"""Synthetic continuous marker-ID lock and current-exposure rigid-board tests.

The camera and controller timestamps in this suite are *fabricated test
evidence*. They are not physical synchronization acceptance.
"""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import IdentityLockedBoardTracker
from light_intensity import BinaryIntensityCalibration, ROIContrast, StampedTrackROI
from light_tracking import TrackEvidence
from board_recognizer import LocalTrackCycle
from temporal_pattern import FrameStamp


API = Path(__file__).resolve().parents[1]
CONTINUOUS = API / "board-patterns.continuous.example.json"


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ContinuousBoardTrackerTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONTINUOUS)
        self.camera_ids = (2, 5, 9)
        self.K = {
            2: np.array([[800., 0., 320.], [0., 790., 240.], [0., 0., 1.]]),
            5: np.array([[1100., 0., 300.], [0., 1050., 230.], [0., 0., 1.]]),
            9: np.array([[1300., 0., 310.], [0., 1200., 250.], [0., 0., 1.]]),
        }
        self.cameras = {
            2: {"R": np.eye(3), "t": np.zeros(3)},
            5: {"R": np.eye(3), "t": np.array([-0.8, 0., 0.])},
            9: {"R": np.eye(3), "t": np.array([0., -0.6, 0.])},
        }
        self.brightness = {
            camera: BinaryIntensityCalibration(0.0, 120.0)
            for camera in self.camera_ids
        }
        self.R = {"board_A": yaw(0.23), "board_B": yaw(-0.31)}
        self.origin = {
            "board_A": np.array([-0.2, 0.05, 3.4]),
            "board_B": np.array([0.45, 0.27, 3.8]),
        }
        self.epoch = 1000
        self.first = 1015
        self.tracker = self.create_tracker()

    def create_tracker(self, **kwargs):
        return IdentityLockedBoardTracker(
            self.registry, self.camera_ids, self.cameras, self.K,
            self.brightness,
            trigger_epoch=self.epoch, max_exposure_skew_ns=60,
            max_identity_age_triggers=kwargs.get("max_identity_age_triggers", 32),
            max_degraded_triggers=2,
            max_reprojection_rms_px=1e-5,
            max_rigid_rms_world_units=1e-5,
        )

    def stamp(self, camera, trigger):
        return FrameStamp(
            camera, trigger,
            50_000_000 + (trigger - self.epoch) * 100_000
            + self.camera_ids.index(camera) * 20,
            "synthetic-hardware-reference",
        )

    def model_world(self, marker, trigger):
        board_id = self.registry.marker_to_board[marker]
        xyz = np.asarray(self.registry.boards[board_id].marker_xyz_m[marker])
        # Same physical board moves 1 millimeter per trigger after the
        # identity-lock bootstrap, with no unreported current observations.
        displacement = np.array([
            0.001 * (trigger - self.first),
            0.0002 * (trigger - self.first),
            0.0,
        ])
        return self.R[board_id] @ xyz + self.origin[board_id] + displacement

    def xy(self, camera, marker, trigger):
        pos = self.cameras[camera]
        xyz = pos["R"] @ self.model_world(marker, trigger) + pos["t"]
        projected = self.K[camera] @ xyz
        return tuple((projected[:2] / projected[2]).tolist())

    def samples(self, trigger, *, missing=frozenset(), corrupt=None):
        frames = [self.stamp(camera, trigger) for camera in self.camera_ids]
        phase = (trigger - self.epoch) % self.registry.period_triggers
        observed = []
        for frame in frames:
            camera = frame.camera_id
            for marker in self.registry.marker_to_board:
                lit = bool(self.registry.codebook.patterns[marker][phase])
                suppressed = (camera, marker) in missing
                actual_xy = self.xy(camera, marker, trigger) if lit and not suppressed else None
                local_id = marker + 1000  # local track identity != global LED identity
                exposure = TrackEvidence(
                    local_id, camera, trigger,
                    self.xy(camera, marker, trigger),
                    actual_xy, 0 if actual_xy is not None else 1,
                )
                bit = int(lit) if not suppressed else None
                if corrupt is not None and (camera, marker) == corrupt:
                    bit = 1 - bit if bit is not None else 1
                contrast = 120.0 if bit == 1 else 0.0
                roi = (
                    ROIContrast(contrast, 0.0, contrast, 10, 30)
                    if bit is not None else None
                )
                observed.append(StampedTrackROI(
                    frame, exposure, roi, predicted_roi_qualified=True
                ))
        return frames, observed

    def validations(self, trigger, *, missing=frozenset()):
        phase = (trigger - self.epoch) % self.registry.period_triggers
        start = trigger - phase
        out = []
        for camera in self.camera_ids:
            for marker in self.registry.marker_to_board:
                if (camera, marker) in missing:
                    continue
                local_id = marker + 1000
                trace = []
                for tick in range(start, trigger + 1):
                    code_phase = (tick - self.epoch) % self.registry.period_triggers
                    lit = self.registry.codebook.patterns[marker][code_phase]
                    xy = self.xy(camera, marker, tick)
                    evidence = TrackEvidence(
                        local_id, camera, tick, xy,
                        xy if lit else None, 0 if lit else 1,
                    )
                    contrast = 120.0 if lit else 0.0
                    trace.append(StampedTrackROI(
                        self.stamp(camera, tick), evidence,
                        ROIContrast(contrast, 0.0, contrast, 10, 30),
                        predicted_roi_qualified=not bool(lit),
                    ))
                # A known ON sample is required for revalidation and
                # initial lock at the current camera exposure.
                if not self.registry.codebook.patterns[marker][phase]:
                    continue
                out.append(LocalTrackCycle(
                    camera, local_id,
                    self.stamp(camera, trigger), tuple(trace),
                ))
        return out

    def bootstrap(self):
        frames, rois = self.samples(self.first)
        admitted = self.validations(self.first)
        result = self.tracker.advance(
            frames, rois, revalidation_traces=admitted,
            controller_config_sha256=self.registry.config_sha256,
        )
        self.assertEqual(len(result.active_locks), 24)
        self.assertEqual(
            tuple((b.state, b.evidence_kind) for b in result.boards.values()),
            (("TRACKING", "MEASURED"), ("TRACKING", "MEASURED")),
        )
        self.assertFalse(result.hardware_provenance_verified)
        return result

    def test_identity_lock_allows_true_measured_pose_at_every_next_trigger(self):
        self.bootstrap()
        for trigger in range(self.first + 1, self.first + 16):
            frames, rois = self.samples(trigger)
            result = self.tracker.advance(frames, rois)
            self.assertEqual(result.trigger_id, trigger)
            self.assertEqual(result.pattern_sha256, self.registry.config_sha256)
            self.assertTrue(all(
                board.evidence_kind == "MEASURED"
                and board.state == "TRACKING"
                and board.measurement_age_triggers == 0
                for board in result.boards.values()
            ))
            self.assertGreaterEqual(len(result.measured_marker_ids), 6)
            for board_id, board in result.boards.items():
                expected_t = self.origin[board_id] + np.array([
                    0.001 * (trigger - self.first),
                    0.0002 * (trigger - self.first), 0,
                ])
                np.testing.assert_allclose(
                    board.outcome.pose.translation_world_from_board,
                    expected_t, atol=1e-7,
                )
                np.testing.assert_allclose(
                    board.outcome.pose.rotation_world_from_board,
                    self.R[board_id], atol=1e-7,
                )
                self.assertFalse(board.outcome.pose.metric_scale_validated)

    def test_revalidation_rejects_unique_decode_beyond_erasure_guarantee(self):
        frames, rois = self.samples(self.first)
        candidates = self.validations(self.first)
        self.assertTrue(candidates)
        candidate = candidates[0]
        trace = list(candidate.trace)
        for phase in (0, 1):
            item = trace[phase]
            trace[phase] = replace(
                item, roi=None, predicted_roi_qualified=True
            )
        candidates[0] = replace(candidate, trace=tuple(trace))

        with self.assertRaisesRegex(ValueError, "guaranteed erasure budget"):
            self.tracker.advance(
                frames, rois,
                revalidation_traces=candidates,
                controller_config_sha256=self.registry.config_sha256,
            )
        self.assertIsNone(self.tracker.last_trigger_id)
        self.assertEqual(self.tracker.active_identity_locks, ())

    def test_revalidation_after_new_cycle_resets_identity_age(self):
        self.bootstrap()
        for trigger in range(self.first + 1, self.first + 17):
            frames, rois = self.samples(trigger)
            result = self.tracker.advance(
                frames, rois,
                revalidation_traces=self.validations(trigger)
                if trigger == self.first + 16 else (),
            )
        self.assertEqual(result.trigger_id, 1031)
        self.assertEqual(len(result.active_locks), 24)
        self.assertTrue(all(
            lock.confirmed_trigger_id == 1031
            for lock in result.active_locks
        ))
        self.assertTrue(all(
            status.state == "TRACKING" for status in result.boards.values()
        ))

    def test_lock_expires_without_revalidation_and_stops_measured_pose(self):
        self.tracker = self.create_tracker(max_identity_age_triggers=2)
        self.bootstrap()
        for trigger in range(self.first + 1, self.first + 4):
            frames, rois = self.samples(trigger)
            result = self.tracker.advance(frames, rois)
        self.assertEqual(result.active_locks, ())
        self.assertEqual(len(result.invalidated_marker_ids), 8)
        self.assertTrue(all(
            status.evidence_kind == "INVALID"
            and status.outcome.pose is None
            for status in result.boards.values()
        ))

    def test_missing_measurement_is_invalid_and_never_reuses_prior_pose(self):
        self.bootstrap()
        for trigger in range(self.first + 1, self.first + 4):
            frames, _ = self.samples(trigger)
            report = self.tracker.advance(frames, [])
            expected_state = "DEGRADED" if trigger <= self.first + 2 else "LOST"
            for board in report.boards.values():
                self.assertEqual(board.state, expected_state)
                self.assertEqual(board.evidence_kind, "INVALID")
                self.assertIsNone(board.outcome.pose)
                self.assertEqual(
                    board.measurement_age_triggers, trigger - self.first
                )
                self.assertEqual(
                    board.measurement_age_ns, (trigger - self.first) * 100_000
                )
                self.assertIsNotNone(board.last_measurement_exposure_timestamp_ns)
        frames, rois = self.samples(self.first + 4)
        recovered = self.tracker.advance(frames, rois)
        self.assertTrue(all(
            b.state == "TRACKING" and b.evidence_kind == "MEASURED"
            for b in recovered.boards.values()
        ))

    def test_one_board_geometry_failure_does_not_discard_other_board(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1)
        # A 120 px 2D outlier in one camera for B marker 105
        # must not suppress A's independently valid 6DoF.
        rois = list(rois)
        for i, entry in enumerate(rois):
            if (entry.track_evidence.camera_id, entry.track_evidence.track_id) == (5, 1105):
                x, y = entry.track_evidence.observed_xy
                rois[i] = replace(entry, track_evidence=replace(
                    entry.track_evidence, observed_xy=(x + 120, y - 50),
                ))
                break
        result = self.tracker.advance(frames, rois)
        self.assertEqual(result.boards["board_A"].state, "TRACKING")
        self.assertEqual(result.boards["board_B"].state, "DEGRADED")
        self.assertEqual(result.boards["board_B"].outcome.status, "geometry_rejected")
        self.assertIsNone(result.boards["board_B"].outcome.pose)

    def test_bad_code_bit_revokes_lock_not_relabels_to_other_board(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1, corrupt=(2, 100))
        rejected = self.tracker.advance(frames, rois)
        self.assertIn(100, rejected.invalidated_marker_ids)
        self.assertNotIn((2, 1100), [
            (lock.camera_id, lock.local_track_id) for lock in rejected.active_locks
        ])
        self.assertFalse(rejected.hardware_provenance_verified)

    def test_wrong_revision_duplicate_frame_or_borrowed_trace_rolls_back(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1)
        for case in ("revision", "duplicate_track", "borrowed_stamp"):
            with self.subTest(case=case), self.assertRaises(ValueError):
                if case == "revision":
                    self.tracker.advance(
                        frames, rois, controller_config_sha256="x" * 64
                    )
                elif case == "duplicate_track":
                    self.tracker.advance(frames, rois + [rois[0]])
                else:
                    wrong = replace(
                        rois[0], frame_stamp=self.stamp(2, self.first)
                    )
                    self.tracker.advance(frames, [wrong] + rois[1:])
            self.assertEqual(self.tracker.last_trigger_id, self.first)
        correct = self.tracker.advance(
            frames, rois, controller_config_sha256=self.registry.config_sha256
        )
        self.assertEqual(correct.trigger_id, self.first + 1)
        self.assertEqual(len(correct.active_locks), 24)

    def test_conflicting_revalidation_identity_is_rejected(self):
        self.bootstrap()
        # Full-cycle validation requires a new cycle's observed code history,
        # not just one phase from its beginning.
        for trigger in range(self.first + 1, self.first + 16):
            frames, rois = self.samples(trigger)
            self.tracker.advance(frames, rois)
        current = self.first + 16
        frames, rois = self.samples(current)
        traces = self.validations(current)
        first = traces[0]
        previous_marker = first.local_track_id - 1000
        new_marker = 101 if previous_marker == 100 else 100
        # The trace really carries another configured LED's complete
        # codeword, but keeps the same camera-local motion track. The
        # service must *decode* the conflict and refuse an ID swap.
        forged_samples = []
        for sample in first.trace:
            phase = (sample.frame_stamp.trigger_id - self.epoch) % self.registry.period_triggers
            new_bit = self.registry.codebook.patterns[new_marker][phase]
            contrast = 120.0 if new_bit else 0.0
            forged_samples.append(replace(
                sample, roi=ROIContrast(contrast, 0.0, contrast, 10, 30),
                predicted_roi_qualified=True,
            ))
        modified = replace(first, trace=tuple(forged_samples))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.tracker.advance(frames, rois, revalidation_traces=[modified])
        self.assertEqual(self.tracker.last_trigger_id, current - 1)
        accepted = self.tracker.advance(frames, rois, revalidation_traces=traces)
        self.assertTrue(all(x.state == "TRACKING" for x in accepted.boards.values()))

    def test_revalidation_requires_local_full_trace_not_fabricated_marker_id(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1)
        # Supplying the old external AdmittedMarker shape or an incomplete
        # single sample cannot inject a new global marker identity.
        with self.assertRaisesRegex(ValueError, "LocalTrackCycle"):
            self.tracker.advance(frames, rois, revalidation_traces=[object()])
        key = self.validations(self.first)[0]
        single_sample = replace(
            key, target_frame=frames[0], trace=(rois[0],)
        )
        with self.assertRaises(ValueError):
            self.tracker.advance(
                frames, rois, revalidation_traces=[single_sample]
            )
        self.assertEqual(self.tracker.last_trigger_id, self.first)

    def test_calibration_arrays_are_frozen_and_report_a_revision(self):
        first = self.bootstrap()
        self.assertEqual(len(first.calibration_sha256), 64)
        self.assertEqual(first.calibration_sha256, self.tracker.calibration_sha256)
        self.assertFalse(self.tracker.intrinsics_by_id[2].flags.writeable)
        self.assertFalse(self.tracker.camera_poses_by_id[2]["R"].flags.writeable)
        before = self.tracker.intrinsics_by_id[2][0, 0]
        self.K[2][0, 0] += 10
        self.assertEqual(self.tracker.intrinsics_by_id[2][0, 0], before)
        self.K[2][0, 0] -= 10

    def test_bad_roi_and_track_types_roll_back_before_state_commit(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1)
        for candidate in (
            replace(rois[0], roi="on"),
            replace(rois[0], track_evidence=None),
        ):
            with self.subTest(candidate=repr(candidate)):
                with self.assertRaisesRegex(ValueError, "ROI evidence"):
                    self.tracker.advance(frames, [candidate] + rois[1:])
                self.assertEqual(self.tracker.last_trigger_id, self.first)

    def test_invalid_real_policy_does_not_initialize_tracker(self):
        for parameter in (
            "max_reprojection_rms_px", "max_rigid_rms_world_units"
        ):
            for value in ("1", True, float("nan"), float("inf"), -1.0, 10**1000):
                with self.subTest(parameter=parameter, value=repr(value)):
                    opts = {parameter: value}
                    with self.assertRaisesRegex(ValueError, parameter):
                        IdentityLockedBoardTracker(
                            self.registry, self.camera_ids,
                            self.cameras, self.K, self.brightness,
                            trigger_epoch=self.epoch,
                            max_exposure_skew_ns=60,
                            **opts,
                        )

    def test_clock_change_and_missing_triggers_fail_before_state_commit(self):
        self.bootstrap()
        frames, rois = self.samples(self.first + 1)
        altered = [
            replace(f, clock_domain="other-clock") for f in frames
        ]
        with self.assertRaisesRegex(ValueError, "clock domain changed"):
            self.tracker.advance(altered, [])
        gap_frames, gap_rois = self.samples(self.first + 2)
        with self.assertRaisesRegex(ValueError, "trigger gap"):
            self.tracker.advance(gap_frames, gap_rois)
        self.assertEqual(self.tracker.last_trigger_id, self.first)
        ok = self.tracker.advance(frames, rois)
        self.assertEqual(ok.trigger_id, self.first + 1)


if __name__ == "__main__":
    unittest.main()
