"""Real BoundedLightTracker outputs into a two-board pose service (synthetic frames).

Detector centroids and FrameStamp are synthetic, not measured hardware facts.
"""
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import IdentityLockedBoardTracker
from light_intensity import BinaryIntensityCalibration, DarkROIQualification
from pose_stream import pose_frame_to_dict
from synchronized_light_pipeline import SynchronizedLocalLightPipeline
from temporal_pattern import FrameStamp


CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"


def yaw(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class SynchronizedLightPipelineTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        self.cameras = (2, 5, 9)
        self.K = {
            2: np.array([[500., 0., 128.], [0., 500., 128.], [0., 0., 1.]]),
            5: np.array([[560., 0., 128.], [0., 530., 128.], [0., 0., 1.]]),
            9: np.array([[510., 0., 128.], [0., 550., 128.], [0., 0., 1.]]),
        }
        self.poses = {
            2: {"R": np.eye(3), "t": np.zeros(3)},
            5: {"R": np.eye(3), "t": np.array([-0.12, 0.0, 0.0])},
            9: {"R": np.eye(3), "t": np.array([0.0, -0.11, 0.0])},
        }
        self.board_rotation = {"board_A": yaw(0.12), "board_B": yaw(-0.19)}
        self.board_origin = {
            "board_A": np.array([-0.16, -0.09, 1.85]),
            "board_B": np.array([0.20, 0.11, 2.10]),
        }
        self.epoch = 1000
        self.calibration = {c: BinaryIntensityCalibration(0., 120.) for c in self.cameras}
        self.identity = IdentityLockedBoardTracker(
            self.registry, self.cameras, self.poses, self.K, self.calibration,
            trigger_epoch=self.epoch,
            max_exposure_skew_ns=40,
            max_identity_age_triggers=32,
            max_reprojection_rms_px=0.05,
            max_rigid_rms_world_units=0.002,
        )
        self.pipeline = SynchronizedLocalLightPipeline(
            self.identity,
            max_tracks_per_camera=8,
            max_motion_px_per_trigger=12.0,
            max_missing_triggers=3,
            minimum_detection_separation_px=0.5,
            ambiguity_margin_fraction=0.03,
            min_decode_observations=8,
        )

    def position(self, marker, trigger):
        board_id = self.registry.marker_to_board[marker]
        local = np.asarray(self.registry.boards[board_id].marker_xyz_m[marker])
        motion = np.array([
            (trigger - 1015) * 0.0004,
            (trigger - 1015) * 0.0002,
            0.,
        ])
        return self.board_rotation[board_id] @ local + self.board_origin[board_id] + motion

    def project(self, camera, marker, trigger):
        pose = self.poses[camera]
        p = pose["R"] @ self.position(marker, trigger) + pose["t"]
        pixel = self.K[camera] @ p
        xy = pixel[:2] / pixel[2]
        self.assertTrue(np.all(xy >= 12.))
        self.assertTrue(np.all(xy <= 244.))
        return tuple(xy.tolist())

    def frames_and_pixels(self, trigger, *, drop=frozenset(), empty_cameras=frozenset()):
        frames = [
            FrameStamp(
                camera, trigger,
                1_000_000_000 + (trigger - self.epoch) * 100_000
                + index * 7,
                "synthetic-common-exposure",
            )
            for index, camera in enumerate(self.cameras)
        ]
        images = {}
        detections = {}
        phase = (trigger - self.epoch) % self.registry.period_triggers
        for camera in self.cameras:
            gray = np.full((256, 256), 15., dtype=np.float32)
            points = []
            if camera not in empty_cameras:
                for marker in self.registry.marker_to_board:
                    if (
                        not self.registry.codebook.patterns[marker][phase]
                        or (camera, marker) in drop
                    ):
                        continue
                    xy = self.project(camera, marker, trigger)
                    x, y = np.rint(xy).astype(int)
                    gray[y-2:y+3, x-2:x+3] += 120.
                    points.append(xy)
            # Invert detector enumeration across exposures; identities must
            # come from local track motion and subsequent codebook decoding.
            if trigger % 2:
                points.reverse()
            images[camera] = gray
            detections[camera] = points
        return frames, images, detections

    def qualified_dark_roi_evidence(self, trigger):
        """Build explicit synthetic ground-truth qualification records."""
        phase = (trigger - self.epoch) % self.registry.period_triggers
        qualifications = []
        for camera in self.cameras:
            local = self.pipeline._local_trackers[camera]
            expected = {
                marker: np.asarray(self.project(camera, marker, trigger))
                for marker in self.registry.marker_to_board
            }
            candidates = {}
            marker_claims = {}
            for track_id, state in local._tracks.items():
                gap = trigger - state.last_observed_trigger
                if gap < 1 or gap > local.max_missing_triggers:
                    continue
                predicted = state.last_xy + gap * state.velocity_per_trigger
                ranked = sorted(
                    (float(np.linalg.norm(predicted - xy)), marker)
                    for marker, xy in expected.items()
                )
                if not ranked or ranked[0][0] > 0.75:
                    continue
                if len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= 0.5:
                    continue
                marker = ranked[0][1]
                candidates[track_id] = marker
                marker_claims.setdefault(marker, []).append(track_id)
            for track_id, marker in sorted(candidates.items()):
                if (
                    len(marker_claims[marker]) != 1
                    or self.registry.codebook.patterns[marker][phase] != 0
                ):
                    continue
                expected_xy = tuple(float(x) for x in expected[marker])
                payload = json.dumps({
                    "camera_id": camera,
                    "local_track_id": track_id,
                    "trigger_id": trigger,
                    "marker_id": marker,
                    "expected_xy": expected_xy,
                    "pattern_sha256": self.registry.config_sha256,
                    "verifier_id": "synthetic-ground-truth-v1",
                }, sort_keys=True, separators=(",", ":")).encode("utf-8")
                qualifications.append(DarkROIQualification(
                    camera_id=camera,
                    local_track_id=track_id,
                    trigger_id=trigger,
                    marker_id=marker,
                    expected_xy=expected_xy,
                    pattern_sha256=self.registry.config_sha256,
                    verifier_id="synthetic-ground-truth-v1",
                    reference_evidence_sha256=hashlib.sha256(payload).hexdigest(),
                ))
        return tuple(qualifications)

    def process(self, trigger, **kwargs):
        frames, images, detections = self.frames_and_pixels(trigger)
        kwargs.setdefault(
            "dark_roi_qualifications",
            self.qualified_dark_roi_evidence(trigger),
        )
        return self.pipeline.process(
            frames, images, detections,
            controller_config_sha256=self.registry.config_sha256,
            **kwargs,
        )

    def bootstrap(self):
        last = None
        for trigger in range(self.epoch, self.epoch + 16):
            last = self.process(trigger)
            if trigger < self.epoch + 15:
                self.assertTrue(all(
                    body.evidence_kind == "INVALID" for body in last.boards.values()
                ))
        self.assertEqual(last.trigger_id, 1015)
        self.assertEqual(len(last.active_locks), 24)
        self.assertTrue(all(
            x.evidence_kind == "MEASURED" and x.state == "TRACKING"
            for x in last.boards.values()
        ))
        self.assertLessEqual(
            self.pipeline.retained_roi_sample_count,
            self.registry.period_triggers * 8 * len(self.cameras)
        )
        return last

    def test_full_rig_acquisition_and_every_trigger_fresh_pose_from_actual_pixels(self):
        initial = self.bootstrap()
        self.assertEqual(len(initial.active_locks), 24)
        for trigger in range(1016, 1028):
            output = self.process(trigger)
            self.assertEqual(output.trigger_id, trigger)
            self.assertEqual(len(output.active_locks), 24)
            self.assertTrue(all(
                b.evidence_kind == "MEASURED"
                and b.state == "TRACKING"
                and b.measurement_age_ns == 0
                for b in output.boards.values()
            ))
            self.assertGreaterEqual(len(output.measured_marker_ids), 6)
            for board_id, body in output.boards.items():
                wanted = self.board_origin[board_id] + np.array([
                    (trigger - 1015) * 0.0004,
                    (trigger - 1015) * 0.0002,
                    0.,
                ])
                np.testing.assert_allclose(
                    body.outcome.pose.translation_world_from_board,
                    wanted, atol=1e-7,
                )
            as_json = pose_frame_to_dict(output, self.registry)
            self.assertEqual(as_json["kind"], "pose_frame")
            self.assertEqual(as_json["frame_id"], trigger)
            self.assertTrue(all(
                b["measurement_age_ns"] == 0
                for b in as_json["rigid_bodies"]
            ))
            self.assertFalse(as_json["hardware_provenance_verified"])

    def test_duplicate_camera_detections_quarantine_only_that_camera(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        original_5 = self.pipeline.active_local_track_ids(5)
        bad = dict(detections)
        bad[5] = list(bad[5])
        bad[5].append(bad[5][0])  # duplicate/overlapping detector centroids

        # A rejected local assignment MUST consume this exposure and
        # invalidate all of camera 5's old LED identities. A pure rollback
        # would risk using their stale IDs on the next trigger.
        frame = self.pipeline.process(frames, images, bad)
        self.assertEqual(frame.trigger_id, 1016)
        self.assertEqual(self.identity.last_trigger_id, 1016)
        self.assertEqual(frame.quarantined_camera_ids, (5,))
        self.assertEqual(self.pipeline.quarantined_camera_ids, (5,))
        self.assertEqual(self.pipeline.active_local_track_ids(5), ())
        self.assertGreater(len(original_5), 0)
        self.assertFalse(any(
            lock.camera_id == 5 for lock in frame.active_locks
        ))
        self.assertTrue(all(
            b.evidence_kind == "MEASURED" for b in frame.boards.values()
        ))
        emitted = pose_frame_to_dict(frame, self.registry)
        self.assertEqual(emitted["quarantined_camera_ids"], [5])

        next_frame = self.process(1017)
        self.assertEqual(next_frame.quarantined_camera_ids, (5,))
        self.assertTrue(self.pipeline.active_local_track_ids(5))
        self.assertFalse(any(
            lock.camera_id == 5 for lock in next_frame.active_locks
        ))

    def test_crossing_near_tie_forces_persistent_camera_quarantine(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        old_ids = self.pipeline.active_local_track_ids(2)
        old_locks = tuple(
            lock for lock in self.identity.active_identity_locks
            if lock.camera_id == 2
        )
        self.assertEqual({x.local_track_id for x in old_locks}, set(old_ids))

        # Two near-midpoint detections have an equally plausible assignment
        # to different local IDs. Even if a later frame looks good, these
        # identities must remain untrusted until a new code cycle is decoded.
        p105 = np.asarray(self.project(2, 105, 1016))
        p107 = np.asarray(self.project(2, 107, 1016))
        middle = 0.5 * (p105 + p107)
        points = list(detections[2])
        left = min(range(len(points)), key=lambda i: np.linalg.norm(
            np.asarray(points[i]) - p105
        ))
        right = min(range(len(points)), key=lambda i: np.linalg.norm(
            np.asarray(points[i]) - p107
        ))
        self.assertNotEqual(left, right)
        points[left] = (float(middle[0]), float(middle[1] + 0.35))
        points[right] = (float(middle[0]), float(middle[1] - 0.35))
        bad = dict(detections)
        bad[2] = points

        output = self.pipeline.process(frames, images, bad)
        self.assertEqual(output.quarantined_camera_ids, (2,))
        self.assertEqual(self.identity.last_trigger_id, 1016)
        self.assertEqual(self.pipeline.active_local_track_ids(2), ())
        self.assertFalse(any(
            lock.camera_id == 2 for lock in output.active_locks
        ))
        self.assertTrue(all(
            status.evidence_kind == "MEASURED"
            for status in output.boards.values()
        ))

        next_output = self.process(1017)
        self.assertEqual(next_output.quarantined_camera_ids, (2,))
        new_ids = self.pipeline.active_local_track_ids(2)
        self.assertTrue(set(new_ids).isdisjoint(set(old_ids)))
        self.assertFalse(any(
            lock.camera_id == 2 for lock in next_output.active_locks
        ))

    def test_multi_trigger_quarantine_only_releases_after_new_full_cycle(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        bad = dict(detections)
        bad[2] = list(bad[2]) + [bad[2][0]]
        first = self.pipeline.process(frames, images, bad)
        self.assertEqual(first.quarantined_camera_ids, (2,))
        self.assertFalse(any(l.camera_id == 2 for l in first.active_locks))

        # A *second* independent unsafe assignment at the next trigger
        # must keep the same camera quarantined rather than resurrecting IDs.
        frames, images, detections = self.frames_and_pixels(1017)
        bad_again = dict(detections)
        bad_again[2] = list(bad_again[2]) + [bad_again[2][0]]
        second = self.pipeline.process(frames, images, bad_again)
        self.assertEqual(second.quarantined_camera_ids, (2,))
        self.assertEqual(self.identity.last_trigger_id, 1017)
        self.assertFalse(any(l.camera_id == 2 for l in second.active_locks))
        self.assertEqual(self.pipeline.active_local_track_ids(2), ())

        last = None
        for trigger in range(1018, 1032):
            last = self.process(trigger)
        self.assertEqual(last.quarantined_camera_ids, (2,))
        self.assertFalse(any(l.camera_id == 2 for l in last.active_locks))
        self.assertTrue(all(x.evidence_kind == "MEASURED" for x in last.boards.values()))

        # Partial post-crossing samples cannot establish identity even if
        # current pixels are geometrically near the prior LED location.
        for trigger in range(1032, 1047):
            last = self.process(trigger)
            self.assertEqual(last.quarantined_camera_ids, (2,))
            self.assertFalse(any(l.camera_id == 2 for l in last.active_locks))
        restored = self.process(1047)
        self.assertEqual(restored.quarantined_camera_ids, ())
        self.assertEqual(self.pipeline.quarantined_camera_ids, ())
        confirmed = [l for l in restored.active_locks if l.camera_id == 2]
        self.assertEqual(len(confirmed), 8)
        self.assertEqual(
            {l.marker_id for l in confirmed},
            set(self.registry.marker_to_board),
        )
        self.assertTrue(all(
            x.evidence_kind == "MEASURED"
            for x in restored.boards.values()
        ))

    def test_two_quarantined_cameras_withhold_pose_until_sufficient_views(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        unsafe = dict(detections)
        for camera in (2, 5):
            unsafe[camera] = list(unsafe[camera]) + [unsafe[camera][0]]
        first = self.pipeline.process(frames, images, unsafe)
        self.assertEqual(first.quarantined_camera_ids, (2, 5))
        self.assertEqual(len(first.active_locks), 8)
        for body in first.boards.values():
            self.assertEqual(body.evidence_kind, "INVALID")
            self.assertIsNone(body.outcome.pose)
            self.assertEqual(body.state, "DEGRADED")
        wire = pose_frame_to_dict(first, self.registry)
        self.assertEqual(wire["quarantined_camera_ids"], [2, 5])
        for body in wire["rigid_bodies"]:
            self.assertIsNone(body["position_world_from_body"])
            self.assertIsNone(body["quality"])
            self.assertGreater(body["measurement_age_ns"], 0)

        next_frame = self.process(1017)
        self.assertEqual(next_frame.quarantined_camera_ids, (2, 5))
        self.assertTrue(all(
            body.evidence_kind == "INVALID"
            and body.outcome.pose is None
            for body in next_frame.boards.values()
        ))

    def test_unsafe_assignment_followed_by_other_camera_error_rolls_back(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        bad = dict(detections)
        bad[2] = list(bad[2]) + [bad[2][0]]  # would quarantine camera 2
        bad[9] = [["bad", "pixel"]]  # unrelated fatal input on next camera
        previous = tuple(
            self.pipeline.active_local_track_ids(camera)
            for camera in self.cameras
        )
        with self.assertRaisesRegex(ValueError, "numeric coordinates"):
            self.pipeline.process(frames, images, bad)
        self.assertEqual(self.identity.last_trigger_id, 1015)
        self.assertEqual(self.pipeline.quarantined_camera_ids, ())
        self.assertEqual(
            previous,
            tuple(
                self.pipeline.active_local_track_ids(camera)
                for camera in self.cameras
            ),
        )
        valid = self.pipeline.process(frames, images, detections)
        self.assertEqual(valid.quarantined_camera_ids, ())
        self.assertTrue(all(
            b.evidence_kind == "MEASURED"
            for b in valid.boards.values()
        ))

    def test_unsafe_assignment_plus_wrong_registry_revision_rolls_back(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        unsafe = dict(detections)
        unsafe[5] = list(unsafe[5]) + [unsafe[5][0]]
        prev_ids = self.pipeline.active_local_track_ids(5)
        with self.assertRaisesRegex(ValueError, "controller pattern SHA-256"):
            self.pipeline.process(
                frames, images, unsafe,
                controller_config_sha256="0" * 64,
            )
        self.assertEqual(self.identity.last_trigger_id, 1015)
        self.assertEqual(self.pipeline.quarantined_camera_ids, ())
        self.assertEqual(self.pipeline.active_local_track_ids(5), prev_ids)
        success = self.pipeline.process(
            frames, images, detections,
            controller_config_sha256=self.registry.config_sha256,
        )
        self.assertEqual(success.trigger_id, 1016)
        self.assertEqual(success.quarantined_camera_ids, ())

    def test_fatal_input_error_still_rolls_back_all_camera_states(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        malformed = dict(detections)
        malformed[5] = [["bad", "pixel"]]
        before = tuple(self.pipeline.active_local_track_ids(c) for c in self.cameras)
        history_size = self.pipeline.retained_roi_sample_count
        with self.assertRaisesRegex(ValueError, "numeric coordinates"):
            self.pipeline.process(frames, images, malformed)
        self.assertEqual(self.identity.last_trigger_id, 1015)
        self.assertEqual(self.pipeline.quarantined_camera_ids, ())
        self.assertEqual(history_size, self.pipeline.retained_roi_sample_count)
        self.assertEqual(
            tuple(self.pipeline.active_local_track_ids(c) for c in self.cameras),
            before,
        )
        accepted = self.pipeline.process(frames, images, detections)
        self.assertEqual(accepted.trigger_id, 1016)

    def test_detector_capacity_failure_does_not_partially_commit(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        bad = dict(detections)
        # More detections than the bounded number of registered LEDs.
        bad[5] = list(bad[5]) + [
            (10.0 + i * 10, 12.0) for i in range(9)
        ]
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.pipeline.process(frames, images, bad)
        self.assertEqual(self.identity.last_trigger_id, 1015)
        after = self.pipeline.process(frames, images, detections)
        self.assertEqual(after.trigger_id, 1016)

    def test_newborn_track_does_not_inherit_old_expired_identity(self):
        self.bootstrap()
        old_camera_ids = set(self.pipeline.active_local_track_ids(2))
        last = None
        for trigger in range(1016, 1021):
            frames, images, detections = self.frames_and_pixels(
                trigger,
                empty_cameras=frozenset((2,)) if trigger <= 1019 else frozenset(),
            )
            last = self.pipeline.process(frames, images, detections)
        new_ids = set(self.pipeline.active_local_track_ids(2))
        self.assertNotEqual(new_ids, old_camera_ids)
        self.assertTrue(new_ids - old_camera_ids)
        self.assertTrue(all(x.state == "TRACKING" for x in last.boards.values()))
        # Without a full new code cycle, the birth has *no* global identity
        # lock, even if it is spatially close to an old LED trajectory.
        locked = {x.local_track_id for x in last.active_locks if x.camera_id == 2}
        self.assertTrue((new_ids - old_camera_ids).isdisjoint(locked))

    def test_expired_real_track_revokes_old_identity_before_new_birth(self):
        self.bootstrap()
        old_camera_ids = set(self.pipeline.active_local_track_ids(2))
        old_locked = {
            x.local_track_id for x in self.identity.active_identity_locks
            if x.camera_id == 2
        }
        self.assertEqual(old_camera_ids, old_locked)
        for trigger in range(1016, 1020):
            frames, images, detections = self.frames_and_pixels(
                trigger, empty_cameras=frozenset((2,))
            )
            result = self.pipeline.process(frames, images, detections)
        self.assertTrue(all(
            lock.camera_id != 2 for lock in result.active_locks
        ))
        self.assertGreater(len(result.invalidated_marker_ids), 0)
        # Reappearing spots create fresh local IDs, but cannot immediately
        # adopt the previously retired global marker IDs.
        frames, images, detections = self.frames_and_pixels(1020)
        result = self.pipeline.process(frames, images, detections)
        fresh_local_ids = set(self.pipeline.active_local_track_ids(2))
        self.assertTrue(fresh_local_ids - old_camera_ids)
        self.assertTrue(all(
            lock.camera_id != 2 for lock in result.active_locks
        ))
        self.assertTrue(all(
            x.state == "TRACKING" for x in result.boards.values()
        ))

    def test_retired_camera_tracks_only_relock_after_full_new_code_cycle(self):
        self.bootstrap()
        initial_ids = set(self.pipeline.active_local_track_ids(2))
        # Four entire missing camera exposures retire all old local tracks.
        for trigger in range(1016, 1020):
            frames, images, detections = self.frames_and_pixels(
                trigger, empty_cameras=frozenset((2,))
            )
            self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=self.qualified_dark_roi_evidence(trigger),
            )
        self.assertTrue(all(
            lock.camera_id != 2
            for lock in self.identity.active_identity_locks
        ))

        partial_result = None
        for trigger in range(1020, 1032):
            frames, images, detections = self.frames_and_pixels(trigger)
            partial_result = self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=self.qualified_dark_roi_evidence(trigger),
            )
        self.assertTrue(set(self.pipeline.active_local_track_ids(2)) - initial_ids)
        # Erasures during phases 0..3 remove the distinguishing bits of
        # some codewords. A partial-code ID is NOT transferred to the new
        # local track even though its location resembles the old light.
        self.assertFalse(any(
            lock.camera_id == 2 for lock in partial_result.active_locks
        ))

        restored = None
        for trigger in range(1032, 1048):
            frames, images, detections = self.frames_and_pixels(trigger)
            restored = self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=self.qualified_dark_roi_evidence(trigger),
            )
        new_camera_locks = [
            lock for lock in restored.active_locks if lock.camera_id == 2
        ]
        self.assertEqual(len(new_camera_locks), 8)
        self.assertEqual(
            {lock.marker_id for lock in new_camera_locks},
            set(self.registry.marker_to_board),
        )
        self.assertTrue(all(
            lock.local_track_id not in initial_ids
            for lock in new_camera_locks
        ))
        self.assertEqual(len(restored.active_locks), 24)
        self.assertTrue(all(
            outcome.state == "TRACKING"
            for outcome in restored.boards.values()
        ))

    def test_bad_source_timestamp_and_wrong_config_roll_back(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        for bad_case in ("config", "different_trigger", "clock"):
            with self.subTest(bad_case=bad_case):
                wrong_frames = list(frames)
                if bad_case == "different_trigger":
                    wrong_frames[1] = FrameStamp(
                        5, 1017, frames[1].exposure_timestamp_ns, frames[1].clock_domain
                    )
                if bad_case == "clock":
                    wrong_frames[1] = FrameStamp(
                        5, 1016, frames[1].exposure_timestamp_ns, "other-clock"
                    )
                with self.assertRaises(ValueError):
                    self.pipeline.process(
                        wrong_frames, images, detections,
                        controller_config_sha256=(
                            "f" * 64 if bad_case == "config"
                            else self.registry.config_sha256
                        ),
                    )
                self.assertEqual(self.identity.last_trigger_id, 1015)
        retry = self.pipeline.process(frames, images, detections)
        self.assertEqual(retry.trigger_id, 1016)

    def test_dark_roi_qualification_rejects_stale_trigger_and_pattern(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        qualifications = self.qualified_dark_roi_evidence(1016)
        self.assertTrue(qualifications)
        candidate = qualifications[0]
        for bad in (
            replace(candidate, trigger_id=1017),
            replace(candidate, pattern_sha256="f" * 64),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.pipeline.process(
                    frames, images, detections,
                    dark_roi_qualifications=(bad,),
                    controller_config_sha256=self.registry.config_sha256,
                )
            self.assertEqual(self.identity.last_trigger_id, 1015)

    def test_dark_roi_qualification_rejects_bad_prediction_residual(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        candidate = self.qualified_dark_roi_evidence(1016)[0]
        bad = replace(candidate, expected_xy=(
            candidate.expected_xy[0] + 50.0, candidate.expected_xy[1]
        ))
        with self.assertRaisesRegex(ValueError, "residual exceeds configured limit"):
            self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=(bad,),
                controller_config_sha256=self.registry.config_sha256,
            )
        self.assertEqual(self.identity.last_trigger_id, 1015)

    def test_dark_roi_qualification_rejects_on_phase_and_duplicate_claims(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        qualifications = self.qualified_dark_roi_evidence(1016)
        self.assertTrue(qualifications)
        candidate = qualifications[0]
        on_marker = next(
            marker for marker, code in self.registry.codebook.patterns.items()
            if code[(1016 - self.epoch) % self.registry.period_triggers] == 1
        )
        bad_on = replace(candidate, marker_id=on_marker)
        with self.assertRaisesRegex(ValueError, "marker that is OFF"):
            self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=(bad_on,),
                controller_config_sha256=self.registry.config_sha256,
            )
        duplicate = replace(candidate, local_track_id=candidate.local_track_id + 10000)
        with self.assertRaisesRegex(ValueError, "multiple tracks claim"):
            self.pipeline.process(
                frames, images, detections,
                dark_roi_qualifications=(candidate, duplicate),
                controller_config_sha256=self.registry.config_sha256,
            )
        self.assertEqual(self.identity.last_trigger_id, 1015)

    def test_default_dark_samples_are_erasures_before_initial_code_lock(self):
        # The first phase includes known OFF LED patterns, which are not
        # observed points. Only actual ON point centroids are provided.
        output = self.process(self.epoch)
        self.assertEqual(output.active_locks, ())
        self.assertTrue(all(x.state == "DISCOVER" for x in output.boards.values()))
        self.assertGreater(self.pipeline.retained_roi_sample_count, 0)


if __name__ == "__main__":
    unittest.main()
