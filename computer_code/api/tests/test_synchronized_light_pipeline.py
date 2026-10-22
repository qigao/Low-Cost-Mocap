"""Real BoundedLightTracker outputs into a two-board pose service (synthetic frames).

Detector centroids and FrameStamp are synthetic, not measured hardware facts.
"""
import unittest
from pathlib import Path

import numpy as np

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import IdentityLockedBoardTracker
from light_intensity import BinaryIntensityCalibration
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

    def process(self, trigger, **kwargs):
        frames, images, detections = self.frames_and_pixels(trigger)
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

    def test_atomic_rollback_of_camera_detection_collision(self):
        self.bootstrap()
        frames, images, detections = self.frames_and_pixels(1016)
        good = tuple(self.pipeline.active_local_track_ids(c) for c in self.cameras)
        count = self.pipeline.retained_roi_sample_count
        bad = dict(detections)
        bad[5] = list(bad[5])
        bad[5].append(bad[5][0])  # duplicate/overlapping detector centroids
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.pipeline.process(frames, images, bad)
        self.assertEqual(self.identity.last_trigger_id, 1015)
        self.assertEqual(count, self.pipeline.retained_roi_sample_count)
        self.assertEqual(good, tuple(
            self.pipeline.active_local_track_ids(c) for c in self.cameras
        ))
        accepted = self.pipeline.process(frames, images, detections)
        self.assertEqual(accepted.trigger_id, 1016)
        self.assertTrue(all(x.state == "TRACKING" for x in accepted.boards.values()))

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

    def test_default_dark_samples_are_erasures_before_initial_code_lock(self):
        # The first phase includes known OFF LED patterns, which are not
        # observed points. Only actual ON point centroids are provided.
        output = self.process(self.epoch)
        self.assertEqual(output.active_locks, ())
        self.assertTrue(all(x.state == "DISCOVER" for x in output.boards.values()))
        self.assertGreater(self.pipeline.retained_roi_sample_count, 0)


if __name__ == "__main__":
    unittest.main()
