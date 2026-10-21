"""Software conformance of multi-board LED registration and 6DoF recognition."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import MappingProxyType

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import (
    LocalTrackCycle, recognize_boards, recognize_admitted_boards,
    recognize_tracked_boards,
)
from light_intensity import BinaryIntensityCalibration, ROIContrast, StampedTrackROI
from light_tracking import TrackEvidence
from marker_admission import admit_track_marker_at_frame
from spatiotemporal_reconstruction import (
    SynchronizedGeometry, reconstruct_synchronized_markers,
)
from temporal_pattern import (
    FrameStamp, group_synchronized_detections,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "board-patterns.example.json"


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class BoardPatternRegistryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_global_codebook_and_led_controller_phase(self):
        registry = BoardPatternRegistry.from_json_file(EXAMPLE)
        self.assertEqual(tuple(registry.boards), ("board_A", "board_B"))
        self.assertEqual(registry.period_triggers, 16)
        self.assertEqual(registry.pose_phase, 15)
        self.assertEqual(registry.board_for_marker(106).board_id, "board_B")
        self.assertEqual(len(registry.codebook.patterns), 8)
        for board_id in registry.boards:
            states = registry.led_states_at_trigger(
                board_id, trigger_id=1015, trigger_epoch=1000
            )
            self.assertTrue(all(states.values()))
            self.assertEqual(len(states), 4)
        with self.assertRaises(TypeError):
            registry.marker_to_board[500] = "board_A"
        with self.assertRaisesRegex(ValueError, "unknown global marker"):
            registry.board_for_marker(999)

    def test_codeword_distance_duty_and_common_all_on_phase(self):
        registry = BoardPatternRegistry.from_json_file(EXAMPLE)
        codes = list(registry.codebook.patterns.values())
        distances = [
            sum(a != b for a, b in zip(first, second))
            for i, first in enumerate(codes) for second in codes[i + 1:]
        ]
        self.assertGreaterEqual(min(distances), 8)
        for code in codes:
            self.assertEqual(sum(code), 8)
            self.assertEqual(code[-1], 1)

    def test_rejects_global_collision_weak_codes_and_non_common_pose_phase(self):
        duplicate = copy.deepcopy(self.data)
        duplicate["boards"]["board_B"]["markers"]["100"] = (
            duplicate["boards"]["board_B"]["markers"].pop("104")
        )
        with self.assertRaisesRegex(ValueError, "duplicate global marker_id"):
            BoardPatternRegistry.from_mapping(duplicate)
        duplicate_code = copy.deepcopy(self.data)
        duplicate_code["boards"]["board_B"]["markers"]["104"]["code"] = (
            duplicate_code["boards"]["board_A"]["markers"]["100"]["code"]
        )
        with self.assertRaisesRegex(ValueError, "Hamming"):
            BoardPatternRegistry.from_mapping(duplicate_code)
        phase_mismatch = copy.deepcopy(self.data)
        phase_mismatch["pose_phase"] = 0
        with self.assertRaisesRegex(ValueError, "pose_phase"):
            BoardPatternRegistry.from_mapping(phase_mismatch)

    def test_rejects_bad_coordinates_degenerate_geometry_and_minimum(self):
        for value in ("0", True, float("nan"), 10**1000):
            invalid = copy.deepcopy(self.data)
            invalid["boards"]["board_B"]["markers"]["105"]["xyz_m"][0] = value
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "coordinate"):
                    BoardPatternRegistry.from_mapping(invalid)
        collinear = copy.deepcopy(self.data)
        for i, marker in enumerate(collinear["boards"]["board_A"]["markers"].values()):
            marker["xyz_m"] = [i * 0.01, 0, 0]
        with self.assertRaisesRegex(ValueError, "collinear"):
            BoardPatternRegistry.from_mapping(collinear)
        too_many_required = copy.deepcopy(self.data)
        too_many_required["boards"]["board_A"]["min_visible_markers"] = 5
        with self.assertRaisesRegex(ValueError, "min_visible_markers"):
            BoardPatternRegistry.from_mapping(too_many_required)

    def test_rejects_reflection_unknown_fields_and_invalid_ids(self):
        reflected = copy.deepcopy(self.data)
        reflected["boards"]["board_A"]["body_from_board"]["rotation"][0] = [-1, 0, 0]
        with self.assertRaisesRegex(ValueError, "proper rigid rotation"):
            BoardPatternRegistry.from_mapping(reflected)
        extra = copy.deepcopy(self.data)
        extra["hidden_fallback"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            BoardPatternRegistry.from_mapping(extra)
        invalid_marker = copy.deepcopy(self.data)
        invalid_marker["boards"]["board_A"]["markers"]["0100"] = (
            invalid_marker["boards"]["board_A"]["markers"].pop("100")
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            BoardPatternRegistry.from_mapping(invalid_marker)

    def test_json_loader_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"version":1,"version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                BoardPatternRegistry.from_json_file(path)
            path.write_text(
                json.dumps(self.data).replace("0.14", "NaN"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid nonfinite JSON"):
                BoardPatternRegistry.from_json_file(path)

    def test_controller_requires_explicit_epoch_not_receive_time(self):
        registry = BoardPatternRegistry.from_json_file(EXAMPLE)
        with self.assertRaisesRegex(ValueError, "precedes"):
            registry.led_states_at_trigger("board_B", 99, trigger_epoch=100)
        with self.assertRaises(TypeError):
            registry.led_states_at_trigger("board_A", 100)


class BoardRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(EXAMPLE)
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
        self.R = {"board_A": rz(0.23), "board_B": rz(-0.31)}
        self.t = {
            "board_A": np.array([-0.2, 0.05, 3.4]),
            "board_B": np.array([0.45, 0.27, 3.8]),
        }
        self.world = {}
        for board_id, board in self.registry.boards.items():
            for marker, xyz in board.marker_xyz_m.items():
                self.world[marker] = self.R[board_id] @ np.asarray(xyz) + self.t[board_id]
        self.epoch = 1000
        self.trigger = 1015
        self.frames = [
            FrameStamp(cam, self.trigger, 20_000_000 + i * 20, "synthetic-rig")
            for i, cam in enumerate((2, 5, 9))
        ]
        self.calibration = BinaryIntensityCalibration(0.0, 120.0)

    def project(self, camera, xyz):
        pose = self.cameras[camera]
        p = self.K[camera] @ (pose["R"] @ xyz + pose["t"])
        return tuple((p[:2] / p[2]).tolist())

    def admissions(self, *, as_traces=False):
        out = []
        for final in self.frames:
            camera = final.camera_id
            for marker, xyz in self.world.items():
                if marker == 107 and camera == 5:
                    continue  # Single missing camera view, not a missing physical ID.
                xy = self.project(camera, xyz)
                code = self.registry.codebook.patterns[marker]
                trace = []
                for phase, bit in enumerate(code):
                    trigger = self.epoch + phase
                    stamp = FrameStamp(
                        camera, trigger,
                        final.exposure_timestamp_ns - (15 - phase) * 1000,
                        final.clock_domain,
                    )
                    track = TrackEvidence(
                        marker, camera, trigger, xy,
                        xy if bit else None, 0 if bit else 1,
                    )
                    contrast = 120.0 if bit else 0.0
                    roi = ROIContrast(contrast, 0.0, contrast, 10, 30)
                    trace.append(StampedTrackROI(
                        stamp, track, roi, predicted_roi_qualified=not bool(bit),
                    ))
                if as_traces:
                    out.append(LocalTrackCycle(camera, marker, final, tuple(trace)))
                else:
                    out.append(admit_track_marker_at_frame(
                        trace, self.registry.codebook, self.calibration,
                        camera_id=camera, local_track_id=marker,
                        target_frame=final, trigger_epoch=self.epoch,
                    ))
        return out

    def recognize(self, admissions=None):
        return recognize_admitted_boards(
            self.admissions() if admissions is None else admissions,
            self.frames, [2, 5, 9], self.cameras, self.K, self.registry,
            trigger_epoch=self.epoch, max_exposure_skew_ns=60,
            max_reprojection_rms_px=1e-5,
            max_rigid_rms_world_units=1e-5,
        )

    def test_two_simultaneous_boards_recover_body_and_board_poses(self):
        result = self.recognize()
        self.assertEqual(result.trigger_id, self.trigger)
        self.assertEqual(result.camera_ids, (2, 5, 9))
        self.assertEqual(tuple(result.boards), ("board_A", "board_B"))
        for board_id in ("board_A", "board_B"):
            board = self.registry.boards[board_id]
            outcome = result.boards[board_id]
            self.assertEqual(outcome.status, "recognized")
            self.assertIsNotNone(outcome.pose)
            self.assertEqual(len(outcome.pose.marker_ids), 4)
            np.testing.assert_allclose(
                outcome.pose.rotation_world_from_board, self.R[board_id], atol=1e-8
            )
            np.testing.assert_allclose(
                outcome.pose.translation_world_from_board, self.t[board_id], atol=1e-8
            )
            np.testing.assert_allclose(
                outcome.pose.translation_world_from_body,
                self.t[board_id]
                - self.R[board_id] @ board.translation_body_from_board_m,
                atol=1e-8,
            )
            self.assertFalse(outcome.pose.metric_scale_validated)

    def test_three_markers_survive_occlusion_two_markers_reject_one_board(self):
        admissions = self.admissions()
        without_one = [a for a in admissions if a.detection.marker_id != 107]
        partial = self.recognize(without_one)
        self.assertEqual(partial.boards["board_A"].status, "recognized")
        self.assertEqual(partial.boards["board_B"].status, "recognized")
        self.assertEqual(partial.boards["board_B"].pose.marker_ids, (104, 105, 106))
        without_two = [a for a in admissions if a.detection.marker_id not in (106, 107)]
        result = self.recognize(without_two)
        self.assertEqual(result.boards["board_A"].status, "recognized")
        self.assertEqual(result.boards["board_B"].status, "insufficient_markers")
        self.assertIsNone(result.boards["board_B"].pose)

    def test_geometric_outlier_rejects_only_affected_board(self):
        admissions = self.admissions()
        batch, grouped = group_synchronized_detections(
            [a.detection for a in admissions],
            self.frames, [2, 5, 9], max_exposure_skew_ns=60,
        )
        reconstruction = reconstruct_synchronized_markers(
            batch, grouped, self.cameras, self.K, max_reprojection_rms_px=1e-5
        )
        markers = dict(reconstruction.markers)
        bad = markers[104]
        markers[104] = replace(
            bad, xyz=(bad.xyz[0] + 0.1, bad.xyz[1], bad.xyz[2])
        )
        bad_scene = SynchronizedGeometry(
            reconstruction.trigger_id, reconstruction.camera_ids,
            MappingProxyType(markers), reconstruction.excluded_marker_ids,
        )
        result = recognize_boards(
            bad_scene, self.registry, max_rigid_rms_world_units=0.01
        )
        self.assertEqual(result.boards["board_A"].status, "recognized")
        self.assertEqual(result.boards["board_B"].status, "geometry_rejected")
        self.assertIsNone(result.boards["board_B"].pose)

    def test_unknown_id_and_duplicate_camera_local_track_fail_closed(self):
        admissions = self.admissions()
        a, b = admissions[0], admissions[1]
        duplicate_local = replace(a, local_track_id=b.local_track_id)
        with self.assertRaisesRegex(ValueError, "local light track reused"):
            self.recognize([b, duplicate_local])
        unknown_id = replace(a, detection=replace(a.detection, marker_id=999))
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.recognize([unknown_id])

    def test_preferred_trace_service_decodes_from_registered_codebook(self):
        tracks = self.admissions(as_traces=True)
        result = recognize_tracked_boards(
            tracks, self.frames, [2, 5, 9],
            self.cameras, self.K,
            {camera: self.calibration for camera in (2, 5, 9)},
            self.registry,
            trigger_epoch=self.epoch,
            max_exposure_skew_ns=60,
            controller_config_sha256=self.registry.config_sha256,
            max_reprojection_rms_px=1e-5,
            max_rigid_rms_world_units=1e-5,
        )
        self.assertEqual(result.config_sha256, self.registry.config_sha256)
        self.assertEqual(len(result.config_sha256), 64)
        self.assertEqual(result.boards["board_A"].status, "recognized")
        self.assertEqual(result.boards["board_B"].status, "recognized")

    def test_controller_registry_revision_mismatch_is_rejected_before_decode(self):
        tracks = self.admissions(as_traces=True)
        with self.assertRaisesRegex(ValueError, "configuration SHA-256 mismatch"):
            recognize_tracked_boards(
                tracks, self.frames, [2, 5, 9],
                self.cameras, self.K,
                {camera: self.calibration for camera in (2, 5, 9)},
                self.registry,
                trigger_epoch=self.epoch,
                max_exposure_skew_ns=60,
                controller_config_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            recognize_tracked_boards(
                [], self.frames, [2, 5, 9],
                self.cameras, self.K, {2: self.calibration},
                self.registry,
                trigger_epoch=self.epoch, max_exposure_skew_ns=60,
            )

    def test_trace_service_rejects_borrowed_frame_and_duplicate_track(self):
        tracks = self.admissions(as_traces=True)
        with self.assertRaisesRegex(ValueError, "duplicate local light trace"):
            recognize_tracked_boards(
                [tracks[0], tracks[0]], self.frames, [2, 5, 9],
                self.cameras, self.K,
                {camera: self.calibration for camera in (2, 5, 9)},
                self.registry,
                trigger_epoch=self.epoch, max_exposure_skew_ns=60,
            )
        with self.assertRaisesRegex(ValueError, "current frame batch"):
            recognize_tracked_boards(
                [replace(tracks[0], target_frame=self.frames[1])],
                self.frames, [2, 5, 9],
                self.cameras, self.K,
                {camera: self.calibration for camera in (2, 5, 9)},
                self.registry,
                trigger_epoch=self.epoch, max_exposure_skew_ns=60,
            )

    def test_wrong_phase_refuses_pose_and_next_cycle_has_no_stale_result(self):
        non_pose_frames = [
            FrameStamp(f.camera_id, self.trigger - 1,
                       f.exposure_timestamp_ns - 1000, f.clock_domain)
            for f in self.frames
        ]
        with self.assertRaisesRegex(ValueError, "shared pose_phase"):
            recognize_admitted_boards(
                [], non_pose_frames, [2, 5, 9], self.cameras, self.K,
                self.registry, trigger_epoch=self.epoch,
                max_exposure_skew_ns=60,
            )
        first = self.recognize()
        self.assertEqual(first.boards["board_A"].status, "recognized")
        second_phase = [
            FrameStamp(f.camera_id, self.trigger + 16,
                       f.exposure_timestamp_ns + 16_000, f.clock_domain)
            for f in self.frames
        ]
        result = recognize_admitted_boards(
            [], second_phase, [2, 5, 9], self.cameras, self.K,
            self.registry, trigger_epoch=self.epoch,
            max_exposure_skew_ns=60, previous_trigger_id=self.trigger,
            allow_trigger_gaps=True,
        )
        for outcome in result.boards.values():
            self.assertEqual(outcome.status, "not_visible")
            self.assertIsNone(outcome.pose)


if __name__ == "__main__":
    unittest.main()
