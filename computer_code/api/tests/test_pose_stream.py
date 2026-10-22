"""Software PoseFrame and BoardDescription separation; no flight IO."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import BoardOutcome, BoardPose
from identity_locked_board_tracker import (
    BoardFrameStatus, LockedBoardFrame,
)
from pose_stream import (
    describe_board_assets, pose_frame_to_dict, pose_frame_to_json,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "board-patterns.continuous.example.json"
)


class PoseStreamTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        theta = 0.4
        c, s = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0., 0., 1.]])
        pose = BoardPose(
            board_id="board_A",
            trigger_id=100,
            marker_ids=(100, 101, 102),
            rotation_world_from_board=self.R,
            translation_world_from_board=np.array([0.2, 0.1, 1.2]),
            rotation_world_from_body=self.R,
            translation_world_from_body=(
                np.array([0.2, 0.1, 1.2])
                - self.R @ self.registry.boards["board_A"].translation_body_from_board_m
            ),
            rigid_rms_error=0.001,
            reprojection_max_rms_px=0.05,
            metric_scale_validated=False,
        )
        measured = BoardOutcome(
            "board_A", "recognized", (100, 101, 102), (103,), pose,
        )
        not_visible = BoardOutcome(
            "board_B", "not_visible", (), (), None,
        )
        self.frame = LockedBoardFrame(
            trigger_id=100,
            camera_ids=(2, 5, 9),
            clock_domain="test-declared-hardware-clock",
            exposure_skew_ns=20,
            latest_exposure_timestamp_ns=2_000_000,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256="a" * 64,
            boards={
                "board_A": BoardFrameStatus(
                    "board_A", "TRACKING", "MEASURED", 100, 0,
                    2_000_000, 0, measured,
                ),
                "board_B": BoardFrameStatus(
                    "board_B", "DISCOVER", "INVALID",
                    None, None, None, None, not_visible,
                ),
            },
            active_locks=(),
            newly_confirmed_marker_ids=(),
            invalidated_marker_ids=(),
            measured_marker_ids=(100, 101, 102),
            hardware_provenance_verified=False,
        )

    def test_static_assets_have_codebook_and_dynamic_frame_does_not(self):
        desc = describe_board_assets(self.registry)
        self.assertEqual(desc["kind"], "asset_description")
        self.assertEqual(desc["schema_version"], 1)
        self.assertEqual(desc["pattern_sha256"], self.registry.config_sha256)
        self.assertEqual(len(desc["boards"]), 2)
        self.assertEqual([m["marker_id"] for m in desc["boards"][0]["markers"]], [100, 101, 102, 103])
        self.assertNotIn("rigid_bodies", desc)
        frame = pose_frame_to_dict(self.frame, self.registry)
        self.assertNotIn("boards", frame)
        self.assertEqual(frame["kind"], "pose_frame")
        self.assertNotIn("markers", frame)

    def test_fresh_measured_and_missing_board_pose_have_disjoint_values(self):
        data = pose_frame_to_dict(self.frame, self.registry)
        self.assertEqual(data["frame_id"], 100)
        self.assertEqual(data["clock_domain"], "test-declared-hardware-clock")
        self.assertEqual(data["exposure_skew_ns"], 20)
        self.assertEqual(data["position_units"], "uncalibrated_world")
        self.assertFalse(data["hardware_provenance_verified"])
        a, b = data["rigid_bodies"]
        self.assertEqual(a["board_id"], "board_A")
        self.assertEqual(a["evidence_kind"], "MEASURED")
        self.assertEqual(a["tracking_status"], "TRACKING")
        self.assertEqual(a["visible_marker_ids"], [100, 101, 102])
        self.assertEqual(a["excluded_marker_ids"], [103])
        np.testing.assert_allclose(
            a["position_world_from_body"],
            np.array([0.2, 0.1, 1.2])
            - self.R @ self.registry.boards["board_A"].translation_body_from_board_m,
            atol=1e-12,
        )
        self.assertAlmostEqual(sum(q * q for q in a["quaternion_world_from_body_xyzw"]), 1.)
        self.assertAlmostEqual(a["quaternion_world_from_body_xyzw"][2], np.sin(0.2))
        self.assertGreater(a["quaternion_world_from_body_xyzw"][3], 0.)
        self.assertEqual(a["quality"]["reprojection_max_rms_px"], 0.05)
        self.assertFalse(a["metric_scale_validated"])
        self.assertEqual(b["board_id"], "board_B")
        self.assertEqual(b["evidence_kind"], "INVALID")
        self.assertIsNone(b["position_world_from_board"])
        self.assertIsNone(b["quaternion_world_from_body_xyzw"])
        self.assertIsNone(b["quality"])
        encoded = pose_frame_to_json(self.frame, self.registry)
        self.assertEqual(json.loads(encoded), data)
        self.assertEqual(encoded, pose_frame_to_json(self.frame, self.registry))

    def test_degraded_board_exposes_measurement_age_not_stale_position(self):
        statuses = dict(self.frame.boards)
        statuses["board_A"] = BoardFrameStatus(
            "board_A", "DEGRADED", "INVALID",
            measurement_trigger_id=99,
            measurement_age_triggers=1,
            last_measurement_exposure_timestamp_ns=1_999_200,
            measurement_age_ns=800,
            outcome=BoardOutcome("board_A", "insufficient_markers", (100, 101), (102,), None),
        )
        frame = replace(self.frame, boards=statuses)
        data = pose_frame_to_dict(frame, self.registry)["rigid_bodies"][0]
        self.assertEqual(data["measurement_trigger_id"], 99)
        self.assertEqual(data["measurement_age_triggers"], 1)
        self.assertEqual(data["measurement_age_ns"], 800)
        self.assertIsNone(data["position_world_from_body"])
        self.assertIsNone(data["quality"])

    def test_inconsistent_metadata_never_publishes_pose(self):
        mismatched = [
            replace(self.frame, pattern_sha256="0" * 64),
            replace(self.frame, hardware_provenance_verified=True),
            replace(self.frame, calibration_sha256="bad"),
            replace(self.frame, exposure_skew_ns=-1),
            replace(self.frame, boards={"board_A": self.frame.boards["board_A"]}),
        ]
        for case in mismatched:
            with self.subTest(case=repr(case)[:110]), self.assertRaises(ValueError):
                pose_frame_to_dict(case, self.registry)

    def test_stale_measured_or_fabricated_geometry_must_fail_closed(self):
        status = self.frame.boards["board_A"]
        malformed = [
            replace(status, measurement_age_triggers=1),
            replace(status, measurement_trigger_id=99),
            replace(status, outcome=replace(status.outcome, pose=replace(
                status.outcome.pose, trigger_id=99
            ))),
            replace(status, outcome=replace(status.outcome, pose=replace(
                status.outcome.pose, translation_world_from_body=np.array([float("nan"), 0, 0])
            ))),
            replace(status, outcome=replace(status.outcome, pose=replace(
                status.outcome.pose, rotation_world_from_body=np.diag([-1., 1., 1.])
            ))),
            replace(status, outcome=replace(status.outcome, pose=replace(
                status.outcome.pose, translation_world_from_body=np.array([999., 0., 0.])
            ))),
            replace(status, outcome=replace(status.outcome, pose=replace(
                status.outcome.pose, rigid_rms_error=10**1000
            ))),
        ]
        for value in malformed:
            with self.subTest(value=repr(value)[:140]):
                with self.assertRaises(ValueError):
                    pose_frame_to_dict(
                        replace(self.frame, boards={"board_A": value, "board_B": self.frame.boards["board_B"]}),
                        self.registry,
                    )

    def test_missing_board_status_cannot_contain_a_pose(self):
        fake = replace(self.frame.boards["board_A"], evidence_kind="INVALID")
        with self.assertRaises(ValueError):
            pose_frame_to_dict(
                replace(self.frame, boards={"board_A": fake, "board_B": self.frame.boards["board_B"]}),
                self.registry,
            )


if __name__ == "__main__":
    unittest.main()
