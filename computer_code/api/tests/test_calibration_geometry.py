"""Independent multi-camera calibration geometry tests (no camera hardware)."""
import unittest

import numpy as np

from calibration_geometry import (
    compose_world_to_camera_pose,
    select_relative_pose_by_cheirality,
)


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project(points, intrinsic, pose):
    camera = (pose["R"] @ points.T).T + pose["t"]
    pixels = (intrinsic @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


class RelativeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.K1 = np.array([[750., 0., 320.], [0., 770., 240.], [0., 0., 1.]])
        self.K2 = np.array([[1200., 0., 300.], [0., 960., 220.], [0., 0., 1.]])
        self.first = {"R": np.eye(3), "t": np.zeros(3)}
        self.relative = {"R": rotation_y(0.12), "t": np.array([-1.0, 0.04, 0.02])}
        rng = np.random.default_rng(20261021)
        self.points = np.column_stack((
            rng.uniform(-0.6, 0.6, 20),
            rng.uniform(-0.5, 0.5, 20),
            rng.uniform(3.0, 7.0, 20),
        ))
        self.left = project(self.points, self.K1, self.first)
        self.right = project(self.points, self.K2, self.relative)

    def test_world_to_camera_composition_uses_rotated_previous_translation(self):
        first = {"R": rotation_y(0.31), "t": np.array([0.2, -0.3, 0.6])}
        relative = {"R": rotation_y(-0.19), "t": np.array([-2.0, 0.3, 0.1])}
        baseline = 1.7
        second = compose_world_to_camera_pose(first, relative, baseline_length=baseline)
        for world in self.points:
            camera_first = first["R"] @ world + first["t"]
            expected_second = (
                relative["R"] @ camera_first
                + baseline * relative["t"] / np.linalg.norm(relative["t"])
            )
            np.testing.assert_allclose(
                second["R"] @ world + second["t"], expected_second, atol=1e-12
            )

    def test_relative_translation_scale_requires_explicit_baseline_gauge(self):
        first = {"R": rotation_y(0.15), "t": [0.2, 0.4, 0.6]}
        a = compose_world_to_camera_pose(first, self.relative, baseline_length=0.9)
        scaled_relative = {**self.relative, "t": 17.0 * self.relative["t"]}
        b = compose_world_to_camera_pose(first, scaled_relative, baseline_length=0.9)
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-12)
        np.testing.assert_allclose(a["t"], b["t"], atol=1e-12)
        with self.assertRaises(TypeError):
            compose_world_to_camera_pose(first, self.relative)

    def test_essential_cheirality_selects_correct_orientation_and_translation_sign(self):
        R = self.relative["R"]
        t = self.relative["t"]
        wrong_rotation = rotation_y(np.pi) @ R
        rotations = [wrong_rotation, R, wrong_rotation, R]
        translations = [t, -t, -t, t]
        candidate, metrics = select_relative_pose_by_cheirality(
            self.left, self.right, self.K1, self.K2,
            rotations, translations,
            min_support=12, max_reprojection_rms_px=0.01,
        )
        self.assertEqual(metrics["selected_candidate_index"], 3)
        self.assertEqual(metrics["support_count"], len(self.points))
        self.assertEqual(metrics["inlier_pair_indices"], list(range(len(self.points))))
        self.assertFalse(metrics["translation_scale_known"])
        np.testing.assert_allclose(candidate["R"], R, atol=1e-10)
        np.testing.assert_allclose(
            candidate["t"], t / np.linalg.norm(t), atol=1e-10
        )

    def test_cheirality_ignores_outlier_matches_instead_of_changing_global_frames(self):
        right = self.right.copy()
        right[0] += [110., -50.]
        right[1] += [-75., 90.]
        candidate, metrics = select_relative_pose_by_cheirality(
            self.left, right, self.K1, self.K2,
            [self.relative["R"], self.relative["R"]],
            [self.relative["t"], -self.relative["t"]],
            min_support=16, max_reprojection_rms_px=0.1,
        )
        self.assertEqual(metrics["selected_candidate_index"], 0)
        self.assertEqual(metrics["support_count"], 18)
        self.assertNotIn(0, metrics["inlier_pair_indices"])
        self.assertNotIn(1, metrics["inlier_pair_indices"])
        np.testing.assert_allclose(candidate["R"], self.relative["R"])

    def test_cheirality_rejects_ambiguous_or_insufficient_support(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            select_relative_pose_by_cheirality(
                self.left, self.right, self.K1, self.K2,
                [self.relative["R"], self.relative["R"]],
                [self.relative["t"], self.relative["t"]],
                min_support=8,
            )
        with self.assertRaisesRegex(ValueError, "sufficient positive-depth support"):
            select_relative_pose_by_cheirality(
                self.left, self.right, self.K1, self.K2,
                [self.relative["R"]], [-self.relative["t"]],
                min_support=8,
            )

    def test_calibration_input_gates(self):
        for baseline in (True, 0.0, -1.0, "1", 10**1000, np.nan):
            with self.subTest(baseline=repr(baseline)):
                with self.assertRaisesRegex(ValueError, "baseline_length"):
                    compose_world_to_camera_pose(
                        self.first, self.relative, baseline_length=baseline
                    )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            compose_world_to_camera_pose(
                self.first, {"R": np.eye(3), "t": np.zeros(3)},
                baseline_length=1.0,
            )
        with self.assertRaisesRegex(ValueError, "paired observations"):
            select_relative_pose_by_cheirality(
                self.left, self.right[:-1], self.K1, self.K2,
                [self.relative["R"]], [self.relative["t"]],
            )
        with self.assertRaisesRegex(ValueError, "candidate lists"):
            select_relative_pose_by_cheirality(
                self.left, self.right, self.K1, self.K2,
                [self.relative["R"]], [],
            )


if __name__ == "__main__":
    unittest.main()
