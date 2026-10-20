import unittest

import numpy as np

from multiview_consensus import triangulate_consensus


class MultiviewConsensusTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        # Camera centers spread in x/y to avoid a single horizontal baseline.
        self.poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-0.8, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([0.0, -0.7, 0.0])},
            {"R": np.eye(3), "t": np.array([0.7, 0.4, 0.0])},
        ]
        self.Ks = [self.K.copy() for _ in self.poses]
        self.truth = np.array([0.12, -0.08, 3.5])

    def project(self, camera_id, point=None):
        pose = self.poses[camera_id]
        xyz = pose["R"] @ (self.truth if point is None else point) + pose["t"]
        pixel = self.Ks[camera_id] @ xyz
        return pixel[:2] / pixel[2]

    def test_three_view_consensus_rejects_extra_bright_spot(self):
        observations = [self.project(i) for i in range(4)]
        observations[3] = observations[3] + np.array([75.0, -45.0])
        xyz, metrics = triangulate_consensus(
            observations, self.poses, self.Ks,
            min_views=3, inlier_threshold_px=0.5, max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2])
        self.assertEqual(metrics["outlier_camera_ids"], [3])
        self.assertGreater(metrics["reprojection_errors_px"][3], 50.0)

    def test_missing_camera_is_not_compacted_or_misindexed(self):
        observations = [self.project(0), None, self.project(2), self.project(3)]
        xyz, metrics = triangulate_consensus(
            observations, self.poses, self.Ks,
            min_views=3, inlier_threshold_px=0.01, max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 2, 3])
        self.assertEqual(metrics["outlier_camera_ids"], [])

    def test_rejects_when_only_two_cameras_are_available(self):
        with self.assertRaisesRegex(ValueError, "at least 3 observations"):
            triangulate_consensus(
                [self.project(0), None, self.project(2), None],
                self.poses, self.Ks,
            )

    def test_rejects_inconsistent_views_without_consensus(self):
        observations = [self.project(i) for i in range(4)]
        observations[0] = observations[0] + np.array([90.0, 50.0])
        observations[1] = observations[1] + np.array([-70.0, 65.0])
        with self.assertRaisesRegex(ValueError, "no multiview hypothesis"):
            triangulate_consensus(
                observations, self.poses, self.Ks,
                min_views=3, inlier_threshold_px=0.1,
            )


    def test_noisy_inliers_survive_single_view_outlier_rejection(self):
        observations = [self.project(i) for i in range(4)]
        observations[0] = observations[0] + np.array([0.12, -0.08])
        observations[1] = observations[1] + np.array([-0.09, 0.11])
        observations[2] = observations[2] + np.array([0.06, 0.04])
        observations[3] = observations[3] + np.array([55.0, -38.0])

        xyz, metrics = triangulate_consensus(
            observations, self.poses, self.Ks,
            min_views=3, inlier_threshold_px=0.5,
            max_reprojection_rms_px=0.5,
        )
        self.assertLess(np.linalg.norm(xyz - self.truth), 0.02)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2])
        self.assertEqual(metrics["outlier_camera_ids"], [3])
        self.assertGreater(metrics["reprojection_errors_px"][3], 40.0)
        self.assertGreater(metrics["reprojection_rms_px"], 0.0)
        self.assertLess(metrics["reprojection_rms_px"], 0.5)

    def test_configurable_parallax_gate_rejects_weak_geometry(self):
        observations = [self.project(i) for i in range(4)]
        with self.assertRaisesRegex(ValueError, "parallax"):
            triangulate_consensus(
                observations, self.poses, self.Ks,
                min_views=3, min_parallax_angle_deg=45.0,
            )


if __name__ == "__main__":
    unittest.main()
