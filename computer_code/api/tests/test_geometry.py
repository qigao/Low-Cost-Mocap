import unittest

import numpy as np

from geometry import triangulate_point


class TriangulationTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        # World-to-camera poses: camera centers at x = 0, 1, and 2.
        self.poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-1.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-2.0, 0.0, 0.0])},
        ]
        self.Ks = [self.K.copy() for _ in self.poses]
        self.truth = np.array([0.2, -0.1, 5.0])

    def project(self, camera_id, point=None):
        pose = self.poses[camera_id]
        xyz = pose["R"] @ (self.truth if point is None else point) + pose["t"]
        q = self.Ks[camera_id] @ xyz
        return q[:2] / q[2]

    def test_recovers_point_from_three_views(self):
        observations = [self.project(i) for i in range(3)]
        point, metrics = triangulate_point(observations, self.poses, self.Ks)
        np.testing.assert_allclose(point, self.truth, atol=1e-9)
        self.assertLess(metrics["reprojection_rms_px"], 1e-8)
        self.assertEqual(metrics["camera_ids"], [0, 1, 2])

    def test_reports_parallax_for_well_conditioned_views(self):
        observations = [self.project(i) for i in range(3)]
        _, metrics = triangulate_point(observations, self.poses, self.Ks)
        self.assertGreater(metrics["max_parallax_angle_deg"], 0.1)

    def test_rejects_nearly_parallel_viewing_rays(self):
        poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-1e-5, 0.0, 0.0])},
        ]
        point = np.array([0.2, -0.1, 5.0])
        observations = []
        for pose in poses:
            camera = pose["R"] @ point + pose["t"]
            pixel = self.K @ camera
            observations.append(pixel[:2] / pixel[2])
        with self.assertRaisesRegex(ValueError, "parallax"):
            triangulate_point(observations, poses, [self.K, self.K])

    def test_missing_middle_camera_keeps_camera_indices(self):
        observations = [self.project(0), None, self.project(2)]
        point, metrics = triangulate_point(observations, self.poses, self.Ks)
        np.testing.assert_allclose(point, self.truth, atol=1e-9)
        self.assertEqual(metrics["camera_ids"], [0, 2])
        self.assertEqual(set(metrics["depths"]), {0, 2})

    def test_rejects_insufficient_views(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            triangulate_point([self.project(0), None, None], self.poses, self.Ks)

    def test_rejects_reprojection_outlier(self):
        observations = [self.project(i) for i in range(3)]
        observations[2] = observations[2] + np.array([80.0, -60.0])
        with self.assertRaisesRegex(ValueError, "reprojection RMS"):
            triangulate_point(
                observations, self.poses, self.Ks, max_reprojection_rms_px=1.0
            )

    def test_rejects_points_behind_cameras(self):
        behind = np.array([0.2, -0.1, -5.0])
        observations = [self.project(i, behind) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "behind camera"):
            triangulate_point(observations, self.poses, self.Ks)

    def test_rejects_non_finite_observations(self):
        observations = [self.project(0), [float("nan"), 10.0], self.project(2)]
        with self.assertRaisesRegex(ValueError, "finite"):
            triangulate_point(observations, self.poses, self.Ks)


if __name__ == "__main__":
    unittest.main()
