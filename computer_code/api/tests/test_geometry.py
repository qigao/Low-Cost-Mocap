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

    def test_intrinsic_matrix_global_scale_does_not_change_triangulation(self):
        observations = [self.project(i) for i in range(3)]
        scaled_intrinsics = [K * 1e-6 for K in self.Ks]
        point, metrics = triangulate_point(observations, self.poses, scaled_intrinsics)
        np.testing.assert_allclose(point, self.truth, atol=1e-9)
        self.assertLess(metrics["reprojection_rms_px"], 1e-8)

    def test_reports_parallax_for_well_conditioned_views(self):
        observations = [self.project(i) for i in range(3)]
        _, metrics = triangulate_point(observations, self.poses, self.Ks)
        self.assertGreater(metrics["max_parallax_angle_deg"], 0.1)

    def test_noisy_observations_report_nonzero_residual_and_recover_depth(self):
        observations = [self.project(i) for i in range(3)]
        observations[0] = observations[0] + np.array([0.15, -0.10])
        observations[1] = observations[1] + np.array([-0.10, 0.12])
        observations[2] = observations[2] + np.array([0.08, 0.05])
        point, metrics = triangulate_point(
            observations, self.poses, self.Ks,
            max_reprojection_rms_px=0.5,
        )
        self.assertGreater(point[2], 0.0)
        self.assertLess(np.linalg.norm(point - self.truth), 0.02)
        self.assertGreater(metrics["reprojection_rms_px"], 0.0)
        self.assertLess(metrics["reprojection_rms_px"], 0.5)

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


    def test_parallax_decreases_as_target_depth_increases(self):
        near = np.array([0.2, -0.1, 5.0])
        far = np.array([0.2, -0.1, 10.0])

        near_observations = [self.project(i, near) for i in range(3)]
        far_observations = [self.project(i, far) for i in range(3)]

        near_point, near_metrics = triangulate_point(
            near_observations, self.poses, self.Ks,
        )
        far_point, far_metrics = triangulate_point(
            far_observations, self.poses, self.Ks,
        )

        np.testing.assert_allclose(near_point, near, atol=1e-9)
        np.testing.assert_allclose(far_point, far, atol=1e-8)
        self.assertGreater(
            near_metrics["max_parallax_angle_deg"],
            far_metrics["max_parallax_angle_deg"],
        )
        self.assertGreater(far_metrics["max_parallax_angle_deg"], 0.1)

        # The same subpixel measurement perturbation should hurt depth more
        # at longer range, where the viewing rays have weaker parallax.
        near_noisy = [self.project(i, near) for i in range(3)]
        far_noisy = [self.project(i, far) for i in range(3)]
        near_noisy[2] = near_noisy[2] + np.array([0.1, -0.1])
        far_noisy[2] = far_noisy[2] + np.array([0.1, -0.1])
        near_estimate, near_noisy_metrics = triangulate_point(
            near_noisy, self.poses, self.Ks,
        )
        far_estimate, far_noisy_metrics = triangulate_point(
            far_noisy, self.poses, self.Ks,
        )
        near_depth_error = abs(near_estimate[2] - near[2])
        far_depth_error = abs(far_estimate[2] - far[2])
        self.assertGreater(far_depth_error, near_depth_error)
        self.assertGreater(near_noisy_metrics["reprojection_rms_px"], 0.0)
        self.assertGreater(far_noisy_metrics["reprojection_rms_px"], 0.0)


    def test_depth_error_grows_across_depth_and_pixel_noise_matrix(self):
        # Keep the rig and perturbation direction fixed so the test isolates
        # the conditioning effect of range and measurement noise.
        depths = (4.0, 5.0, 6.0, 8.0, 10.0)
        noise_levels = (0.05, 0.1, 0.2, 0.4)

        errors_by_noise = {}
        for noise_px in noise_levels:
            depth_errors = []
            for depth in depths:
                truth = np.array([0.2, -0.1, depth])
                observations = [self.project(i, truth) for i in range(3)]
                observations[2] = observations[2] + np.array([noise_px, -noise_px])
                estimate, _ = triangulate_point(observations, self.poses, self.Ks)
                depth_errors.append(abs(float(estimate[2]) - depth))

            # For this fixed rig and perturbation direction, range should
            # monotonically amplify the same pixel error into depth error.
            self.assertTrue(
                all(later > earlier for earlier, later in zip(depth_errors, depth_errors[1:])),
                msg=f"depth errors were not increasing for {noise_px}px noise: {depth_errors}",
            )
            errors_by_noise[noise_px] = depth_errors

        for lower_noise, higher_noise in zip(noise_levels, noise_levels[1:]):
            for depth_index, depth in enumerate(depths):
                self.assertGreater(
                    errors_by_noise[higher_noise][depth_index],
                    errors_by_noise[lower_noise][depth_index],
                    msg=(
                        f"depth error did not grow with pixel noise at z={depth}: "
                        f"{errors_by_noise[lower_noise][depth_index]} vs "
                        f"{errors_by_noise[higher_noise][depth_index]}"
                    ),
                )

    def test_triangulation_with_rotated_cameras_and_varied_focal_lengths(self):
        truth = np.array([0.35, -0.22, 6.0])
        centers = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.1, 0.05, 0.0]),
            np.array([-0.25, 0.9, 0.0]),
            np.array([0.65, -0.55, 0.0]),
        )
        # Small world-to-camera yaw rotations; translation is derived from
        # the camera center so the pose convention stays explicit.
        yaw_angles = (0.0, 0.025, -0.035, 0.018)

        for focal in (500.0, 800.0, 1200.0):
            with self.subTest(focal_length=focal):
                K = np.array([
                    [focal, 0.0, 320.0],
                    [0.0, focal, 240.0],
                    [0.0, 0.0, 1.0],
                ])
                poses = []
                observations = []
                for center, yaw in zip(centers, yaw_angles):
                    c, s = np.cos(yaw), np.sin(yaw)
                    R = np.array([
                        [c, 0.0, s],
                        [0.0, 1.0, 0.0],
                        [-s, 0.0, c],
                    ])
                    t = -(R @ center)
                    pose = {"R": R, "t": t}
                    camera_point = R @ truth + t
                    projected = K @ camera_point
                    observations.append(projected[:2] / projected[2])
                    poses.append(pose)

                estimate, metrics = triangulate_point(
                    observations, poses, [K.copy() for _ in poses],
                    max_reprojection_rms_px=1e-6,
                )
                np.testing.assert_allclose(estimate, truth, atol=1e-8)
                self.assertLess(metrics["reprojection_rms_px"], 1e-7)
                self.assertGreater(metrics["max_parallax_angle_deg"], 0.1)
                self.assertEqual(metrics["camera_ids"], [0, 1, 2, 3])


    def test_rotated_camera_geometry_remains_stable_under_subpixel_noise(self):
        truth = np.array([0.35, -0.22, 6.0])
        centers = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.1, 0.05, 0.0]),
            np.array([-0.25, 0.9, 0.0]),
            np.array([0.65, -0.55, 0.0]),
        )
        yaw_angles = (0.0, 0.025, -0.035, 0.018)
        noise = (
            np.array([0.12, -0.08]),
            np.array([-0.09, 0.11]),
            np.array([0.06, 0.04]),
            np.array([-0.05, -0.07]),
        )
        K = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        poses = []
        observations = []
        for center, yaw, perturbation in zip(centers, yaw_angles, noise):
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ])
            t = -(R @ center)
            pose = {"R": R, "t": t}
            camera_point = R @ truth + t
            pixel = K @ camera_point
            observations.append(pixel[:2] / pixel[2] + perturbation)
            poses.append(pose)

        estimate, metrics = triangulate_point(
            observations, poses, [K.copy() for _ in poses],
            max_reprojection_rms_px=0.5,
        )
        self.assertLess(np.linalg.norm(estimate - truth), 0.02)
        self.assertGreater(metrics["reprojection_rms_px"], 0.0)
        self.assertLess(metrics["reprojection_rms_px"], 0.5)
        self.assertTrue(all(depth > 0.0 for depth in metrics["depths"].values()))
        self.assertEqual(metrics["camera_ids"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
