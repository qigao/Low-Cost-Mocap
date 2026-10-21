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

    def test_rejects_non_real_numeric_observation_coordinates(self):
        valid = [self.project(i) for i in range(4)]
        invalid_observations = (
            ["320", "240"],
            [True, False],
            [320, True],
            [np.float64(320.0), np.bool_(False)],
            [320.0 + 1.0j, 240.0 + 0.0j],
        )
        for observation in invalid_observations:
            with self.subTest(observation=repr(observation)):
                observations = list(valid)
                observations[1] = observation
                with self.assertRaisesRegex(ValueError, "real numeric coordinates"):
                    triangulate_consensus(observations, self.poses, self.Ks)

    def test_consensus_is_invariant_to_global_intrinsic_scale(self):
        observations = [self.project(i) for i in range(4)]
        scaled_intrinsics = [K * 1e-14 for K in self.Ks]
        xyz, metrics = triangulate_consensus(
            observations, self.poses, scaled_intrinsics,
            min_views=3, inlier_threshold_px=0.01,
            max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2, 3])
        self.assertLess(metrics["reprojection_rms_px"], 1e-8)

    def test_consensus_ignores_signed_intrinsic_scale(self):
        observations = [self.project(i) for i in range(4)]
        signed_intrinsics = [
            self.K.copy(), -self.K.copy(), 0.5 * self.K, -2.0 * self.K
        ]
        xyz, metrics = triangulate_consensus(
            observations, self.poses, signed_intrinsics,
            min_views=3, inlier_threshold_px=0.01,
            max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2, 3])
        self.assertLess(metrics["reprojection_rms_px"], 1e-8)

    def test_consensus_invariant_to_extreme_projective_intrinsic_scale(self):
        observations = [self.project(i) for i in range(4)]
        scales = (1e-280, -1e300, 1e250, -1e-220)
        intrinsics = [scale * K for scale, K in zip(scales, self.Ks)]
        xyz, metrics = triangulate_consensus(
            observations, self.poses, intrinsics,
            min_views=3, inlier_threshold_px=0.01,
            max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2, 3])
        self.assertEqual(metrics["outlier_camera_ids"], [])
        self.assertLess(metrics["reprojection_rms_px"], 1e-7)

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
        self.assertEqual(metrics["fit_metrics"]["camera_ids"], metrics["inlier_camera_ids"])
        self.assertGreater(metrics["reprojection_errors_px"][3], 40.0)
        self.assertGreater(metrics["reprojection_rms_px"], 0.0)
        self.assertLess(metrics["reprojection_rms_px"], 0.5)

    def test_multiple_outliers_and_missing_view_preserve_camera_ids(self):
        poses = self.poses + [
            {"R": np.eye(3), "t": np.array([-0.35, 0.65, 0.0])},
            {"R": np.eye(3), "t": np.array([0.95, -0.55, 0.0])},
        ]
        intrinsics = [self.K.copy() for _ in poses]
        observations = []
        for pose, intrinsic in zip(poses, intrinsics):
            camera_point = pose["R"] @ self.truth + pose["t"]
            pixel = intrinsic @ camera_point
            observations.append(pixel[:2] / pixel[2])

        observations[3] = observations[3] + np.array([62.0, -41.0])
        observations[4] = observations[4] + np.array([-48.0, 57.0])
        observations[5] = None

        xyz, metrics = triangulate_consensus(
            observations, poses, intrinsics,
            min_views=3, inlier_threshold_px=0.5,
            max_reprojection_rms_px=1e-5,
        )
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2])
        self.assertEqual(metrics["outlier_camera_ids"], [3, 4])
        self.assertEqual(metrics["fit_metrics"]["camera_ids"], [0, 1, 2])
        self.assertNotIn(5, metrics["reprojection_errors_px"])
        self.assertGreater(metrics["reprojection_errors_px"][3], 40.0)
        self.assertGreater(metrics["reprojection_errors_px"][4], 40.0)

    def test_rejects_two_disjoint_equal_support_3d_hypotheses(self):
        poses = self.poses + [
            {"R": np.eye(3), "t": np.array([-0.35, 0.65, 0.0])},
            {"R": np.eye(3), "t": np.array([0.95, -0.55, 0.0])},
        ]
        intrinsics = [self.K.copy() for _ in poses]
        point_a = np.array([-0.25, 0.1, 3.2])
        point_b = np.array([0.55, -0.3, 4.6])
        observations = []
        for camera_id, (pose, intrinsic) in enumerate(zip(poses, intrinsics)):
            point = point_a if camera_id < 3 else point_b
            camera_point = pose["R"] @ point + pose["t"]
            pixel = intrinsic @ camera_point
            observations.append(pixel[:2] / pixel[2])

        with self.assertRaisesRegex(ValueError, "ambiguous multiview consensus"):
            triangulate_consensus(
                observations, poses, intrinsics,
                min_views=3, inlier_threshold_px=0.1,
                max_reprojection_rms_px=1e-5,
            )

    def test_rejects_competing_hypotheses_with_partial_overlap(self):
        poses = self.poses + [
            {"R": np.eye(3), "t": np.array([-0.35, 0.65, 0.0])},
            {"R": np.eye(3), "t": np.array([0.95, -0.55, 0.0])},
        ]
        intrinsics = [self.K.copy() for _ in poses]
        point_a = np.array([-0.25, 0.1, 3.2])
        # Put point_b on the same camera-2 ray as point_a. Camera 2 therefore
        # supports both spatially distinct hypotheses, while cameras 0/1 and
        # 3/4 support different hypotheses.
        camera_center_2 = np.array([0.0, 0.7, 0.0])
        point_b = camera_center_2 + 1.5 * (point_a - camera_center_2)
        observations = []
        for camera_id, (pose, intrinsic) in enumerate(zip(poses, intrinsics)):
            point = point_a if camera_id in (0, 1, 2) else point_b
            camera_point = pose["R"] @ point + pose["t"]
            pixel = intrinsic @ camera_point
            observations.append(pixel[:2] / pixel[2])
        observations[5] = None

        with self.assertRaisesRegex(ValueError, "ambiguous multiview consensus"):
            triangulate_consensus(
                observations, poses, intrinsics,
                min_views=3, inlier_threshold_px=0.1,
                max_reprojection_rms_px=1e-5,
            )

    def test_full_support_candidates_are_refit_to_the_same_support(self):
        # With a permissive threshold, every seed classifies all cameras as
        # inliers. Refitting each candidate to that identical support must
        # converge to the same solution rather than inventing ambiguity from
        # differences between the original subset seeds.
        observations = [self.project(i) for i in range(4)]
        observations[3] = observations[3] + np.array([12.0, -8.0])

        xyz, metrics = triangulate_consensus(
            observations, self.poses, self.Ks,
            min_views=3, inlier_threshold_px=100.0,
            max_reprojection_rms_px=100.0,
            ambiguity_rms_tolerance_px=100.0,
        )
        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2, 3])
        self.assertEqual(metrics["outlier_camera_ids"], [])
        self.assertTrue(np.isfinite(xyz).all())
        self.assertLessEqual(metrics["reprojection_rms_px"], 100.0)

    def test_rejects_boolean_and_non_integer_min_views(self):
        for min_views in (True, 3.0, 3.5, np.bool_(False)):
            with self.subTest(min_views=repr(min_views)):
                with self.assertRaisesRegex(ValueError, "min_views must be an integer"):
                    triangulate_consensus(
                        [self.project(i) for i in range(4)], self.poses, self.Ks,
                        min_views=min_views,
                    )

    def test_rejects_boolean_or_non_scalar_numeric_thresholds(self):
        cases = (
            ("inlier_threshold_px", True),
            ("max_reprojection_rms_px", np.bool_(False)),
            ("min_depth", "0.1"),
            ("min_parallax_angle_deg", np.array([1.0])),
            ("ambiguity_rms_tolerance_px", 1.0 + 0j),
        )
        for name, value in cases:
            with self.subTest(name=name, value=repr(value)):
                with self.assertRaisesRegex(ValueError, name):
                    triangulate_consensus(
                        [self.project(i) for i in range(4)], self.poses, self.Ks,
                        **{name: value},
                    )

    def test_rejects_scalar_thresholds_larger_than_float64(self):
        observations = [self.project(i) for i in range(4)]
        for name in (
            "inlier_threshold_px",
            "max_reprojection_rms_px",
            "min_depth",
            "min_parallax_angle_deg",
            "ambiguity_rms_tolerance_px",
        ):
            with self.subTest(parameter=name):
                with self.assertRaisesRegex(ValueError, name):
                    triangulate_consensus(
                        observations, self.poses, self.Ks, **{name: 10**1000}
                    )

    def test_ambiguity_rms_tolerance_must_be_finite_and_nonnegative(self):
        for tolerance in (-0.001, np.nan, np.inf, -np.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "ambiguity_rms_tolerance_px"):
                    triangulate_consensus(
                        [self.project(i) for i in range(4)], self.poses, self.Ks,
                        ambiguity_rms_tolerance_px=tolerance,
                    )

    def test_rejects_parallax_threshold_above_acute_angle_range(self):
        observations = [self.project(i) for i in range(4)]
        for threshold in (90.000001, 120.0, 180.0):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "between 0 and 90"):
                    triangulate_consensus(
                        observations, self.poses, self.Ks,
                        min_parallax_angle_deg=threshold,
                    )

    def test_configurable_parallax_gate_rejects_weak_geometry(self):
        observations = [self.project(i) for i in range(4)]
        with self.assertRaisesRegex(ValueError, "parallax"):
            triangulate_consensus(
                observations, self.poses, self.Ks,
                min_views=3, min_parallax_angle_deg=45.0,
            )


if __name__ == "__main__":
    unittest.main()
