"""Independent calibrated bundle-adjustment regressions."""
import unittest
import numpy as np

from bundle_geometry import bundle_reprojection_residuals, refine_calibrated_bundle


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project(points, K, pose):
    camera = (pose["R"] @ points.T).T + pose["t"]
    homogeneous = (K @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


class CalibratedBundleTests(unittest.TestCase):
    def setUp(self):
        self.K = [
            np.array([[700., 0., 300.], [0., 750., 230.], [0., 0., 1.]]),
            np.array([[1000., 0., 330.], [0., 950., 225.], [0., 0., 1.]]),
            np.array([[1300., 0., 320.], [0., 1200., 240.], [0., 0., 1.]]),
        ]
        self.poses = [
            {"R": np.eye(3), "t": np.zeros(3)},
            {"R": rotation_y(0.07), "t": np.array([-0.8, 0., 0.])},
            {"R": rotation_y(-0.11), "t": np.array([0.25, -0.75, 0.03])},
        ]
        rng = np.random.default_rng(2501)
        self.world = np.column_stack((
            rng.uniform(-0.7, 0.7, 15),
            rng.uniform(-0.55, 0.55, 15),
            rng.uniform(3., 7., 15),
        ))
        self.observations = np.stack(
            [project(self.world, K, pose) for K, pose in zip(self.K, self.poses)],
            axis=1,
        )

    def test_signed_residuals_have_fixed_order_and_depend_on_camera_intrinsics(self):
        baseline = bundle_reprojection_residuals(
            self.observations, self.poses, self.K, self.world
        )
        self.assertEqual(len(baseline), 2 * 15 * 3)
        self.assertLess(np.max(np.abs(baseline)), 1e-9)
        changed = self.observations.copy()
        changed[0, 1, 0] += 0.35
        errors = bundle_reprojection_residuals(changed, self.poses, self.K, self.world)
        self.assertAlmostEqual(errors[2], -0.35, places=9)
        self.assertAlmostEqual(errors[3], 0.0, places=9)

        swapped = [self.K[0], self.K[2], self.K[1]]
        wrong_K_errors = bundle_reprojection_residuals(
            self.observations, self.poses, swapped, self.world
        )
        self.assertGreater(np.linalg.norm(wrong_K_errors), 100.)

    def test_sparse_track_identity_and_fixed_residual_dimension(self):
        observations = self.observations.astype(object)
        observations[::3, 1] = [None, None]
        observations[1::4, 0] = [None, None]
        initial_poses = [
            {"R": pose["R"].copy(), "t": pose["t"].copy()}
            for pose in self.poses
        ]
        initial_poses[2]["R"] = rotation_y(-0.107)
        initial_poses[2]["t"] += [0.01, -0.01, 0.005]
        optimized, metrics = refine_calibrated_bundle(
            observations, initial_poses, self.K,
            min_tracks=10, max_nfev=160,
        )
        self.assertEqual(metrics["admitted_track_indices"], list(range(15)))
        self.assertEqual(metrics["signed_residual_count"], 2 * int(np.sum(
            np.any(np.asarray(observations, dtype=object) != None, axis=2)
        )))
        self.assertTrue(metrics["camera_zero_pose_fixed"])
        self.assertTrue(metrics["camera_one_translation_fixed"])
        self.assertTrue(metrics["intrinsics_fixed"])
        self.assertFalse(metrics["translation_scale_known"])
        self.assertLess(metrics["reprojection_rms_px"], metrics["initial_reprojection_rms_px"])
        np.testing.assert_allclose(optimized[0]["R"], self.poses[0]["R"], atol=1e-12)
        np.testing.assert_allclose(optimized[1]["t"], self.poses[1]["t"], atol=1e-12)

    def test_joint_refinement_reduces_noisy_reprojection_error(self):
        rng = np.random.default_rng(991)
        noisy = self.observations + rng.normal(0, 0.08, self.observations.shape)
        initial = [
            {"R": pose["R"].copy(), "t": pose["t"].copy()}
            for pose in self.poses
        ]
        initial[1]["R"] = rotation_y(0.077)
        initial[2]["R"] = rotation_y(-0.104)
        initial[2]["t"] += [0.012, -0.009, 0.006]
        original = [K.copy() for K in self.K]
        optimized, metrics = refine_calibrated_bundle(
            noisy, initial, self.K, min_tracks=10, max_nfev=180,
        )
        self.assertLess(metrics["reprojection_rms_px"], 0.3)
        self.assertLess(metrics["reprojection_rms_px"], metrics["initial_reprojection_rms_px"])
        np.testing.assert_allclose(optimized[1]["t"], initial[1]["t"], atol=1e-12)
        np.testing.assert_allclose(optimized[0]["R"], initial[0]["R"], atol=1e-12)
        for K, unchanged in zip(self.K, original):
            np.testing.assert_array_equal(K, unchanged)

    def test_invalid_or_disconnected_tracks_fail_closed(self):
        malformed = self.observations.astype(object)
        malformed[0, 1] = [None, 100.0]
        with self.assertRaisesRegex(ValueError, "incomplete observation"):
            refine_calibrated_bundle(malformed, self.poses, self.K)
        invisible = self.observations.astype(object)
        invisible[:, 2] = [None, None]
        with self.assertRaisesRegex(ValueError, "each camera"):
            refine_calibrated_bundle(invisible, self.poses, self.K)

    def test_no_mutation_and_scalar_input_contract(self):
        initial = [
            {"R": pose["R"].copy(), "t": pose["t"].copy()}
            for pose in self.poses
        ]
        with self.assertRaisesRegex(ValueError, "robust_scale_px"):
            refine_calibrated_bundle(self.observations, initial, self.K, robust_scale_px=10**1000)
        bad = [
            {"R": pose["R"].copy(), "t": pose["t"].copy()}
            for pose in self.poses
        ]
        bad[1]["t"][:] = 0
        with self.assertRaisesRegex(ValueError, "nonzero translation gauge"):
            refine_calibrated_bundle(self.observations, bad, self.K)


if __name__ == "__main__":
    unittest.main()
