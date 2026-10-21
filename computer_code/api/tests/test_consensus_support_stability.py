import unittest

import numpy as np

from multiview_consensus import triangulate_consensus


class ConsensusSupportStabilityTests(unittest.TestCase):
    def test_reported_support_matches_reprojection_of_returned_point(self):
        intrinsic = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-0.8, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([0.0, -0.7, 0.0])},
            {"R": np.eye(3), "t": np.array([0.7, 0.4, 0.0])},
        ]
        intrinsics = [intrinsic.copy() for _ in poses]
        truth = np.array([0.12, -0.08, 3.5])
        observations = []
        for camera_id, pose in enumerate(poses):
            camera_point = pose["R"] @ truth + pose["t"]
            pixel = intrinsic @ camera_point
            observations.append(pixel[:2] / pixel[2])
        observations[0] = observations[0] + np.array([0.12, -0.08])
        observations[1] = observations[1] + np.array([-0.09, 0.11])
        observations[2] = observations[2] + np.array([0.06, 0.04])
        observations[3] = observations[3] + np.array([55.0, -38.0])

        threshold = 0.5
        xyz, metrics = triangulate_consensus(
            observations, poses, intrinsics,
            min_views=3,
            inlier_threshold_px=threshold,
            max_reprojection_rms_px=0.5,
        )

        independently_classified = []
        inlier_errors = []
        for camera_id, observed in enumerate(observations):
            camera_point = poses[camera_id]["R"] @ xyz + poses[camera_id]["t"]
            projected_h = intrinsics[camera_id] @ camera_point
            self.assertGreater(camera_point[2], 0.0)
            projected = projected_h[:2] / projected_h[2]
            error = float(np.linalg.norm(projected - observed))
            self.assertAlmostEqual(
                metrics["reprojection_errors_px"][camera_id], error, places=9
            )
            if error <= threshold:
                independently_classified.append(camera_id)
                inlier_errors.append(error)

        self.assertEqual(metrics["inlier_camera_ids"], independently_classified)
        self.assertEqual(metrics["fit_metrics"]["camera_ids"], independently_classified)
        self.assertEqual(metrics["outlier_camera_ids"], [
            i for i in range(len(observations)) if i not in independently_classified
        ])
        expected_rms = float(np.sqrt(np.mean(np.square(inlier_errors))))
        self.assertAlmostEqual(metrics["reprojection_rms_px"], expected_rms, places=9)

    def test_non_finite_final_projection_is_reported_as_infinite_residual(self):
        intrinsic = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        truth = np.array([0.12, -0.08, 3.5])
        poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-0.8, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([0.0, -0.7, 0.0])},
            # This held-out camera has a nonsingular but nonstandard projective
            # matrix whose denominator is zero at the fitted point.
            {"R": np.eye(3), "t": np.array([-truth[0], 0.0, 0.0])},
        ]
        singular_projection_intrinsic = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [1.0, 0.0, 0.0],
        ])
        intrinsics = [intrinsic.copy() for _ in range(3)] + [
            singular_projection_intrinsic
        ]
        observations = []
        for camera_id in range(3):
            camera_point = poses[camera_id]["R"] @ truth + poses[camera_id]["t"]
            pixel = intrinsic @ camera_point
            observations.append(pixel[:2] / pixel[2])
        observations.append(np.array([1000.0, 1000.0]))

        _, metrics = triangulate_consensus(
            observations, poses, intrinsics,
            min_views=3,
            inlier_threshold_px=0.5,
            max_reprojection_rms_px=0.5,
        )

        self.assertEqual(metrics["inlier_camera_ids"], [0, 1, 2])
        self.assertEqual(metrics["outlier_camera_ids"], [3])
        self.assertTrue(np.isinf(metrics["reprojection_errors_px"][3]))


if __name__ == "__main__":
    unittest.main()
