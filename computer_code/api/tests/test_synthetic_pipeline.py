"""Synthetic integration check for pairwise association, triangulation, and rigid fitting."""
import unittest
import numpy as np

from geometry import triangulate_point
from multiview_association import associate_pairwise
from rigid_pattern import fit_rigid_pattern


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_poses(K1, pose1, K2, pose2):
    R1 = np.asarray(pose1["R"], dtype=float)
    t1 = np.asarray(pose1["t"], dtype=float).reshape(3)
    R2 = np.asarray(pose2["R"], dtype=float)
    t2 = np.asarray(pose2["t"], dtype=float).reshape(3)
    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1
    return np.linalg.inv(K2).T @ skew(t21) @ R21 @ np.linalg.inv(K1)


def project(points, K, pose):
    R = np.asarray(pose["R"], dtype=float)
    t = np.asarray(pose["t"], dtype=float).reshape(3)
    camera = (R @ np.asarray(points).T).T + t
    pixels = (K @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


class SyntheticPipelineTests(unittest.TestCase):
    def test_associate_then_triangulate_then_fit_fixed_pattern(self):
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([-0.8, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([0.0, -0.6, 0.0])},
        ]
        model = np.array([
            [-0.10, -0.04, 0.0],
            [0.05, -0.04, 0.0],
            [-0.10, 0.055, 0.0],
            [-0.10, -0.04, 0.08],
        ])
        angle = 0.23
        R_obj = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_obj = np.array([0.08, -0.03, 3.2])
        world = (R_obj @ model.T).T + t_obj
        pixels = [project(world, K, pose) for pose in poses]

        # Permute camera 1 detections and ensure epipolar assignment recovers
        # the correct indices rather than relying on detector list order.
        permutation = np.array([2, 0, 3, 1])
        F01 = fundamental_from_poses(K, poses[0], K, poses[1])
        matches = associate_pairwise(pixels[0], pixels[1][permutation], F01, max_sampson_error_px=0.01)
        by_left = {match.left_index: match.right_index for match in matches}
        self.assertEqual(len(by_left), len(model))
        for marker_index in range(len(model)):
            self.assertEqual(permutation[by_left[marker_index]], marker_index)

        reconstructed = []
        for marker_index in range(len(model)):
            right_index = by_left[marker_index]
            camera1_observation = pixels[1][permutation[right_index]]
            observations = [pixels[0][marker_index], camera1_observation, pixels[2][marker_index]]
            xyz, metrics = triangulate_point(observations, poses, [K, K, K], max_reprojection_rms_px=1e-5)
            reconstructed.append(xyz)
            self.assertLess(metrics["reprojection_rms_px"], 1e-5)

        R_fit, t_fit, metrics = fit_rigid_pattern(model, np.asarray(reconstructed), max_rms_error=1e-5)
        np.testing.assert_allclose(R_fit, R_obj, atol=1e-8)
        np.testing.assert_allclose(t_fit, t_obj, atol=1e-8)
        self.assertLess(metrics["rms_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
