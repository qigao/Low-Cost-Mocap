"""Legacy live triangulation adapter tests, without pseyepy/OpenCV imports."""
import ast
from pathlib import Path
import unittest

import numpy as np
from scipy import linalg


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class LegacySparseCameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = [
            np.array([[650., 0., 310.], [0., 700., 240.], [0., 0., 1.]]),
            np.array([[1100., 0., 340.], [0., 950., 230.], [0., 0., 1.]]),
            np.array([[1450., 0., 290.], [0., 1350., 250.], [0., 0., 1.]]),
        ]
        self.poses = [
            {"R": np.eye(3), "t": np.array([0.0, 0.0, 0.0])},
            {"R": rotation_y(0.04), "t": np.array([-0.7, 0.0, 0.0])},
            {"R": rotation_y(-0.06), "t": np.array([0.25, -0.8, 0.0])},
        ]
        self.truth = np.array([0.3, -0.12, 4.8])
        self.pixels = []
        for K, pose in zip(self.K, self.poses):
            homogeneous = K @ (pose["R"] @ self.truth + pose["t"])
            self.pixels.append((homogeneous[:2] / homogeneous[2]).tolist())

        class FakeCameras:
            @classmethod
            def instance(cls):
                return cls()
            def get_camera_params(inner_self, i):
                return {"intrinsic_matrix": self.K[i]}

        class FakeCV:
            @staticmethod
            def projectPoints(xyz, R, t, K, distortion):
                camera_xyz = R @ np.asarray(xyz).reshape(3) + np.asarray(t).reshape(3)
                homogeneous = K @ camera_xyz
                pixel = homogeneous[:2] / homogeneous[2]
                return pixel.reshape(1, 1, 2), None

        source = Path(__file__).resolve().parents[1] / "helpers.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        selected = [
            node for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("triangulate_point", "calculate_reprojection_error")
        ]
        self.assertEqual(len(selected), 2)
        code = compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(source), "exec",
        )
        namespace = {
            "np": np,
            "linalg": linalg,
            "Cameras": FakeCameras,
            "cv": FakeCV,
        }
        exec(code, namespace)
        self.triangulate = namespace["triangulate_point"]
        self.reprojection_error = namespace["calculate_reprojection_error"]

    def test_missing_middle_camera_retains_original_intrinsics(self):
        sparse = [self.pixels[0], [None, None], self.pixels[2]]
        xyz = self.triangulate(sparse, self.poses)
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertLess(self.reprojection_error(sparse, xyz, self.poses), 1e-15)

    def test_missing_first_camera_retains_original_intrinsics(self):
        sparse = [[None, None], self.pixels[1], self.pixels[2]]
        xyz = self.triangulate(sparse, self.poses)
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertLess(self.reprojection_error(sparse, xyz, self.poses), 1e-15)

    def test_three_camera_dense_triangulation_is_consistent(self):
        xyz = self.triangulate(self.pixels, self.poses)
        np.testing.assert_allclose(xyz, self.truth, atol=1e-8)
        self.assertLess(self.reprojection_error(self.pixels, xyz, self.poses), 1e-15)

    def test_missing_or_rank_deficient_views_fail_closed(self):
        sparse = [self.pixels[0], [None, None], [None, None]]
        self.assertEqual(self.triangulate(sparse, self.poses), [None, None, None])
        identical_poses = [self.poses[0], self.poses[0], self.poses[0]]
        duplicate = [self.pixels[0]] * 3
        self.assertEqual(self.triangulate(duplicate, identical_poses), [None, None, None])
        self.assertIsNone(
            self.reprojection_error([[None, None]] * 3, [None] * 3, self.poses)
        )

    def test_partial_observation_rejected_instead_of_reindexed(self):
        sparse = [self.pixels[0], [None, 12.0], self.pixels[2]]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.triangulate(sparse, self.poses)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.reprojection_error(sparse, self.truth, self.poses)


if __name__ == "__main__":
    unittest.main()
