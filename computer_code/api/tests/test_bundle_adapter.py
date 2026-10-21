"""Legacy runtime BA adapter verified without camera SDK or OpenCV imports."""
import ast
from pathlib import Path
import unittest

import numpy as np


class BundleAdapterTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "helpers.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        func = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "bundle_adjustment"
        )
        self.K = [
            np.array([[700., 0., 320.], [0., 725., 240.], [0., 0., 1.]]),
            np.array([[950., 0., 300.], [0., 980., 250.], [0., 0., 1.]]),
            np.array([[1200., 0., 310.], [0., 1100., 235.], [0., 0., 1.]]),
        ]
        self.poses = [
            {"R": np.eye(3), "t": np.array([0., 0., 0.])},
            {"R": np.eye(3), "t": np.array([-1., 0., 0.])},
            {"R": np.eye(3), "t": np.array([0., -1., 0.])},
        ]
        self.observations = np.zeros((12, 3, 2), dtype=np.float64)
        self.calls = []
        self.emitted = []

        class FakeCameras:
            @classmethod
            def instance(cls):
                return cls()
            def get_camera_params(instance, camera_id):
                return {"intrinsic_matrix": self.K[camera_id].copy()}

        def fake_solver(points, poses, intrinsics):
            self.calls.append((points, poses, intrinsics))
            return poses, {"reprojection_rms_px": 0.0}

        namespace = {
            "Cameras": FakeCameras,
            "refine_calibrated_bundle": fake_solver,
        }
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
            str(source), "exec"
        )
        exec(code, namespace)
        self.run = namespace["bundle_adjustment"]

    def test_uses_every_camera_intrinsic_and_preserves_caller_inputs(self):
        original = [K.copy() for K in self.K]
        return_value = self.run(
            self.observations, self.poses,
            type("Sink", (), {"emit": lambda _, *args: self.emitted.append(args)})()
        )
        self.assertIs(return_value, self.poses)
        self.assertEqual(len(self.calls), 1)
        points, poses, intrinsics = self.calls[0]
        self.assertIs(points, self.observations)
        self.assertIs(poses, self.poses)
        for expected, passed in zip(self.K, intrinsics):
            np.testing.assert_array_equal(passed, expected)
        for expected, unchanged in zip(original, self.K):
            np.testing.assert_array_equal(unchanged, expected)
        self.assertEqual(self.emitted, [])

    def test_solver_error_is_not_caught_or_emitted_as_partial_solution(self):
        def fail_solver(*args):
            raise ValueError("insufficient calibrated observations")

        self.run.__globals__["refine_calibrated_bundle"] = fail_solver
        with self.assertRaisesRegex(ValueError, "insufficient calibrated observations"):
            self.run(self.observations, self.poses, None)


if __name__ == "__main__":
    unittest.main()
