"""Hardware-free contract tests for the legacy calibration event adapter.

Compile only the route function's AST, avoiding serial, camera SDK and OpenCV
imports. Mock the cv.sfm boundary; execute the *actual* pure calibration math.
"""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from calibration_geometry import (
    compose_world_to_camera_pose,
    select_relative_pose_by_cheirality,
)


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project(world, intrinsic, pose):
    camera = (pose["R"] @ world.T).T + pose["t"]
    homogeneous = (intrinsic @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


class CalibrationAdapterTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "index.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "calculate_camera_pose"
        )
        node.decorator_list = []
        program = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))

        self.K = [
            np.array([[800.0, 0, 320.0], [0, 790.0, 240.0], [0, 0, 1]]),
            np.array([[1100.0, 0, 310.0], [0, 1050.0, 250.0], [0, 0, 1]]),
            np.array([[1350.0, 0, 330.0], [0, 1320.0, 230.0], [0, 0, 1]]),
        ]
        self.rel01 = {"R": rotation_y(0.09), "t": np.array([-1.0, 0.02, 0.0])}
        self.rel12 = {"R": rotation_y(-0.14), "t": np.array([-0.9, -0.03, 0.01])}
        first = {"R": np.eye(3), "t": np.zeros(3)}
        second = compose_world_to_camera_pose(first, self.rel01, baseline_length=1.0)
        third = compose_world_to_camera_pose(second, self.rel12, baseline_length=1.0)
        self.poses = [first, second, third]
        rng = np.random.default_rng(2026)
        world = np.column_stack([
            rng.uniform(-0.6, 0.6, 20),
            rng.uniform(-0.4, 0.4, 20),
            rng.uniform(4.0, 8.0, 20),
        ])
        self.observations = [
            project(world, K, pose) for K, pose in zip(self.K, self.poses)
        ]
        self.emitted = []
        self.intrinsic_pairs = []

        class FakeCameras:
            num_cameras = 3
            @classmethod
            def instance(cls):
                return cls()
            def get_camera_params(inner_self, camera_id):
                return {"intrinsic_matrix": self.K[camera_id]}

        def find_fundamental(left, right, *args):
            return np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float), np.ones((len(left), 1), np.uint8)

        def essential_from_fundamental(fundamental, K1, K2):
            pair = (
                next(i for i, K in enumerate(self.K) if np.array_equal(K1, K)),
                next(i for i, K in enumerate(self.K) if np.array_equal(K2, K)),
            )
            self.intrinsic_pairs.append(pair)
            return pair

        def motion_from_essential(pair):
            relative = {(0, 1): self.rel01, (1, 2): self.rel12}[pair]
            wrong_R = rotation_y(np.pi) @ relative["R"]
            true_R = relative["R"]
            true_t = relative["t"]
            return (
                [wrong_R, true_R, wrong_R, true_R],
                [true_t, -true_t, -true_t, true_t],
            )

        self.cv = SimpleNamespace(
            FM_RANSAC=1,
            findFundamentalMat=find_fundamental,
            sfm=SimpleNamespace(
                essentialFromFundamental=essential_from_fundamental,
                motionFromEssential=motion_from_essential,
            ),
        )
        def bundle_adjustment(image_points, camera_poses, socketio):
            return camera_poses

        def serialize(camera_poses):
            return [
                {"R": pose["R"].tolist(), "t": pose["t"].tolist()}
                for pose in camera_poses
            ]

        namespace = {
            "np": np,
            "cv": self.cv,
            "Cameras": FakeCameras,
            "socketio": SimpleNamespace(
                emit=lambda name, payload: self.emitted.append((name, payload))
            ),
            "bundle_adjustment": bundle_adjustment,
            "compose_world_to_camera_pose": compose_world_to_camera_pose,
            "select_relative_pose_by_cheirality": select_relative_pose_by_cheirality,
            "camera_pose_to_serializable": serialize,
        }
        exec(compile(program, str(source), "exec"), namespace)
        self.handler = namespace["calculate_camera_pose"]

    def test_three_camera_pose_chain_uses_each_pair_intrinsics(self):
        frames = np.stack(self.observations, axis=1).tolist()
        self.handler({"cameraPoints": frames})
        self.assertEqual(self.intrinsic_pairs, [(0, 1), (1, 2)])
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0][0], "camera-pose")
        result = self.emitted[0][1]
        self.assertNotIn("error", result)
        self.assertEqual(len(result["camera_poses"]), 3)
        for actual, expected in zip(result["camera_poses"], self.poses):
            np.testing.assert_allclose(actual["R"], expected["R"], atol=1e-9)
            np.testing.assert_allclose(actual["t"], expected["t"], atol=1e-9)

    def test_missing_inliers_fails_without_partial_calibration(self):
        observations = np.stack(self.observations, axis=1).astype(object)
        observations[:15, 2, :] = None
        self.handler({"cameraPoints": observations.tolist()})
        self.assertEqual(len(self.emitted), 1)
        self.assertIn("error", self.emitted[0][1])
        self.assertIn("camera pair 1/2", self.emitted[0][1]["error"])

    def test_calibration_event_uses_real_gauge_fixed_bundle_solver(self):
        from bundle_geometry import refine_calibrated_bundle

        actual_metrics = []
        def real_bundle(image_points, camera_poses, socketio):
            refined, metrics = refine_calibrated_bundle(
                image_points, camera_poses, self.K, min_tracks=12
            )
            actual_metrics.append(metrics)
            return refined

        self.handler.__globals__["bundle_adjustment"] = real_bundle
        frames = np.stack(self.observations, axis=1).tolist()
        self.handler({"cameraPoints": frames})
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.intrinsic_pairs, [(0, 1), (1, 2)])
        self.assertNotIn("error", self.emitted[0][1])
        self.assertEqual(len(actual_metrics), 1)
        self.assertEqual(actual_metrics[0]["signed_residual_count"], 2 * 20 * 3)
        self.assertTrue(actual_metrics[0]["camera_one_translation_fixed"])
        self.assertTrue(actual_metrics[0]["intrinsics_fixed"])
        self.assertFalse(actual_metrics[0]["translation_scale_known"])
        for actual, expected in zip(
            self.emitted[0][1]["camera_poses"], self.poses
        ):
            np.testing.assert_allclose(actual["R"], expected["R"], atol=1e-6)
            np.testing.assert_allclose(actual["t"], expected["t"], atol=1e-6)

    def test_bundle_failure_does_not_publish_partial_camera_pose(self):
        def failing_bundle(image_points, camera_poses, socketio):
            raise ValueError("bundle did not converge")

        self.handler.__globals__["bundle_adjustment"] = failing_bundle
        self.handler({"cameraPoints": np.stack(self.observations, axis=1).tolist()})
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0][0], "camera-pose")
        self.assertEqual(self.emitted[0][1], {"error": "bundle did not converge"})

    def test_invalid_ransac_result_fails_closed(self):
        self.cv.findFundamentalMat = lambda *_args: (None, None)
        self.handler({"cameraPoints": np.stack(self.observations, axis=1).tolist()})
        self.assertEqual(len(self.emitted), 1)
        self.assertIn("fundamental matrix", self.emitted[0][1]["error"])


if __name__ == "__main__":
    unittest.main()
