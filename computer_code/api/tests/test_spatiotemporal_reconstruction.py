"""Physical camera IDs must survive 2D -> 3D -> rigid pattern fitting."""
import unittest

import numpy as np

from spatiotemporal_reconstruction import (
    fit_synchronized_rigid_pattern,
    reconstruct_synchronized_markers,
)
from temporal_pattern import (
    FrameStamp, StampedIdentifiedDetection, group_synchronized_detections,
)


class PhysicalCameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera_ids = (2, 5, 9)
        self.K = {
            2: np.array([[800., 0., 320.], [0., 790., 240.], [0., 0., 1.]]),
            5: np.array([[1100., 0., 300.], [0., 1050., 230.], [0., 0., 1.]]),
            9: np.array([[1300., 0., 310.], [0., 1200., 250.], [0., 0., 1.]]),
        }
        self.poses = {
            2: {"R": np.eye(3), "t": np.zeros(3)},
            5: {"R": np.eye(3), "t": np.array([-0.8, 0., 0.])},
            9: {"R": np.eye(3), "t": np.array([0., -0.6, 0.])},
        }
        self.model = {
            10: [-0.10, -0.04, 0.0],
            11: [0.05, -0.04, 0.0],
            12: [-0.10, 0.055, 0.0],
            13: [-0.10, -0.04, 0.08],
        }
        theta = 0.21
        c, s = np.cos(theta), np.sin(theta)
        self.true_R = np.array([
            [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]
        ])
        self.true_t = np.array([0.08, -0.03, 3.2])
        self.xyz = {
            marker: self.true_R @ np.asarray(p) + self.true_t
            for marker, p in self.model.items()
        }
        self.frames = [
            FrameStamp(camera, 500, 1_000_000 + idx * 20, "hw-clock")
            for idx, camera in enumerate(self.camera_ids)
        ]
        detections = []
        for idx, camera in enumerate(self.camera_ids):
            for marker in reversed(list(self.model)):
                if marker == 13 and camera == 5:
                    continue
                pos = self.poses[camera]
                camera_xyz = pos["R"] @ self.xyz[marker] + pos["t"]
                h = self.K[camera] @ camera_xyz
                xy = tuple((h[:2] / h[2]).tolist())
                detections.append(StampedIdentifiedDetection(
                    self.frames[idx], marker, xy,
                ))
        self.detections = detections

    def group(self, detections=None):
        return group_synchronized_detections(
            self.detections if detections is None else detections,
            self.frames, self.camera_ids,
            max_exposure_skew_ns=50,
        )

    def test_noncontiguous_calibration_mapping_preserves_marker_geometry(self):
        batch, grouped = self.group()
        reconstructed = reconstruct_synchronized_markers(
            batch, grouped, self.poses, self.K,
            max_reprojection_rms_px=1e-6,
        )
        self.assertEqual(reconstructed.camera_ids, (2, 5, 9))
        self.assertEqual(tuple(sorted(reconstructed.markers)), (10, 11, 12, 13))
        self.assertEqual(reconstructed.markers[13].camera_ids, (2, 9))
        self.assertEqual(
            tuple(reconstructed.markers[13].reprojection_errors_px_by_camera),
            (2, 9),
        )
        for marker in self.model:
            np.testing.assert_allclose(
                reconstructed.markers[marker].xyz,
                self.xyz[marker], atol=1e-8,
            )
        fitted = fit_synchronized_rigid_pattern(
            reconstructed, self.model, max_rms_error=1e-6
        )
        np.testing.assert_allclose(fitted.rotation, self.true_R, atol=1e-8)
        np.testing.assert_allclose(fitted.translation, self.true_t, atol=1e-8)
        self.assertLess(fitted.rigid_rms_error, 1e-8)
        self.assertFalse(fitted.metric_scale_validated)
        self.assertEqual(fitted.marker_ids, (10, 11, 12, 13))

    def test_omitted_or_extra_camera_calibration_fails_closed(self):
        batch, grouped = self.group()
        for poses, intrinsics in (
            ({2: self.poses[2], 5: self.poses[5]}, self.K),
            (self.poses, {2: self.K[2], 9: self.K[9]}),
            (self.poses, {**self.K, 0: self.K[2]}),
        ):
            with self.subTest(camera_poses=tuple(poses), intrinsics=tuple(intrinsics)):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    reconstruct_synchronized_markers(
                        batch, grouped, poses, intrinsics
                    )

    def test_geometry_outlier_rejects_entire_frame_not_partial_pose(self):
        corrupt = list(self.detections)
        for index, item in enumerate(corrupt):
            if item.marker_id == 10 and item.frame_stamp.camera_id == 5:
                corrupt[index] = StampedIdentifiedDetection(
                    item.frame_stamp, 10, (item.xy[0] + 120, item.xy[1] - 85)
                )
        batch, grouped = self.group(corrupt)
        with self.assertRaisesRegex(ValueError, "reprojection RMS"):
            reconstruct_synchronized_markers(
                batch, grouped, self.poses, self.K,
                max_reprojection_rms_px=0.2,
            )

    def test_rigid_pattern_rejects_unknown_marker_and_wrong_geometry(self):
        batch, grouped = self.group()
        reconstructed = reconstruct_synchronized_markers(
            batch, grouped, self.poses, self.K,
            max_reprojection_rms_px=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            fit_synchronized_rigid_pattern(
                reconstructed, {10: self.model[10], 11: self.model[11],
                                12: self.model[12]}
            )
        wrong_model = {
            marker: np.asarray(value).copy()
            for marker, value in self.model.items()
        }
        wrong_model[13] += [0.0, 0.0, 0.5]
        with self.assertRaisesRegex(ValueError, "RMS"):
            fit_synchronized_rigid_pattern(
                reconstructed, wrong_model, max_rms_error=0.01,
            )


if __name__ == "__main__":
    unittest.main()
