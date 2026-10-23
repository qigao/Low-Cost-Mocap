import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialObservation, FiducialTagMap, TagAnchor
from fiducial_localization import (
    CameraBodyExtrinsic,
    FiducialUncertaintyModel,
    map_body_observation_from_fiducials,
)
from fiducial_multitag_pose import solve_multi_tag_camera_pose
from fiducial_pose import (
    FiducialCameraCalibration,
    solve_single_tag_pose,
)
from localization_contract import LocalizationQuality


CAMERA_ID = 2
CAMERA_SERIAL = "CAM-02"
CALIBRATION_SHA = "a" * 64
T0 = 8_000_000
K = np.asarray(((500.0, 0.0, 320.0), (0.0, 510.0, 240.0), (0.0, 0.0, 1.0)))
SIZE_M = 0.16


def make_map(*, moved_tag_id=None, moved_by=(0.0, 0.0, 0.0)):
    anchors = []
    positions = {
        1: (0.0, 0.0, 0.0),
        2: (0.6, 0.0, 0.0),
        3: (0.0, 0.6, 0.0),
        4: (0.6, 0.6, 0.0),
    }
    for tag_id, position in positions.items():
        xyz = np.asarray(position, dtype=float)
        if tag_id == moved_tag_id:
            xyz = xyz + np.asarray(moved_by, dtype=float)
        anchors.append(TagAnchor(
            tag_family="tag36h11",
            tag_id=tag_id,
            map_frame_id="site-map",
            position_xyz_m=tuple(xyz),
            orientation_map_from_tag_xyzw=(0.0, 0.0, 0.0, 1.0),
            physical_size_m=SIZE_M,
            position_stddev_m=(0.002, 0.002, 0.003),
            orientation_stddev_rad=(0.003, 0.003, 0.004),
            reference_method="surveyed",
        ))
    return FiducialTagMap("indoor_dense", "site-map", tuple(anchors))


def camera_calibration():
    return FiducialCameraCalibration(
        camera_id=CAMERA_ID,
        camera_serial=CAMERA_SERIAL,
        intrinsic_matrix=tuple(tuple(float(v) for v in row) for row in K),
        distortion_model="none",
        distortion_coefficients=(),
        calibration_sha256=CALIBRATION_SHA,
    )


def tag_corners(size=SIZE_M):
    h = size * 0.5
    return np.asarray(((-h, h, 0.0), (h, h, 0.0), (h, -h, 0.0), (-h, -h, 0.0)))


def render_observation(anchor, active_map, camera_position, rotation_map_from_camera, *, noise_px=0.0, tag_id_override=None):
    rotation_map_from_tag = Rotation.from_quat(anchor.orientation_map_from_tag_xyzw).as_matrix()
    translation_map_from_tag = np.asarray(anchor.position_xyz_m, dtype=float)
    rotation_camera_from_tag = rotation_map_from_camera.T @ rotation_map_from_tag
    translation_camera_from_tag = rotation_map_from_camera.T @ (
        translation_map_from_tag - camera_position
    )
    points_camera = (rotation_camera_from_tag @ tag_corners().T).T + translation_camera_from_tag
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("synthetic tag must be in front of camera")
    normalized = points_camera[:, :2] / points_camera[:, 2, None]
    homogeneous = (K @ np.column_stack((normalized, np.ones(4))).T).T
    corners = homogeneous[:, :2] / homogeneous[:, 2, None]
    if noise_px:
        # Fixed deterministic pattern; no random seed or nondeterministic test data.
        corners = corners + np.asarray(((noise_px, -noise_px), (-noise_px, noise_px),
                                        (noise_px, noise_px), (-noise_px, -noise_px)))
    return FiducialObservation(
        tag_family="tag36h11",
        tag_id=anchor.tag_id if tag_id_override is None else tag_id_override,
        camera_id=CAMERA_ID,
        camera_serial=CAMERA_SERIAL,
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=T0,
        clock_domain="capture-ptp-A",
        image_corners_px=tuple(tuple(float(v) for v in p) for p in corners),
        corrected_bit_errors=0,
        corner_rms_px=0.2,
        calibration_sha256=CALIBRATION_SHA,
        map_revision_sha256=active_map.revision_sha256,
    )


def localize_frame(active_map, *, noise_px=0.0, map_override=None, wrong_id=None):
    # Camera looks down onto a planar tag field. The body pose is known only to
    # this synthetic generator; the localization algorithm receives image corners.
    camera_position = np.asarray((0.3, 0.25, 1.4))
    rotation_map_from_camera = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    calibration = camera_calibration()
    observations = []
    for anchor in active_map.tags[:3]:
        observation = render_observation(
            anchor, active_map, camera_position, rotation_map_from_camera,
            noise_px=noise_px,
            tag_id_override=wrong_id if wrong_id is not None and anchor.tag_id == 3 else None,
        )
        observations.append(solve_single_tag_pose(
            observation,
            active_map if map_override is None else map_override,
            calibration,
            ambiguity_rms_tolerance_px=0.05,
        ))
    camera_pose = solve_multi_tag_camera_pose(observations, active_map)
    body_observation = map_body_observation_from_fiducials(
        camera_pose,
        active_map,
        CameraBodyExtrinsic(
            camera_id=CAMERA_ID,
            camera_serial=CAMERA_SERIAL,
            camera_frame_id="camera-frame",
            body_frame_id="rigid-body",
            translation_body_from_camera_m=(0.0, 0.0, 0.0),
            orientation_body_from_camera_xyzw=(0.0, 0.0, 0.0, 1.0),
            reference_method="calibrated",
        ),
        FiducialUncertaintyModel(
            model_id="pipeline-test-v1",
            base_position_stddev_m=(0.01, 0.01, 0.015),
            base_orientation_stddev_rad=(0.01, 0.01, 0.01),
            translation_residual_gain=1.5,
            orientation_residual_gain=1.5,
        ),
    )
    return camera_pose, body_observation


class FiducialPipelineReplayTests(unittest.TestCase):
    def test_image_corners_recover_metric_body_pose_end_to_end(self):
        active_map = make_map()
        camera_pose, body = localize_frame(active_map)
        self.assertEqual(camera_pose.inlier_tag_keys, (
            ("tag36h11", 1), ("tag36h11", 2), ("tag36h11", 3)
        ))
        self.assertEqual(body.quality, LocalizationQuality.TRACKING)
        np.testing.assert_allclose(body.position_xyz, (0.3, 0.25, 1.4), atol=1e-5)
        actual_rotation = Rotation.from_quat(body.orientation_xyzw).as_matrix()
        expected_rotation = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        angle = Rotation.from_matrix(expected_rotation.T @ actual_rotation).magnitude()
        self.assertLess(angle, 1e-5)
        self.assertTrue(body.metric_scale_qualified)

    def test_deterministic_corner_noise_is_bounded_and_repeatable(self):
        active_map = make_map()
        first_pose, first_body = localize_frame(active_map, noise_px=0.2)
        second_pose, second_body = localize_frame(active_map, noise_px=0.2)
        self.assertEqual(first_pose, second_pose)
        self.assertEqual(first_body, second_body)
        self.assertLess(first_pose.mean_single_tag_reprojection_rms_px, 2.0)

    def test_wrong_tag_id_fails_closed(self):
        active_map = make_map()
        with self.assertRaises(ValueError):
            localize_frame(active_map, wrong_id=99)

    def test_stale_tag_map_revision_is_rejected(self):
        active_map = make_map()
        moved_map = make_map(moved_tag_id=2, moved_by=(0.1, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "map revision"):
            localize_frame(active_map, map_override=moved_map)

    def test_anchor_perturbation_changes_map_revision(self):
        nominal = make_map()
        perturbed = make_map(moved_tag_id=2, moved_by=(0.005, 0.0, 0.0))
        self.assertNotEqual(nominal.revision_sha256, perturbed.revision_sha256)


if __name__ == "__main__":
    unittest.main()
