import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialObservation, FiducialTagMap, TagAnchor
from fiducial_pose import (
    FiducialCameraCalibration,
    solve_single_tag_pose,
)


CALIBRATION = "a" * 64
SIZE_M = 0.16
K = (
    (500.0, 0.0, 320.0),
    (0.0, 510.0, 240.0),
    (0.0, 0.0, 1.0),
)


def tag_map():
    return FiducialTagMap(
        profile="indoor_dense",
        map_frame_id="indoor-map",
        tags=(
            TagAnchor(
                tag_family="tag36h11",
                tag_id=7,
                map_frame_id="indoor-map",
                position_xyz_m=(1.0, 2.0, 0.5),
                orientation_map_from_tag_xyzw=(0.0, 0.0, 0.0, 1.0),
                physical_size_m=SIZE_M,
                position_stddev_m=(0.001, 0.001, 0.002),
                orientation_stddev_rad=(0.002, 0.002, 0.003),
                reference_method="calibrated",
            ),
        ),
    )


def calibration(*, distortion_model="none", distortion_coefficients=()):
    return FiducialCameraCalibration(
        camera_id=2,
        camera_serial="CAM-02",
        intrinsic_matrix=K,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
        calibration_sha256=CALIBRATION,
    )


def object_corners():
    half = SIZE_M * 0.5
    return np.asarray([
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ])


def distort(points, coefficients):
    if not coefficients:
        return points
    k1, k2, p1, p2, k3 = coefficients
    x = points[:, 0]
    y = points[:, 1]
    radius2 = x * x + y * y
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    return np.column_stack((
        x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
        y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
    ))


def project(rotation, translation, *, coefficients=()):
    points = (rotation @ object_corners().T).T + translation
    normalized = points[:, :2] / points[:, 2, None]
    normalized = distort(normalized, coefficients)
    intrinsic = np.asarray(K)
    homogeneous = (
        intrinsic @ np.column_stack((normalized, np.ones(4))).T
    ).T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def observation(corners, tag_map_value=None, **changes):
    active_map = tag_map_value or tag_map()
    values = dict(
        tag_family="tag36h11",
        tag_id=7,
        camera_id=2,
        camera_serial="CAM-02",
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        image_corners_px=tuple(tuple(float(v) for v in point) for point in corners),
        corrected_bit_errors=0,
        corner_rms_px=0.2,
        calibration_sha256=CALIBRATION,
        map_revision_sha256=active_map.revision_sha256,
    )
    values.update(changes)
    return FiducialObservation(**values)


def rotation_error_rad(first, second):
    relative = first.T @ second
    cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) * 0.5))
    return math.acos(cosine)


class FiducialCameraCalibrationTests(unittest.TestCase):
    def test_valid_none_and_brown5_calibrations(self):
        self.assertEqual(calibration().distortion_coefficients, ())
        brown = calibration(
            distortion_model="brown5",
            distortion_coefficients=(-0.12, 0.03, 0.001, -0.0005, -0.005),
        )
        self.assertEqual(len(brown.distortion_coefficients), 5)

    def test_rejects_bad_intrinsics_and_distortion_contract(self):
        with self.assertRaisesRegex(ValueError, "singular"):
            FiducialCameraCalibration(
                2, "CAM-02",
                ((0.0, 0.0, 0.0),) * 3,
                "none", (), CALIBRATION,
            )
        with self.assertRaisesRegex(ValueError, "zero coefficients"):
            calibration(distortion_model="none", distortion_coefficients=(0.1,))
        with self.assertRaisesRegex(ValueError, "five coefficients"):
            calibration(distortion_model="brown5", distortion_coefficients=(0.1, 0.2))
        with self.assertRaisesRegex(ValueError, "distortion_model"):
            calibration(distortion_model="fisheye4", distortion_coefficients=())


class SingleTagPoseTests(unittest.TestCase):
    def test_recovers_metric_camera_from_tag_pose(self):
        active_map = tag_map()
        expected_rotation = Rotation.from_euler(
            "xyz", (170.0, -10.0, 4.0), degrees=True
        ).as_matrix()
        expected_translation = np.asarray((0.04, -0.02, 1.2))
        corners = project(expected_rotation, expected_translation)

        result = solve_single_tag_pose(
            observation(corners, active_map),
            active_map,
            calibration(),
            ambiguity_rms_tolerance_px=0.05,
        )

        actual_rotation = Rotation.from_quat(
            result.orientation_camera_from_tag_xyzw
        ).as_matrix()
        self.assertLess(
            rotation_error_rad(actual_rotation, expected_rotation), 1e-6
        )
        np.testing.assert_allclose(
            result.translation_camera_from_tag_m,
            expected_translation,
            atol=1e-7,
        )
        self.assertLess(result.reprojection_rms_px, 1e-8)
        self.assertGreater(result.image_area_px2, 1000.0)
        self.assertGreater(result.minimum_corner_depth_m, 1.0)
        self.assertEqual(result.calibration_sha256, CALIBRATION)
        self.assertEqual(result.map_revision_sha256, active_map.revision_sha256)

    def test_recovers_pose_through_brown5_distortion(self):
        active_map = tag_map()
        coefficients = (-0.12, 0.03, 0.001, -0.0005, -0.005)
        expected_rotation = Rotation.from_euler(
            "xyz", (168.0, 8.0, -6.0), degrees=True
        ).as_matrix()
        expected_translation = np.asarray((-0.03, 0.015, 1.0))
        corners = project(
            expected_rotation, expected_translation, coefficients=coefficients
        )

        result = solve_single_tag_pose(
            observation(corners, active_map),
            active_map,
            calibration(
                distortion_model="brown5",
                distortion_coefficients=coefficients,
            ),
            ambiguity_rms_tolerance_px=0.05,
        )
        actual_rotation = Rotation.from_quat(
            result.orientation_camera_from_tag_xyzw
        ).as_matrix()
        self.assertLess(
            rotation_error_rad(actual_rotation, expected_rotation), 1e-5
        )
        np.testing.assert_allclose(
            result.translation_camera_from_tag_m,
            expected_translation,
            atol=1e-6,
        )
        self.assertLess(result.reprojection_rms_px, 1e-7)

    def test_rejects_competing_far_planar_pose(self):
        active_map = tag_map()
        # Fixed noisy far-range observation with two physically distinct,
        # positive-depth planar pose minima whose reprojection RMS values differ
        # by less than 0.001 px. A best-RMS-only solver would silently choose one.
        corners = (
            (312.77765398, 229.41449748),
            (328.92789719, 230.27205104),
            (328.33069851, 246.05172770),
            (312.17834596, 245.56650407),
        )
        with self.assertRaisesRegex(ValueError, "planar tag pose is ambiguous"):
            solve_single_tag_pose(
                observation(corners, active_map),
                active_map,
                calibration(),
                max_reprojection_rms_px=1.0,
                ambiguity_rms_tolerance_px=0.01,
            )

    def test_rejects_calibration_identity_revision_and_map_revision_mismatch(self):
        active_map = tag_map()
        rotation = Rotation.from_euler(
            "xyz", (170.0, -10.0, 4.0), degrees=True
        ).as_matrix()
        corners = project(rotation, np.asarray((0.04, -0.02, 1.2)))
        base = observation(corners, active_map)

        with self.assertRaisesRegex(ValueError, "camera ID mismatch"):
            solve_single_tag_pose(
                base,
                active_map,
                replace(calibration(), camera_id=5),
            )
        with self.assertRaisesRegex(ValueError, "calibration revision mismatch"):
            solve_single_tag_pose(
                base,
                active_map,
                replace(calibration(), calibration_sha256="b" * 64),
            )
        with self.assertRaisesRegex(ValueError, "tag map revision mismatch"):
            solve_single_tag_pose(
                replace(base, map_revision_sha256="c" * 64),
                active_map,
                calibration(),
            )

    def test_rejects_weak_image_area_and_excessive_reprojection(self):
        active_map = tag_map()
        tiny = (
            (100.0, 100.0),
            (102.0, 100.0),
            (102.0, 102.0),
            (100.0, 102.0),
        )
        with self.assertRaisesRegex(ValueError, "image area"):
            solve_single_tag_pose(
                observation(tiny, active_map),
                active_map,
                calibration(),
                min_image_area_px2=16.0,
            )

        rotation = Rotation.from_euler(
            "xyz", (170.0, -10.0, 4.0), degrees=True
        ).as_matrix()
        corners = project(rotation, np.asarray((0.04, -0.02, 1.2)))
        noisy = corners.copy()
        noisy[0] += np.asarray((3.0, -2.0))
        with self.assertRaisesRegex(ValueError, "reprojection RMS"):
            solve_single_tag_pose(
                observation(noisy, active_map),
                active_map,
                calibration(),
                max_reprojection_rms_px=0.05,
                ambiguity_rms_tolerance_px=0.0,
            )

    def test_rejects_pose_outside_viewing_angle_gate(self):
        active_map = tag_map()
        rotation = Rotation.from_euler(
            "xyz", (165.0, -12.0, 3.0), degrees=True
        ).as_matrix()
        corners = project(rotation, np.asarray((0.02, -0.01, 1.0)))
        with self.assertRaisesRegex(
            ValueError, "no positive-depth front-facing tag pose candidate"
        ):
            solve_single_tag_pose(
                observation(corners, active_map),
                active_map,
                calibration(),
                max_viewing_angle_deg=5.0,
            )

    def test_threshold_arguments_fail_closed(self):
        active_map = tag_map()
        rotation = Rotation.from_euler(
            "xyz", (170.0, -10.0, 4.0), degrees=True
        ).as_matrix()
        obs = observation(
            project(rotation, np.asarray((0.04, -0.02, 1.2))),
            active_map,
        )
        for name, changes in (
            ("max_reprojection_rms_px", {"max_reprojection_rms_px": -1.0}),
            ("min_image_area_px2", {"min_image_area_px2": float("nan")}),
            ("max_viewing_angle_deg", {"max_viewing_angle_deg": 91.0}),
            ("ambiguity_rms_tolerance_px", {"ambiguity_rms_tolerance_px": -0.1}),
            ("min_depth_m", {"min_depth_m": -1.0}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                solve_single_tag_pose(
                    obs, active_map, calibration(), **changes
                )


if __name__ == "__main__":
    unittest.main()
