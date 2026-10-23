import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialTagMap, TagAnchor
from fiducial_multitag_pose import solve_multi_tag_camera_pose
from fiducial_pose import FiducialTagPoseMeasurement


CALIBRATION = "a" * 64


def anchor(tag_id, position, *, yaw_deg=0.0):
    quaternion = Rotation.from_euler(
        "z", yaw_deg, degrees=True
    ).as_quat()
    return TagAnchor(
        tag_family="tag36h11",
        tag_id=tag_id,
        map_frame_id="site-map",
        position_xyz_m=position,
        orientation_map_from_tag_xyzw=tuple(quaternion),
        physical_size_m=0.16,
        position_stddev_m=(0.001, 0.001, 0.002),
        orientation_stddev_rad=(0.002, 0.002, 0.003),
        reference_method="calibrated",
    )


def active_map():
    return FiducialTagMap(
        "indoor_dense",
        "site-map",
        (
            anchor(1, (0.0, 0.0, 0.0), yaw_deg=0.0),
            anchor(2, (1.0, 0.0, 0.1), yaw_deg=15.0),
            anchor(3, (0.0, 1.0, 0.2), yaw_deg=-20.0),
            anchor(4, (1.0, 1.0, 0.3), yaw_deg=35.0),
        ),
    )


def measurement_for_pose(
    tag_map,
    tag_id,
    rotation_map_from_camera,
    translation_map_from_camera,
    *,
    reprojection_rms_px=0.2,
    **changes,
):
    tag = tag_map.by_key[("tag36h11", tag_id)]
    rotation_map_from_tag = Rotation.from_quat(
        tag.orientation_map_from_tag_xyzw
    ).as_matrix()
    translation_map_from_tag = np.asarray(tag.position_xyz_m)
    rotation_camera_from_tag = (
        rotation_map_from_camera.T @ rotation_map_from_tag
    )
    translation_camera_from_tag = (
        rotation_map_from_camera.T
        @ (translation_map_from_tag - translation_map_from_camera)
    )
    values = dict(
        tag_family="tag36h11",
        tag_id=tag_id,
        camera_id=2,
        camera_serial="CAM-02",
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        translation_camera_from_tag_m=tuple(translation_camera_from_tag),
        orientation_camera_from_tag_xyzw=tuple(
            Rotation.from_matrix(rotation_camera_from_tag).as_quat()
        ),
        reprojection_rms_px=reprojection_rms_px,
        image_area_px2=2000.0,
        viewing_angle_deg=10.0,
        minimum_corner_depth_m=1.0,
        candidate_count=2,
        ambiguity_rms_margin_px=0.8,
        calibration_sha256=CALIBRATION,
        map_revision_sha256=tag_map.revision_sha256,
    )
    values.update(changes)
    return FiducialTagPoseMeasurement(**values)


def rotation_error(first, second):
    relative = first.T @ second
    cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) * 0.5))
    return math.acos(cosine)


class MultiTagCameraPoseTests(unittest.TestCase):
    def setUp(self):
        self.tag_map = active_map()
        self.rotation = Rotation.from_euler(
            "xyz", (4.0, -7.0, 18.0), degrees=True
        ).as_matrix()
        self.translation = np.asarray((0.35, -0.2, 1.4))

    def measurements(self, ids=(1, 2, 3)):
        return [
            measurement_for_pose(
                self.tag_map,
                tag_id,
                self.rotation,
                self.translation,
            )
            for tag_id in ids
        ]

    def test_exact_multi_tag_consensus_recovers_map_from_camera(self):
        result = solve_multi_tag_camera_pose(
            self.measurements(),
            self.tag_map,
        )
        actual_rotation = Rotation.from_quat(
            result.orientation_map_from_camera_xyzw
        ).as_matrix()
        self.assertLess(rotation_error(actual_rotation, self.rotation), 1e-10)
        np.testing.assert_allclose(
            result.position_map_from_camera_m,
            self.translation,
            atol=1e-12,
        )
        self.assertEqual(
            result.inlier_tag_keys,
            (("tag36h11", 1), ("tag36h11", 2), ("tag36h11", 3)),
        )
        self.assertEqual(result.excluded_tag_keys, ())
        self.assertLess(result.max_translation_residual_m, 1e-12)
        self.assertLess(result.max_orientation_residual_rad, 1e-10)
        self.assertEqual(result.map_frame_id, "site-map")

    def test_one_pose_outlier_is_excluded_with_three_tags(self):
        measurements = self.measurements()
        outlier_translation = self.translation + np.asarray((0.5, 0.0, 0.0))
        measurements[2] = measurement_for_pose(
            self.tag_map,
            3,
            self.rotation,
            outlier_translation,
        )
        result = solve_multi_tag_camera_pose(
            measurements,
            self.tag_map,
            max_translation_disagreement_m=0.08,
        )
        self.assertEqual(
            result.inlier_tag_keys,
            (("tag36h11", 1), ("tag36h11", 2)),
        )
        self.assertEqual(
            result.excluded_tag_keys,
            (("tag36h11", 3),),
        )
        np.testing.assert_allclose(
            result.position_map_from_camera_m,
            self.translation,
            atol=1e-12,
        )

    def test_orientation_outlier_is_excluded(self):
        measurements = self.measurements()
        outlier_rotation = (
            Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
            @ self.rotation
        )
        measurements[2] = measurement_for_pose(
            self.tag_map,
            3,
            outlier_rotation,
            self.translation,
        )
        result = solve_multi_tag_camera_pose(
            measurements,
            self.tag_map,
            max_orientation_disagreement_rad=math.radians(8.0),
        )
        self.assertEqual(
            result.inlier_tag_keys,
            (("tag36h11", 1), ("tag36h11", 2)),
        )
        self.assertEqual(
            result.excluded_tag_keys,
            (("tag36h11", 3),),
        )

    def test_equal_support_two_vs_two_clusters_fail_closed(self):
        second_translation = self.translation + np.asarray((0.4, 0.0, 0.0))
        measurements = [
            measurement_for_pose(
                self.tag_map, 1, self.rotation, self.translation
            ),
            measurement_for_pose(
                self.tag_map, 2, self.rotation, self.translation
            ),
            measurement_for_pose(
                self.tag_map, 3, self.rotation, second_translation
            ),
            measurement_for_pose(
                self.tag_map, 4, self.rotation, second_translation
            ),
        ]
        with self.assertRaisesRegex(
            ValueError, "ambiguous competing multi-tag pose clusters"
        ):
            solve_multi_tag_camera_pose(
                measurements,
                self.tag_map,
                max_translation_disagreement_m=0.08,
            )

    def test_high_reprojection_tag_is_preexcluded(self):
        measurements = self.measurements()
        measurements[2] = replace(
            measurements[2], reprojection_rms_px=4.0
        )
        result = solve_multi_tag_camera_pose(
            measurements,
            self.tag_map,
            max_single_tag_reprojection_rms_px=1.0,
        )
        self.assertEqual(
            result.inlier_tag_keys,
            (("tag36h11", 1), ("tag36h11", 2)),
        )
        self.assertEqual(
            result.excluded_tag_keys,
            (("tag36h11", 3),),
        )

    def test_mixed_metadata_duplicate_and_unknown_tags_fail_closed(self):
        measurements = self.measurements()
        with self.assertRaisesRegex(ValueError, "mix frame/camera/time/revision"):
            solve_multi_tag_camera_pose(
                [measurements[0], replace(measurements[1], frame_id=899)],
                self.tag_map,
            )
        with self.assertRaisesRegex(ValueError, "duplicate tag identity"):
            solve_multi_tag_camera_pose(
                [measurements[0], measurements[0]],
                self.tag_map,
            )
        with self.assertRaisesRegex(ValueError, "outside the active map"):
            solve_multi_tag_camera_pose(
                [measurements[0], replace(measurements[1], tag_id=99)],
                self.tag_map,
            )
        with self.assertRaisesRegex(ValueError, "map revision"):
            solve_multi_tag_camera_pose(
                [
                    replace(measurements[0], map_revision_sha256="b" * 64),
                    replace(measurements[1], map_revision_sha256="b" * 64),
                ],
                self.tag_map,
            )

    def test_no_consensus_and_insufficient_quality_fail_closed(self):
        measurements = self.measurements(ids=(1, 2))
        second_pose = self.translation + np.asarray((0.5, 0.0, 0.0))
        measurements[1] = measurement_for_pose(
            self.tag_map, 2, self.rotation, second_pose
        )
        with self.assertRaisesRegex(ValueError, "sufficient support"):
            solve_multi_tag_camera_pose(
                measurements,
                self.tag_map,
                max_translation_disagreement_m=0.08,
            )

        weak = [
            replace(measurement, reprojection_rms_px=5.0)
            for measurement in self.measurements()
        ]
        with self.assertRaisesRegex(ValueError, "insufficient usable"):
            solve_multi_tag_camera_pose(
                weak,
                self.tag_map,
                max_single_tag_reprojection_rms_px=1.0,
            )

    def test_thresholds_are_bounded(self):
        measurements = self.measurements()
        for name, changes in (
            ("min_tags", {"min_tags": 1}),
            (
                "max_translation_disagreement_m",
                {"max_translation_disagreement_m": -0.1},
            ),
            (
                "max_orientation_disagreement_rad",
                {"max_orientation_disagreement_rad": math.pi + 0.1},
            ),
            (
                "max_single_tag_reprojection_rms_px",
                {"max_single_tag_reprojection_rms_px": float("nan")},
            ),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                solve_multi_tag_camera_pose(
                    measurements,
                    self.tag_map,
                    **changes,
                )


if __name__ == "__main__":
    unittest.main()
