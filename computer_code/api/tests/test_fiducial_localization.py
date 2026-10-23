import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialTagMap, TagAnchor
from fiducial_localization import (
    CameraBodyExtrinsic,
    FiducialUncertaintyModel,
    localization_calibration_sha256,
    map_body_observation_from_fiducials,
)
from fiducial_multitag_pose import MultiTagCameraPose
from localization_contract import (
    LocalizationQuality,
    LocalizationTopology,
    PoseSource,
    ScaleReference,
)


CAMERA_CALIBRATION = "a" * 64


def anchor(tag_id, position, *, method="calibrated", pos_std=(0.002, 0.003, 0.004)):
    return TagAnchor(
        tag_family="tag36h11",
        tag_id=tag_id,
        map_frame_id="site-map",
        position_xyz_m=position,
        orientation_map_from_tag_xyzw=(0.0, 0.0, 0.0, 1.0),
        physical_size_m=0.16,
        position_stddev_m=pos_std,
        orientation_stddev_rad=(0.003, 0.004, 0.005),
        reference_method=method,
    )


def tag_map(*, method="calibrated"):
    return FiducialTagMap(
        "indoor_dense",
        "site-map",
        (
            anchor(1, (0.0, 0.0, 0.0), method=method),
            anchor(
                2,
                (1.0, 0.0, 0.0),
                method=method,
                pos_std=(0.003, 0.004, 0.005),
            ),
            anchor(
                3,
                (0.0, 1.0, 0.0),
                method=method,
                pos_std=(0.004, 0.005, 0.006),
            ),
        ),
    )


def camera_pose(active_map, *, inliers=(1, 2, 3), trans_res=0.01, rot_res=0.02):
    rotation = Rotation.from_euler(
        "xyz", (5.0, -8.0, 17.0), degrees=True
    ).as_quat()
    return MultiTagCameraPose(
        camera_id=2,
        camera_serial="CAM-02",
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        map_frame_id="site-map",
        position_map_from_camera_m=(0.4, -0.2, 1.3),
        orientation_map_from_camera_xyzw=tuple(rotation),
        inlier_tag_keys=tuple(
            ("tag36h11", tag_id) for tag_id in inliers
        ),
        excluded_tag_keys=tuple(
            ("tag36h11", tag_id)
            for tag_id in (1, 2, 3)
            if tag_id not in inliers
        ),
        max_translation_residual_m=trans_res,
        max_orientation_residual_rad=rot_res,
        mean_single_tag_reprojection_rms_px=0.35,
        calibration_sha256=CAMERA_CALIBRATION,
        map_revision_sha256=active_map.revision_sha256,
    )


def extrinsic(**changes):
    values = dict(
        camera_id=2,
        camera_serial="CAM-02",
        camera_frame_id="body-camera",
        body_frame_id="rigid-body",
        translation_body_from_camera_m=(0.12, -0.03, 0.05),
        orientation_body_from_camera_xyzw=tuple(
            Rotation.from_euler(
                "xyz", (1.0, -2.0, 3.0), degrees=True
            ).as_quat()
        ),
        reference_method="calibrated",
    )
    values.update(changes)
    return CameraBodyExtrinsic(**values)


def uncertainty_model(**changes):
    values = dict(
        model_id="fiducial-v1",
        base_position_stddev_m=(0.01, 0.02, 0.03),
        base_orientation_stddev_rad=(0.005, 0.006, 0.007),
        translation_residual_gain=1.5,
        orientation_residual_gain=2.0,
    )
    values.update(changes)
    return FiducialUncertaintyModel(**values)


def rotation_error(first, second):
    relative = first.T @ second
    cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) * 0.5))
    return math.acos(cosine)


class FiducialBodyLocalizationTests(unittest.TestCase):
    def test_composes_map_camera_with_body_extrinsic(self):
        active_map = tag_map()
        pose = camera_pose(active_map)
        ext = extrinsic()
        model = uncertainty_model()

        result = map_body_observation_from_fiducials(
            pose, active_map, ext, model
        )

        rotation_map_from_camera = Rotation.from_quat(
            pose.orientation_map_from_camera_xyzw
        ).as_matrix()
        translation_map_from_camera = np.asarray(
            pose.position_map_from_camera_m
        )
        rotation_body_from_camera = Rotation.from_quat(
            ext.orientation_body_from_camera_xyzw
        ).as_matrix()
        translation_body_from_camera = np.asarray(
            ext.translation_body_from_camera_m
        )
        expected_rotation = (
            rotation_map_from_camera @ rotation_body_from_camera.T
        )
        expected_translation = (
            translation_map_from_camera
            - expected_rotation @ translation_body_from_camera
        )

        actual_rotation = Rotation.from_quat(
            result.orientation_xyzw
        ).as_matrix()
        self.assertLess(
            rotation_error(actual_rotation, expected_rotation), 1e-10
        )
        np.testing.assert_allclose(
            result.position_xyz, expected_translation, atol=1e-12
        )
        self.assertEqual(result.source, PoseSource.FIDUCIAL_LOCALIZATION)
        self.assertEqual(result.topology, LocalizationTopology.INSIDE_OUT)
        self.assertEqual(result.tracked_frame_id, "rigid-body")
        self.assertEqual(result.map_frame_id, "site-map")
        self.assertEqual(result.position_unit, "meters")
        self.assertTrue(result.metric_scale_qualified)
        self.assertEqual(
            result.scale_reference, ScaleReference.CALIBRATED_TAG_MAP
        )
        self.assertEqual(result.quality, LocalizationQuality.TRACKING)
        self.assertEqual(result.inlier_count, 3)
        self.assertAlmostEqual(result.reprojection_rms_px, 0.35)

    def test_uncertainty_model_combines_base_anchor_and_consensus_residuals(self):
        active_map = tag_map()
        pose = camera_pose(active_map, trans_res=0.02, rot_res=0.01)
        model = uncertainty_model()
        result = map_body_observation_from_fiducials(
            pose, active_map, extrinsic(), model
        )

        anchor_position = np.asarray((0.004, 0.005, 0.006))
        expected_position = np.sqrt(
            np.square((0.01, 0.02, 0.03))
            + np.square(anchor_position)
            + (1.5 * 0.02) ** 2
        )
        anchor_orientation = np.asarray((0.003, 0.004, 0.005))
        expected_orientation = np.sqrt(
            np.square((0.005, 0.006, 0.007))
            + np.square(anchor_orientation)
            + (2.0 * 0.01) ** 2
        )
        np.testing.assert_allclose(
            result.position_stddev, expected_position, atol=1e-12
        )
        np.testing.assert_allclose(
            result.orientation_stddev_rad,
            expected_orientation,
            atol=1e-12,
        )

    def test_two_tag_or_high_residual_solution_is_explicitly_degraded(self):
        active_map = tag_map()
        two_tag = map_body_observation_from_fiducials(
            camera_pose(active_map, inliers=(1, 2)),
            active_map,
            extrinsic(),
            uncertainty_model(),
        )
        self.assertEqual(two_tag.quality, LocalizationQuality.DEGRADED)

        high_residual = map_body_observation_from_fiducials(
            camera_pose(active_map, trans_res=0.05),
            active_map,
            extrinsic(),
            uncertainty_model(),
        )
        self.assertEqual(
            high_residual.quality, LocalizationQuality.DEGRADED
        )

    def test_surveyed_anchor_consensus_uses_surveyed_scale_reference(self):
        active_map = tag_map(method="surveyed")
        result = map_body_observation_from_fiducials(
            camera_pose(active_map),
            active_map,
            extrinsic(reference_method="surveyed"),
            uncertainty_model(),
        )
        self.assertEqual(
            result.scale_reference, ScaleReference.SURVEYED_TAGS
        )

    def test_relocalized_flag_and_acquisition_time_are_preserved(self):
        active_map = tag_map()
        result = map_body_observation_from_fiducials(
            camera_pose(active_map),
            active_map,
            extrinsic(),
            uncertainty_model(),
            relocalized=True,
        )
        self.assertTrue(result.relocalized)
        self.assertEqual(result.timestamp_ns, 8_000_000)
        self.assertEqual(result.clock_domain, "capture-ptp-A")

    def test_composed_calibration_revision_is_deterministic_and_sensitive(self):
        ext = extrinsic()
        model = uncertainty_model()
        first = localization_calibration_sha256(
            CAMERA_CALIBRATION, ext, model
        )
        second = localization_calibration_sha256(
            CAMERA_CALIBRATION, ext, model
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            localization_calibration_sha256(
                "b" * 64, ext, model
            ),
        )
        self.assertNotEqual(
            first,
            localization_calibration_sha256(
                CAMERA_CALIBRATION,
                extrinsic(
                    translation_body_from_camera_m=(0.13, -0.03, 0.05)
                ),
                model,
            ),
        )
        self.assertNotEqual(
            first,
            localization_calibration_sha256(
                CAMERA_CALIBRATION,
                ext,
                uncertainty_model(model_id="fiducial-v2"),
            ),
        )

    def test_identity_map_revision_and_inlier_map_mismatch_fail_closed(self):
        active_map = tag_map()
        pose = camera_pose(active_map)
        with self.assertRaisesRegex(ValueError, "identity"):
            map_body_observation_from_fiducials(
                pose,
                active_map,
                extrinsic(camera_id=5),
                uncertainty_model(),
            )
        with self.assertRaisesRegex(ValueError, "map revision"):
            map_body_observation_from_fiducials(
                replace(pose, map_revision_sha256="b" * 64),
                active_map,
                extrinsic(),
                uncertainty_model(),
            )
        with self.assertRaisesRegex(ValueError, "inliers"):
            map_body_observation_from_fiducials(
                replace(
                    pose,
                    inlier_tag_keys=(
                        ("tag36h11", 1),
                        ("tag36h11", 99),
                    ),
                    excluded_tag_keys=(("tag36h11", 2), ("tag36h11", 3)),
                ),
                active_map,
                extrinsic(),
                uncertainty_model(),
            )

    def test_extrinsic_and_uncertainty_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            extrinsic(body_frame_id="body-camera")
        with self.assertRaisesRegex(ValueError, "reference_method"):
            extrinsic(reference_method="declared")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            uncertainty_model(base_position_stddev_m=(-0.1, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite real scalar"):
            uncertainty_model(orientation_residual_gain=float("nan"))

    def test_tracking_policy_thresholds_are_validated(self):
        active_map = tag_map()
        pose = camera_pose(active_map)
        for name, changes in (
            ("tracking_min_tags", {"tracking_min_tags": 1}),
            (
                "max_tracking_translation_residual_m",
                {"max_tracking_translation_residual_m": -1.0},
            ),
            (
                "max_tracking_orientation_residual_rad",
                {"max_tracking_orientation_residual_rad": math.pi + 0.1},
            ),
            ("relocalized", {"relocalized": 1}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                map_body_observation_from_fiducials(
                    pose,
                    active_map,
                    extrinsic(),
                    uncertainty_model(),
                    **changes,
                )


if __name__ == "__main__":
    unittest.main()
