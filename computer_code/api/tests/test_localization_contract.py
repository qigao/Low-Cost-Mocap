import unittest
from dataclasses import replace

from localization_contract import (
    LocalizationQuality,
    LocalizationTopology,
    MapAlignment,
    MapPoseObservation,
    PoseSource,
)


H = "a" * 64


def pose(**changes):
    values = dict(
        source=PoseSource.VISUAL_SLAM,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id="body-camera",
        map_frame_id="local-map-7",
        timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        position_xyz_m=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_stddev_m=(0.02, 0.02, 0.04),
        orientation_stddev_rad=(0.01, 0.01, 0.02),
        map_revision_sha256=H,
        calibration_sha256="b" * 64,
        quality=LocalizationQuality.TRACKING,
        inlier_count=120,
        relocalized=False,
        reprojection_rms_px=0.4,
    )
    values.update(changes)
    return MapPoseObservation(**values)


class MapPoseObservationTests(unittest.TestCase):
    def test_valid_inside_out_slam_pose(self):
        result = pose()
        self.assertEqual(result.source, PoseSource.VISUAL_SLAM)
        self.assertEqual(result.topology, LocalizationTopology.INSIDE_OUT)
        self.assertAlmostEqual(sum(v * v for v in result.orientation_xyzw), 1.0)
        self.assertEqual(result.inlier_count, 120)

    def test_visual_slam_and_outside_in_mocap_require_explicit_topology(self):
        with self.assertRaisesRegex(ValueError, "inside_out topology"):
            pose(topology=LocalizationTopology.OUTSIDE_IN)
        outside = pose(
            source=PoseSource.OUTSIDE_IN_MOCAP,
            topology=LocalizationTopology.OUTSIDE_IN,
            tracked_frame_id="tracked-rigid-body",
        )
        self.assertEqual(outside.topology, LocalizationTopology.OUTSIDE_IN)
        with self.assertRaisesRegex(ValueError, "outside_in topology"):
            pose(source=PoseSource.OUTSIDE_IN_MOCAP)

    def test_fiducial_localization_can_be_used_with_either_camera_topology(self):
        for topology in (LocalizationTopology.INSIDE_OUT, LocalizationTopology.OUTSIDE_IN):
            result = pose(source=PoseSource.FIDUCIAL_LOCALIZATION, topology=topology)
            self.assertEqual(result.topology, topology)

    def test_degraded_pose_is_explicit_and_relocalization_is_reported(self):
        result = pose(quality=LocalizationQuality.DEGRADED, relocalized=True)
        self.assertEqual(result.quality, LocalizationQuality.DEGRADED)
        self.assertTrue(result.relocalized)

    def test_rejects_bad_time_frames_hashes_and_inlier_count(self):
        cases = (
            ({"timestamp_ns": True}, "timestamp_ns"),
            ({"clock_domain": " "}, "clock_domain"),
            ({"tracked_frame_id": ""}, "tracked_frame_id"),
            ({"map_revision_sha256": "Z" * 64}, "lowercase SHA-256"),
            ({"calibration_sha256": "bad"}, "lowercase SHA-256"),
            ({"inlier_count": 0}, "inlier_count"),
            ({"relocalized": 1}, "relocalized"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(pose(), **changes)

    def test_rejects_invalid_pose_and_uncertainty(self):
        cases = (
            ({"position_xyz_m": (0.0, float("nan"), 0.0)}, "finite"),
            ({"orientation_xyzw": (0.0, 0.0, 0.0, 0.0)}, "norm is zero"),
            ({"position_stddev_m": (-0.1, 0.0, 0.0)}, "nonnegative"),
            ({"orientation_stddev_rad": (0.0, float("inf"), 0.0)}, "finite"),
            ({"reprojection_rms_px": -0.1}, "nonnegative"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(pose(), **changes)


class MapAlignmentTests(unittest.TestCase):
    def alignment(self, **changes):
        values = dict(
            source_map_frame_id="slam-local-7",
            target_map_frame_id="site-ENU",
            translation_xyz_m=(10.0, 20.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale_m_per_source_unit=1.0,
            metric_scale_qualified=True,
            scale_reference="rtk_control_points",
            source_map_revision_sha256="a" * 64,
            target_map_revision_sha256="b" * 64,
            alignment_revision_sha256="c" * 64,
        )
        values.update(changes)
        return MapAlignment(**values)

    def test_valid_metric_map_alignment(self):
        result = self.alignment()
        self.assertTrue(result.metric_scale_qualified)
        self.assertEqual(result.target_map_frame_id, "site-ENU")

    def test_monocular_slam_can_be_explicitly_unscaled_until_anchored(self):
        result = self.alignment(
            metric_scale_qualified=False,
            scale_reference="none",
            scale_m_per_source_unit=1.0,
        )
        self.assertFalse(result.metric_scale_qualified)

    def test_rejects_scale_claim_without_independent_reference(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.alignment(metric_scale_qualified=True, scale_reference="none")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.alignment(metric_scale_qualified=False, scale_reference="surveyed_tags")

    def test_rejects_same_frame_invalid_scale_and_bad_digest(self):
        cases = (
            ({"target_map_frame_id": "slam-local-7"}, "must differ"),
            ({"scale_m_per_source_unit": 0.0}, "must be positive"),
            ({"source_map_revision_sha256": "bad"}, "lowercase SHA-256"),
            ({"rotation_xyzw": (0.0, 0.0, 0.0, 0.0)}, "norm is zero"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.alignment(**changes)


if __name__ == "__main__":
    unittest.main()
