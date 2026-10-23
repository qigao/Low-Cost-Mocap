import unittest
from dataclasses import replace

from localization_contract import (
    LocalizationQuality,
    LocalizationStatus,
    LocalizationTopology,
    MapAlignment,
    MapPoseObservation,
    PoseSource,
    ScaleReference,
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
        position_xyz=(1.0, 2.0, 3.0),
        position_unit="meters",
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_stddev=(0.02, 0.02, 0.04),
        orientation_stddev_rad=(0.01, 0.01, 0.02),
        map_revision_sha256=H,
        calibration_sha256="b" * 64,
        quality=LocalizationQuality.TRACKING,
        inlier_count=120,
        metric_scale_qualified=True,
        scale_reference=ScaleReference.STEREO_BASELINE,
        relocalized=False,
        reprojection_rms_px=0.4,
    )
    values.update(changes)
    return MapPoseObservation(**values)


def status(**changes):
    values = dict(
        source=PoseSource.VISUAL_SLAM,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id="body-camera",
        map_frame_id="local-map-7",
        timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        map_revision_sha256=H,
        calibration_sha256="b" * 64,
        quality=LocalizationQuality.LOST,
        reason="feature tracking lost",
        last_pose_timestamp_ns=7_900_000,
    )
    values.update(changes)
    return LocalizationStatus(**values)


class MapPoseObservationTests(unittest.TestCase):
    def test_valid_inside_out_slam_pose(self):
        result = pose()
        self.assertEqual(result.source, PoseSource.VISUAL_SLAM)
        self.assertEqual(result.topology, LocalizationTopology.INSIDE_OUT)
        self.assertAlmostEqual(sum(v * v for v in result.orientation_xyzw), 1.0)
        self.assertEqual(result.inlier_count, 120)
        self.assertTrue(result.metric_scale_qualified)
        self.assertEqual(result.scale_reference, ScaleReference.STEREO_BASELINE)

    def test_visual_slam_and_outside_in_mocap_require_explicit_topology(self):
        with self.assertRaisesRegex(ValueError, "inside_out topology"):
            pose(topology=LocalizationTopology.OUTSIDE_IN)
        outside = pose(
            source=PoseSource.OUTSIDE_IN_MOCAP,
            topology=LocalizationTopology.OUTSIDE_IN,
            tracked_frame_id="tracked-rigid-body",
            scale_reference=ScaleReference.CALIBRATED_CAMERA_GEOMETRY,
        )
        self.assertEqual(outside.topology, LocalizationTopology.OUTSIDE_IN)
        with self.assertRaisesRegex(ValueError, "outside_in topology"):
            pose(source=PoseSource.OUTSIDE_IN_MOCAP)

    def test_fiducial_localization_can_be_used_with_either_camera_topology(self):
        for topology in (LocalizationTopology.INSIDE_OUT, LocalizationTopology.OUTSIDE_IN):
            result = pose(
                source=PoseSource.FIDUCIAL_LOCALIZATION,
                topology=topology,
                scale_reference=ScaleReference.SURVEYED_TAGS,
            )
            self.assertEqual(result.topology, topology)

    def test_slam_local_map_units_are_explicitly_unscaled(self):
        result = pose(
            position_unit="map_units",
            position_xyz=(120.0, -3.0, 7.0),
            metric_scale_qualified=False,
            scale_reference=ScaleReference.NONE,
        )
        self.assertEqual(result.position_unit, "map_units")
        self.assertFalse(result.metric_scale_qualified)
        with self.assertRaisesRegex(ValueError, "position_unit must be"):
            pose(position_unit="centimeters")

    def test_rejects_unqualified_meter_claim_and_qualified_map_units(self):
        with self.assertRaisesRegex(ValueError, "meters require"):
            pose(
                metric_scale_qualified=False,
                scale_reference=ScaleReference.NONE,
            )
        with self.assertRaisesRegex(ValueError, "map_units cannot claim"):
            pose(
                position_unit="map_units",
                metric_scale_qualified=True,
                scale_reference=ScaleReference.STEREO_BASELINE,
            )
        with self.assertRaisesRegex(ValueError, "qualification and scale reference disagree"):
            pose(
                metric_scale_qualified=True,
                scale_reference=ScaleReference.NONE,
            )

    def test_degraded_pose_is_explicit_and_relocalization_is_reported(self):
        result = pose(quality=LocalizationQuality.DEGRADED, relocalized=True)
        self.assertEqual(result.quality, LocalizationQuality.DEGRADED)
        self.assertTrue(result.relocalized)

    def test_lost_and_invalid_states_cannot_carry_a_pose(self):
        for quality in (LocalizationQuality.LOST, LocalizationQuality.INVALID):
            with self.subTest(quality=quality), self.assertRaisesRegex(
                ValueError, "cannot carry a pose"
            ):
                pose(quality=quality)

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
            ({"position_xyz": (0.0, float("nan"), 0.0)}, "finite"),
            ({"orientation_xyzw": (0.0, 0.0, 0.0, 0.0)}, "norm is zero"),
            ({"position_stddev": (-0.1, 0.0, 0.0)}, "nonnegative"),
            ({"orientation_stddev_rad": (0.0, float("inf"), 0.0)}, "finite"),
            ({"reprojection_rms_px": -0.1}, "nonnegative"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(pose(), **changes)


class LocalizationStatusTests(unittest.TestCase):
    def test_lost_state_is_pose_free_and_keeps_last_pose_time_for_diagnostics(self):
        result = status()
        self.assertEqual(result.quality, LocalizationQuality.LOST)
        self.assertEqual(result.last_pose_timestamp_ns, 7_900_000)
        self.assertEqual(result.reason, "feature tracking lost")

    def test_invalid_state_is_supported_without_fabricating_geometry(self):
        result = status(
            quality=LocalizationQuality.INVALID,
            reason="calibration revision mismatch",
            last_pose_timestamp_ns=None,
        )
        self.assertEqual(result.quality, LocalizationQuality.INVALID)
        self.assertIsNone(result.last_pose_timestamp_ns)
        self.assertFalse(hasattr(result, "position_xyz"))

    def test_status_rejects_future_last_pose_timestamp_and_bad_topology(self):
        with self.assertRaisesRegex(ValueError, "cannot be newer"):
            status(last_pose_timestamp_ns=8_000_001)
        with self.assertRaisesRegex(ValueError, "inside_out topology"):
            status(topology=LocalizationTopology.OUTSIDE_IN)


class MapAlignmentTests(unittest.TestCase):
    def alignment(self, **changes):
        values = dict(
            source_map_frame_id="slam-local-7",
            target_map_frame_id="site-ENU",
            translation_xyz_m=(10.0, 20.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale_m_per_source_unit=0.0125,
            metric_scale_qualified=True,
            scale_reference=ScaleReference.RTK_CONTROL_POINTS,
            source_map_revision_sha256="a" * 64,
            target_map_revision_sha256="b" * 64,
            alignment_revision_sha256="c" * 64,
        )
        values.update(changes)
        return MapAlignment(**values)

    def test_valid_metric_map_alignment(self):
        result = self.alignment()
        self.assertTrue(result.metric_scale_qualified)
        self.assertAlmostEqual(result.scale_m_per_source_unit, 0.0125)
        self.assertEqual(result.target_map_frame_id, "site-ENU")

    def test_monocular_slam_unknown_scale_has_no_numeric_placeholder(self):
        result = self.alignment(
            metric_scale_qualified=False,
            scale_reference=ScaleReference.NONE,
            scale_m_per_source_unit=None,
        )
        self.assertFalse(result.metric_scale_qualified)
        self.assertIsNone(result.scale_m_per_source_unit)

    def test_rejects_fake_scale_or_reference_mismatch(self):
        with self.assertRaisesRegex(ValueError, "unqualified map scale must be None"):
            self.alignment(
                metric_scale_qualified=False,
                scale_reference=ScaleReference.NONE,
                scale_m_per_source_unit=1.0,
            )
        with self.assertRaisesRegex(ValueError, "qualification and scale reference disagree"):
            self.alignment(metric_scale_qualified=True, scale_reference=ScaleReference.NONE)
        with self.assertRaisesRegex(ValueError, "qualification and scale reference disagree"):
            self.alignment(
                metric_scale_qualified=False,
                scale_reference=ScaleReference.SURVEYED_TAGS,
                scale_m_per_source_unit=None,
            )
        with self.assertRaisesRegex(ValueError, "requires scale_m_per_source_unit"):
            self.alignment(scale_m_per_source_unit=None)

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
