import unittest
from dataclasses import replace

from fiducial_contract import (
    FiducialObservation,
    FiducialTagMap,
    TagAnchor,
    validate_fiducial_observation,
)


CALIBRATION = "a" * 64


def anchor(tag_id=1, *, position=(0.0, 0.0, 0.0), method="calibrated", frame="indoor-map"):
    return TagAnchor(
        tag_family="tag36h11",
        tag_id=tag_id,
        map_frame_id=frame,
        position_xyz_m=position,
        orientation_map_from_tag_xyzw=(0.0, 0.0, 0.0, 1.0),
        physical_size_m=0.12,
        position_stddev_m=(0.001, 0.001, 0.002),
        orientation_stddev_rad=(0.002, 0.002, 0.003),
        reference_method=method,
    )


def observation(tag_map, **changes):
    values = dict(
        tag_family="tag36h11",
        tag_id=1,
        camera_id=2,
        camera_serial="CAM-02",
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        image_corners_px=((100.0, 100.0), (200.0, 100.0),
                          (200.0, 200.0), (100.0, 200.0)),
        corrected_bit_errors=0,
        corner_rms_px=0.25,
        calibration_sha256=CALIBRATION,
        map_revision_sha256=tag_map.revision_sha256,
    )
    values.update(changes)
    return FiducialObservation(**values)


def validate(obs, tag_map):
    return validate_fiducial_observation(
        obs, tag_map,
        expected_camera_id=2,
        expected_camera_serial="CAM-02",
        expected_frame_id=900,
        expected_trigger_id=42,
        expected_exposure_timestamp_ns=8_000_000,
        expected_clock_domain="capture-ptp-A",
        expected_calibration_sha256=CALIBRATION,
        max_corrected_bit_errors=0,
        max_corner_rms_px=1.0,
    )


class FiducialMapContractTests(unittest.TestCase):
    def setUp(self):
        self.indoor = FiducialTagMap(
            profile="indoor_dense",
            map_frame_id="indoor-map",
            tags=(anchor(1), anchor(2, position=(1.0, 0.0, 0.0))),
        )

    def test_valid_indoor_map_and_frame_bound_observation(self):
        obs = observation(self.indoor)
        self.assertEqual(validate(obs, self.indoor), obs)
        self.assertEqual(self.indoor.by_key[("tag36h11", 1)].physical_size_m, 0.12)
        self.assertEqual(len(self.indoor.revision_sha256), 64)

    def test_outdoor_anchor_map_requires_surveyed_reference_poses(self):
        with self.assertRaisesRegex(ValueError, "surveyed reference poses"):
            FiducialTagMap(
                profile="outdoor_sparse_anchors",
                map_frame_id="site-ENU",
                tags=(anchor(1, frame="site-ENU", method="calibrated"),),
            )
        outdoor = FiducialTagMap(
            profile="outdoor_sparse_anchors",
            map_frame_id="site-ENU",
            tags=(anchor(1, frame="site-ENU", method="surveyed"),),
        )
        self.assertEqual(outdoor.profile, "outdoor_sparse_anchors")

    def test_map_digest_is_order_independent_and_geometry_sensitive(self):
        reversed_map = FiducialTagMap(
            profile="indoor_dense", map_frame_id="indoor-map",
            tags=tuple(reversed(self.indoor.tags)),
        )
        self.assertEqual(self.indoor.revision_sha256, reversed_map.revision_sha256)
        changed = FiducialTagMap(
            profile="indoor_dense", map_frame_id="indoor-map",
            tags=(anchor(1, position=(0.01, 0.0, 0.0)), self.indoor.tags[1]),
        )
        self.assertNotEqual(self.indoor.revision_sha256, changed.revision_sha256)

    def test_map_rejects_duplicate_ids_empty_tags_and_mixed_frames(self):
        cases = (
            ((), "at least one tag"),
            ((anchor(1), anchor(1)), "must be unique"),
            ((anchor(1), anchor(2, frame="another-frame")), "coordinate frame"),
        )
        for tags, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                FiducialTagMap("indoor_dense", "indoor-map", tags)

    def test_map_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "profile must be"):
            FiducialTagMap("outdoor_guess", "site-ENU", (anchor(1, frame="site-ENU"),))

    def test_anchor_rejects_bad_size_quaternion_and_uncertainty(self):
        base = anchor()
        for changes, message in (
            ({"physical_size_m": 0.0}, "physical_size_m"),
            ({"orientation_map_from_tag_xyzw": (0.0, 0.0, 0.0, 0.0)}, "norm is zero"),
            ({"position_stddev_m": (-1.0, 0.0, 0.0)}, "nonnegative"),
            ({"orientation_stddev_rad": (0.0, float("nan"), 0.0)}, "finite"),
            ({"reference_method": "declared"}, "reference_method"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(base, **changes)


class FiducialObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.tag_map = FiducialTagMap(
            "indoor_dense", "indoor-map",
            (anchor(1), anchor(2, position=(1.0, 0.0, 0.0))),
        )
        self.obs = observation(self.tag_map)

    def test_rejects_camera_frame_trigger_timestamp_clock_and_calibration_mismatch(self):
        cases = (
            ({"camera_id": 5}, "camera ID mismatch"),
            ({"camera_serial": "OTHER"}, "camera serial mismatch"),
            ({"frame_id": 899}, "frame ID mismatch"),
            ({"trigger_id": 41}, "trigger ID mismatch"),
            ({"exposure_timestamp_ns": 7_999_999}, "exposure timestamp mismatch"),
            ({"clock_domain": "host-clock"}, "clock domain mismatch"),
            ({"calibration_sha256": "b" * 64}, "calibration revision mismatch"),
            ({"map_revision_sha256": "c" * 64}, "tag map revision mismatch"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate(replace(self.obs, **changes), self.tag_map)

    def test_rejects_unknown_tag_and_low_quality_detection(self):
        with self.assertRaisesRegex(ValueError, "not present in the active map"):
            validate(replace(self.obs, tag_id=99), self.tag_map)
        with self.assertRaisesRegex(ValueError, "corrected too many"):
            validate(replace(self.obs, corrected_bit_errors=1), self.tag_map)
        with self.assertRaisesRegex(ValueError, "corner RMS"):
            validate(replace(self.obs, corner_rms_px=1.01), self.tag_map)

    def test_rejects_nonconvex_duplicate_small_or_nonfinite_corners(self):
        cases = (
            (((100, 100), (200, 200), (200, 100), (100, 200)), "convex ordered"),
            (((100, 100), (100, 100), (200, 200), (100, 200)), "distinct corners"),
            (((1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5)), "area is too small"),
            (((100, 100), (200, 100), (200, float("inf")), (100, 200)), "finite"),
        )
        for corners, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replace(self.obs, image_corners_px=corners)

    def test_observation_rejects_invalid_digest_and_negative_quality(self):
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            replace(self.obs, calibration_sha256="Z" * 64)
        with self.assertRaisesRegex(ValueError, "corner_rms_px must be nonnegative"):
            replace(self.obs, corner_rms_px=-0.1)
        with self.assertRaisesRegex(ValueError, "corrected_bit_errors"):
            replace(self.obs, corrected_bit_errors=-1)


if __name__ == "__main__":
    unittest.main()
