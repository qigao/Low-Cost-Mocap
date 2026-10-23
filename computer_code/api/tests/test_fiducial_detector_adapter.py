import unittest
from dataclasses import replace

from fiducial_detector_adapter import (
    DetectorFrameContext,
    FiducialDetectionBatch,
    RawFiducialDetection,
    normalize_detector_output,
)


CALIBRATION = "a" * 64
MAP_REVISION = "b" * 64


def context(**changes):
    values = dict(
        camera_id=2,
        camera_serial="CAM-02",
        frame_id=900,
        trigger_id=42,
        exposure_timestamp_ns=8_000_000,
        clock_domain="capture-ptp-A",
        calibration_sha256=CALIBRATION,
        map_revision_sha256=MAP_REVISION,
    )
    values.update(changes)
    return DetectorFrameContext(**values)


def detection(tag_id=1, **changes):
    values = dict(
        tag_family="tag36h11",
        tag_id=tag_id,
        image_corners_px=(
            (100.0, 100.0),
            (200.0, 100.0),
            (200.0, 200.0),
            (100.0, 200.0),
        ),
        corrected_bit_errors=0,
        corner_rms_px=0.25,
    )
    values.update(changes)
    return RawFiducialDetection(**values)


class FakeAdapter:
    backend_name = "fake-apriltag"
    backend_revision = "1.2.3"

    def __init__(self, results):
        self.results = results
        self.images = []

    def detect(self, image):
        self.images.append(image)
        return self.results


class FailingAdapter(FakeAdapter):
    def detect(self, image):
        raise RuntimeError("backend failed")


class FiducialDetectorAdapterTests(unittest.TestCase):
    def test_normalizes_detector_data_with_host_owned_frame_metadata(self):
        adapter = FakeAdapter([detection(2), detection(1)])
        image = object()
        batch = normalize_detector_output(
            adapter,
            image,
            context(),
            allowed_families=("tag36h11",),
        )

        self.assertEqual(adapter.images, [image])
        self.assertEqual(batch.backend_name, "fake-apriltag")
        self.assertEqual(batch.backend_revision, "1.2.3")
        self.assertEqual(
            [(obs.tag_family, obs.tag_id) for obs in batch.observations],
            [("tag36h11", 1), ("tag36h11", 2)],
        )
        for obs in batch.observations:
            self.assertEqual(obs.camera_id, 2)
            self.assertEqual(obs.camera_serial, "CAM-02")
            self.assertEqual(obs.frame_id, 900)
            self.assertEqual(obs.trigger_id, 42)
            self.assertEqual(obs.exposure_timestamp_ns, 8_000_000)
            self.assertEqual(obs.clock_domain, "capture-ptp-A")
            self.assertEqual(obs.calibration_sha256, CALIBRATION)
            self.assertEqual(obs.map_revision_sha256, MAP_REVISION)

    def test_zero_detections_is_a_valid_current_frame_result(self):
        batch = normalize_detector_output(
            FakeAdapter([]),
            object(),
            context(),
        )
        self.assertEqual(batch.observations, ())

    def test_rejects_duplicate_tag_identity_in_one_detector_frame(self):
        with self.assertRaisesRegex(ValueError, "duplicate tag family/ID"):
            normalize_detector_output(
                FakeAdapter([detection(1), detection(1)]),
                object(),
                context(),
            )

    def test_rejects_family_outside_active_profile(self):
        with self.assertRaisesRegex(ValueError, "outside the active profile"):
            normalize_detector_output(
                FakeAdapter([detection(tag_family="aruco_4x4_50")]),
                object(),
                context(),
                allowed_families=("tag36h11",),
            )

    def test_fiducial_contract_remains_authoritative_for_corner_geometry(self):
        bad = detection(
            image_corners_px=(
                (100.0, 100.0),
                (200.0, 200.0),
                (200.0, 100.0),
                (100.0, 200.0),
            )
        )
        with self.assertRaisesRegex(ValueError, "convex ordered"):
            normalize_detector_output(FakeAdapter([bad]), object(), context())

    def test_raw_detection_rejects_invalid_identity_and_quality(self):
        cases = (
            ({"tag_id": -1}, "tag_id"),
            ({"corrected_bit_errors": -1}, "corrected_bit_errors"),
            ({"corner_rms_px": -0.1}, "corner_rms_px"),
            ({"corner_rms_px": float("nan")}, "corner_rms_px"),
            ({"tag_family": " "}, "tag_family"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                detection(**changes)

    def test_context_rejects_invalid_capture_metadata(self):
        cases = (
            ({"camera_id": True}, "camera_id"),
            ({"frame_id": -1}, "frame_id"),
            ({"clock_domain": " "}, "clock_domain"),
            ({"calibration_sha256": "bad"}, "lowercase SHA-256"),
            ({"map_revision_sha256": "Z" * 64}, "lowercase SHA-256"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                context(**changes)

    def test_backend_contract_rejects_missing_metadata_or_wrong_output_type(self):
        class MissingMetadata:
            def detect(self, image):
                return []

        with self.assertRaisesRegex(ValueError, "backend_name"):
            normalize_detector_output(MissingMetadata(), object(), context())

        adapter = FakeAdapter((item for item in [detection()]))
        with self.assertRaisesRegex(ValueError, "output must be a sequence"):
            normalize_detector_output(adapter, object(), context())

        adapter = FakeAdapter([object()])
        with self.assertRaisesRegex(ValueError, "RawFiducialDetection"):
            normalize_detector_output(adapter, object(), context())

    def test_backend_exception_propagates_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            normalize_detector_output(
                FailingAdapter([]),
                object(),
                context(),
            )

    def test_batch_rejects_observation_metadata_not_owned_by_context(self):
        batch = normalize_detector_output(
            FakeAdapter([detection()]),
            object(),
            context(),
        )
        mismatched = replace(batch.observations[0], frame_id=899)
        with self.assertRaisesRegex(ValueError, "does not match detector frame context"):
            FiducialDetectionBatch(
                backend_name=batch.backend_name,
                backend_revision=batch.backend_revision,
                context=batch.context,
                observations=(mismatched,),
            )


if __name__ == "__main__":
    unittest.main()
