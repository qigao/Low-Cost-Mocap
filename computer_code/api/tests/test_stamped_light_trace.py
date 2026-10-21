"""Strict per-track, per-exposure provenance before temporal ID decoding."""
import unittest

from light_intensity import (
    BinaryIntensityCalibration, ROIContrast,
    StampedTrackROI, quantize_stamped_track_trace,
)
from light_tracking import TrackEvidence
from temporal_pattern import FrameStamp, TemporalCodebook, decode_temporal_marker


class StampedTrackTraceTests(unittest.TestCase):
    def setUp(self):
        self.calibration = BinaryIntensityCalibration(0.0, 120.0)
        self.codes = TemporalCodebook({
            10: (0, 0, 0, 0, 1, 1, 1, 1),
            11: (0, 0, 1, 1, 0, 0, 1, 1),
            12: (0, 1, 0, 1, 0, 1, 0, 1),
            13: (0, 1, 1, 0, 1, 0, 0, 1),
        }, max_bit_errors=1)
        self.epoch = 1000
        self.camera_id = 2
        self.track_id = 5
        self.bits = self.codes.patterns[13]
        self.trace = [
            self.make_entry(i, bit)
            for i, bit in enumerate(self.bits)
        ]

    def make_entry(self, phase, bit, *, qualified=True):
        trigger = self.epoch + phase
        stamp = FrameStamp(2, trigger, 1_000_000 + phase * 1000, "rig-hw-exposure")
        evidence = TrackEvidence(
            self.track_id, self.camera_id, trigger,
            (25.0, 25.0),
            (25.0, 25.0) if bit else None,
            0 if bit else 1,
        )
        roi = ROIContrast(
            120.0 if bit else 0.0,
            0.0,
            120.0 if bit else 0.0,
            10, 30,
        )
        return StampedTrackROI(
            stamp, evidence, roi,
            predicted_roi_qualified=qualified and not bool(bit),
        )

    def quantize(self, samples):
        return quantize_stamped_track_trace(
            samples, self.calibration,
            camera_id=self.camera_id, track_id=self.track_id,
        )

    def test_same_track_and_frame_trigger_decodes_full_code(self):
        light_samples = self.quantize(self.trace)
        self.assertEqual(
            [x.illuminated for x in light_samples], list(self.bits)
        )
        decoded = decode_temporal_marker(
            light_samples, self.codes, trigger_epoch=self.epoch
        )
        self.assertEqual(decoded.marker_id, 13)
        self.assertEqual(decoded.erased_samples, 0)

    def test_unqualified_predicted_dark_roi_fails_closed(self):
        bad = list(self.trace)
        bad[0] = self.make_entry(0, 0, qualified=False)
        with self.assertRaisesRegex(ValueError, "independent motion qualification"):
            self.quantize(bad)

    def test_missing_roi_is_erasure_not_known_dark_phase(self):
        altered = list(self.trace)
        first = altered[0]
        altered[0] = StampedTrackROI(
            first.frame_stamp, first.track_evidence, None
        )
        samples = self.quantize(altered)
        self.assertIsNone(samples[0].illuminated)
        decoded = decode_temporal_marker(
            samples, self.codes, trigger_epoch=self.epoch
        )
        self.assertEqual(decoded.marker_id, 13)
        self.assertEqual(decoded.erased_samples, 1)

    def test_rejects_track_identity_switch_or_camera_mixing(self):
        first = self.trace[2]
        for changed in (
            TrackEvidence(6, 2, self.epoch + 2, (25., 25.), (25., 25.), 0),
            TrackEvidence(5, 1, self.epoch + 2, (25., 25.), (25., 25.), 0),
        ):
            with self.subTest(changed=repr(changed)):
                mixed = list(self.trace)
                mixed[2] = StampedTrackROI(
                    first.frame_stamp, changed, first.roi
                )
                with self.assertRaisesRegex(ValueError, "mixed camera or local track"):
                    self.quantize(mixed)

    def test_rejects_borrowed_trigger_or_swapped_exposure_stamp(self):
        first = self.trace[2]
        for stamp in (
            FrameStamp(2, self.epoch + 3, 1_002_000, "rig-hw-exposure"),
            FrameStamp(1, self.epoch + 2, 1_002_000, "rig-hw-exposure"),
        ):
            with self.subTest(stamp=stamp):
                bad = list(self.trace)
                bad[2] = StampedTrackROI(
                    stamp, first.track_evidence, first.roi
                )
                with self.assertRaises(ValueError):
                    self.quantize(bad)

    def test_rejects_repeated_out_of_order_clock_or_over_capacity(self):
        with self.assertRaisesRegex(ValueError, "stale or repeated"):
            self.quantize(self.trace[:3] + [self.trace[2]])
        with self.assertRaisesRegex(ValueError, "stale or repeated"):
            self.quantize(self.trace[::-1])
        changed = list(self.trace)
        first = changed[3]
        changed[3] = StampedTrackROI(
            FrameStamp(2, self.epoch + 3, 1_001_000, "rig-hw-exposure"),
            first.track_evidence, first.roi,
        )
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            self.quantize(changed)
        changed = list(self.trace)
        first = changed[3]
        changed[3] = StampedTrackROI(
            FrameStamp(2, self.epoch + 3, 1_003_000, "other-clock"),
            first.track_evidence, first.roi,
        )
        with self.assertRaisesRegex(ValueError, "mixes clock domains"):
            self.quantize(changed)
        with self.assertRaisesRegex(ValueError, "capacity"):
            quantize_stamped_track_trace(
                self.trace, self.calibration,
                camera_id=2, track_id=5, max_samples=4,
            )


if __name__ == "__main__":
    unittest.main()
