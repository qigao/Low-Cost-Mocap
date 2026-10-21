import unittest

import numpy as np

from light_intensity import (
    sample_light_roi,
    BinaryIntensityCalibration,
    quantize_roi_track,
)
from temporal_pattern import TemporalCodebook, decode_temporal_marker


class RawLightEvidenceTests(unittest.TestCase):
    def make_image(self, ambient, strength=0.0, center=(24, 24)):
        image = np.full((48, 48), ambient, dtype=np.float64)
        cx, cy = center
        image[cy-2:cy+3, cx-2:cx+3] += strength
        return image

    def test_background_subtraction_is_invariant_to_ambient_light(self):
        contrasts = []
        for background in (5.0, 40.0, 78.0):
            with self.subTest(background=background):
                roi = sample_light_roi(self.make_image(background, 120.0), (24, 24))
                contrasts.append(roi.contrast)
                self.assertGreater(roi.foreground_pixels, 1)
                self.assertGreater(roi.background_pixels, roi.foreground_pixels)
        np.testing.assert_allclose(contrasts, [120., 120., 120.], atol=1e-12)

    def test_calibrated_dead_zone_yields_explicit_erasure(self):
        calibration = BinaryIntensityCalibration(
            dark_contrast=0.0, illuminated_contrast=120.0
        )
        self.assertEqual(calibration.classify(0.0), 0)
        self.assertEqual(calibration.classify(120.0), 1)
        self.assertIsNone(calibration.classify(60.0))
        self.assertIsNone(calibration.classify(None))

    def test_raw_frames_through_quantization_to_temporal_identity(self):
        patterns = {
            0: (0, 0, 0, 0, 1, 1, 1, 1),
            1: (0, 0, 1, 1, 0, 0, 1, 1),
            2: (0, 1, 0, 1, 0, 1, 0, 1),
            3: (0, 1, 1, 0, 1, 0, 0, 1),
        }
        codes = TemporalCodebook(patterns, max_bit_errors=1)
        measured = []
        for phase, bit in enumerate(patterns[3]):
            # Mid-level ROI produces an erasure, not a guessed bit.
            strength = 60.0 if phase == 3 else (
                120.0 if (1 - bit if phase == 5 else bit) else 0.0
            )
            image = self.make_image(20.0 + 12.0 * (phase % 3), strength)
            measured.append((500 + phase, sample_light_roi(image, (24, 24))))
        calibration = BinaryIntensityCalibration(0.0, 120.0)
        lights = quantize_roi_track(measured, calibration)
        self.assertIsNone(lights[3].illuminated)
        self.assertEqual(lights[5].illuminated, 1)
        result = decode_temporal_marker(
            lights, codes, trigger_epoch=500, min_valid_samples=7
        )
        self.assertEqual(result.marker_id, 3)
        self.assertEqual(result.bit_errors, 1)
        self.assertEqual(result.erased_samples, 1)

    def test_roi_and_calibration_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            sample_light_roi(self.make_image(10.0), (2, 2))
        with self.assertRaisesRegex(ValueError, "grayscale"):
            sample_light_roi(np.zeros((48, 48, 3)), (24, 24))
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = self.make_image(10.0)
            bad[24, 24] = np.nan
            sample_light_roi(bad, (24, 24))
        with self.assertRaisesRegex(ValueError, "radius"):
            sample_light_roi(self.make_image(10.0), (24, 24),
                             foreground_radius_px=6, background_radius_px=6)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            BinaryIntensityCalibration(0.0, 3.0, minimum_contrast_separation=10.0)
        with self.assertRaisesRegex(ValueError, "dead_zone_fraction"):
            BinaryIntensityCalibration(0.0, 100.0, dead_zone_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            BinaryIntensityCalibration(0.0, 100.0).classify(np.inf)
        with self.assertRaisesRegex(ValueError, "ROI evidence"):
            quantize_roi_track([(0, 10.0)], BinaryIntensityCalibration(0.0, 100.0))

    def test_absence_is_not_converted_into_off_bit(self):
        calibration = BinaryIntensityCalibration(0.0, 120.0)
        samples = quantize_roi_track(
            [(11, None), (12, sample_light_roi(self.make_image(8.0), (24, 24)))],
            calibration,
        )
        self.assertIsNone(samples[0].illuminated)
        self.assertEqual(samples[1].illuminated, 0)


if __name__ == "__main__":
    unittest.main()
