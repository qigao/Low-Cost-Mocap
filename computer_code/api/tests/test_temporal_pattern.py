import unittest

import numpy as np

from temporal_pattern import (
    FrameStamp,
    TemporalCodebook,
    LightSample,
    IdentifiedDetection,
    validate_trigger_batch,
    decode_temporal_marker,
    group_decoded_observations,
)


class TriggerSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.frames = [
            FrameStamp(0, 300, 1000000, "hardware-trigger"),
            FrameStamp(1, 300, 1000005, "hardware-trigger"),
            FrameStamp(2, 300, 1000008, "hardware-trigger"),
        ]

    def test_all_cameras_share_trigger_and_common_exposure_clock(self):
        batch = validate_trigger_batch(
            self.frames, [0, 1, 2], max_exposure_skew_ns=10,
            previous_trigger_id=299,
        )
        self.assertEqual(batch.trigger_id, 300)
        self.assertEqual(batch.exposure_skew_ns, 8)
        self.assertEqual(batch.camera_ids, (0, 1, 2))

    def test_rejects_missing_duplicate_or_mixed_trigger_frame(self):
        bad_batches = (
            (self.frames[:2], "missing or extra"),
            ([self.frames[0], self.frames[0], self.frames[2]], "duplicate camera"),
            ([self.frames[0], FrameStamp(1, 301, 1000005, "hardware-trigger"),
              self.frames[2]], "different trigger"),
            ([self.frames[0], FrameStamp(1, 300, 1000005, "local-clock"),
              self.frames[2]], "different clock domains"),
        )
        for frames, error in bad_batches:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    validate_trigger_batch(frames, [0, 1, 2],
                                           max_exposure_skew_ns=10)

    def test_exposure_skew_and_stale_frame_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "timestamp skew"):
            validate_trigger_batch(self.frames, [0, 1, 2],
                                   max_exposure_skew_ns=7)
        with self.assertRaisesRegex(ValueError, "stale or repeated"):
            validate_trigger_batch(self.frames, [0, 1, 2],
                                   max_exposure_skew_ns=10,
                                   previous_trigger_id=300)
        with self.assertRaisesRegex(ValueError, "trigger gap"):
            validate_trigger_batch(self.frames, [0, 1, 2],
                                   max_exposure_skew_ns=10,
                                   previous_trigger_id=298)
        accepted = validate_trigger_batch(
            self.frames, [0, 1, 2], max_exposure_skew_ns=10,
            previous_trigger_id=298, allow_trigger_gaps=True,
        )
        self.assertEqual(accepted.trigger_id, 300)

    def test_rejects_boolean_and_non_integer_metadata(self):
        invalid_frames = (
            FrameStamp(True, 300, 1000000, "hardware-trigger"),
            FrameStamp(0, 300.0, 1000000, "hardware-trigger"),
            FrameStamp(0, 300, True, "hardware-trigger"),
            FrameStamp(0, 300, 1000000, ""),
        )
        for bad in invalid_frames:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_trigger_batch(
                        [bad, self.frames[1], self.frames[2]],
                        [0, 1, 2], max_exposure_skew_ns=10,
                    )


class TemporalIdentityTests(unittest.TestCase):
    def setUp(self):
        # Balanced 8-trigger frequency/time patterns at pairwise distance 4.
        self.patterns = {
            10: (0, 0, 0, 0, 1, 1, 1, 1),
            11: (0, 0, 1, 1, 0, 0, 1, 1),
            12: (0, 1, 0, 1, 0, 1, 0, 1),
            13: (0, 1, 1, 0, 1, 0, 0, 1),
        }
        self.codes = TemporalCodebook(self.patterns, max_bit_errors=1)

    def test_decode_with_occlusion_noise_and_out_of_order_arrivals(self):
        truth = self.patterns[12]
        samples = [
            LightSample(100 + index, None if index in (1, 6)
                        else (1 - bit if index == 0 else bit))
            for index, bit in enumerate(truth)
        ]
        decoded = decode_temporal_marker(
            samples[::-1], self.codes, trigger_epoch=100, min_valid_samples=6
        )
        self.assertEqual(decoded.marker_id, 12)
        self.assertEqual(decoded.valid_samples, 6)
        self.assertEqual(decoded.erased_samples, 2)
        self.assertEqual(decoded.bit_errors, 1)
        self.assertGreaterEqual(decoded.competing_error_margin, 1)

    def test_codes_are_explicitly_phase_anchored_to_trigger_epoch(self):
        truth = self.patterns[11]
        samples = [LightSample(200 + index, bit)
                   for index, bit in enumerate(truth)]
        self.assertEqual(
            decode_temporal_marker(samples, self.codes,
                                   trigger_epoch=200).marker_id, 11
        )
        with self.assertRaisesRegex(ValueError, "different code cycles"):
            decode_temporal_marker(samples, self.codes, trigger_epoch=199)
        with self.assertRaisesRegex(ValueError, "precedes"):
            decode_temporal_marker(samples, self.codes, trigger_epoch=201)

    def test_ambiguous_erasures_do_not_pick_first_marker(self):
        # All four patterns start with zero, so one observed phase cannot
        # identify a physical marker. Never choose the lowest marker ID.
        samples = [LightSample(300, 0)] + [
            LightSample(300 + phase, None) for phase in range(1, 8)
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            decode_temporal_marker(
                samples, self.codes, trigger_epoch=300, min_valid_samples=1
            )

    def test_rejects_mismatched_codebook_and_invalid_samples(self):
        with self.assertRaisesRegex(ValueError, "Hamming"):
            TemporalCodebook({0: [0, 0, 0, 1], 1: [0, 0, 1, 1]},
                             max_bit_errors=1)
        with self.assertRaisesRegex(ValueError, "binary bits"):
            TemporalCodebook({0: [0, 1, 0, 2], 1: [1, 0, 1, 0]})
        samples = [LightSample(400 + i, bit)
                   for i, bit in enumerate(self.patterns[10])]
        for invalid in (
            samples + [samples[0]],
            samples[:-1] + [LightSample(409, 1)],
            samples[:-1] + [LightSample(407, 0.4)],
        ):
            with self.subTest(samples=repr(invalid)):
                with self.assertRaises(ValueError):
                    decode_temporal_marker(
                        invalid, self.codes, trigger_epoch=400
                    )

    def test_insufficient_visibility_is_not_accepted(self):
        samples = [LightSample(500 + i, None if i < 6 else bit)
                   for i, bit in enumerate(self.patterns[10])]
        with self.assertRaisesRegex(ValueError, "insufficient non-erased"):
            decode_temporal_marker(samples, self.codes, trigger_epoch=500)

    def test_two_cameras_identify_same_marker_under_distinct_occlusions(self):
        code = self.patterns[13]
        view0 = [LightSample(600 + i, None if i == 2 else b)
                 for i, b in enumerate(code)]
        view1 = [LightSample(600 + i, None if i in (0, 5) else b)
                 for i, b in enumerate(code)]
        id0 = decode_temporal_marker(view0, self.codes, trigger_epoch=600).marker_id
        id1 = decode_temporal_marker(view1, self.codes, trigger_epoch=600).marker_id
        self.assertEqual((id0, id1), (13, 13))


class CrossCameraIdentityTests(unittest.TestCase):
    def test_group_ids_independently_of_detector_order(self):
        detections = [
            IdentifiedDetection(1, 9, (260, 240)),
            IdentifiedDetection(0, 8, (120, 220)),
            IdentifiedDetection(2, 8, (118, 222)),
            IdentifiedDetection(2, 9, (255, 241)),
            IdentifiedDetection(0, 9, (250, 238)),
            IdentifiedDetection(1, 8, (123, 220)),
            IdentifiedDetection(1, 10, (330, 280)),  # only one camera
        ]
        grouped = group_decoded_observations(
            detections, camera_count=3, allowed_marker_ids=[8, 9, 10]
        )
        self.assertEqual(sorted(grouped.observations_by_marker_id), [8, 9])
        self.assertEqual(grouped.excluded_marker_ids, (10,))
        np.testing.assert_array_equal(grouped.observations_by_marker_id[9][0],
                                      np.array([250, 238]))

    def test_duplicate_marker_in_single_camera_is_rejected(self):
        detections = [
            IdentifiedDetection(0, 8, [120, 220]),
            IdentifiedDetection(0, 8, [122, 222]),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate decoded marker"):
            group_decoded_observations(detections, camera_count=2)

    def test_rejects_invalid_or_unknown_camera_and_marker_id(self):
        for detection in (
            IdentifiedDetection(True, 8, (120, 220)),
            IdentifiedDetection(0, 8.0, (120, 220)),
            IdentifiedDetection(2, 8, (120, 220)),
            IdentifiedDetection(0, 9, (120, 220)),
            IdentifiedDetection(0, 8, ("120", "220")),
        ):
            with self.subTest(detection=repr(detection)):
                with self.assertRaises(ValueError):
                    group_decoded_observations(
                        [detection], camera_count=2, allowed_marker_ids=[8]
                    )


if __name__ == "__main__":
    unittest.main()
