import unittest

import numpy as np

from temporal_pattern import (
    FrameStamp,
    TemporalCodebook,
    LightSample,
    IdentifiedDetection,
    StampedIdentifiedDetection,
    group_synchronized_detections,
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

    def test_pseyepy_software_dequeue_timestamp_without_trigger_is_not_valid(self):
        # Upstream pseyepy exposes host dequeue timestamps but no verified
        # exposure trigger ID: this record must not enter synchronized capture.
        unverified = [
            FrameStamp(0, None, 1_700_000_000_000_000_000, "pseyepy-dequeue"),
            FrameStamp(1, None, 1_700_000_000_000_000_020, "pseyepy-dequeue"),
            FrameStamp(2, None, 1_700_000_000_000_000_030, "pseyepy-dequeue"),
        ]
        with self.assertRaisesRegex(ValueError, "trigger_id"):
            validate_trigger_batch(
                unverified, [0, 1, 2], max_exposure_skew_ns=100
            )

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

    def test_codebook_exposes_guaranteed_erasure_budget_for_track_dropout(self):
        # d_min=4 and one correctable bit error guarantee one additional
        # erased phase. Particular patterns may decode with more erasures,
        # but the caller must rely on runtime ambiguity checks for those cases.
        self.assertEqual(self.codes.minimum_hamming_distance, 4)
        self.assertEqual(self.codes.guaranteed_erasure_budget, 1)

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

    def test_missing_trigger_phases_are_counted_as_erasures(self):
        code = self.patterns[10]
        samples = [
            LightSample(900 + phase, code[phase])
            for phase in (0, 1, 2, 3, 4)
        ]
        decoded = decode_temporal_marker(
            samples, self.codes, trigger_epoch=900, min_valid_samples=5,
        )
        self.assertEqual(decoded.marker_id, 10)
        self.assertEqual(decoded.valid_samples, 5)
        self.assertEqual(decoded.erased_samples, 3)
        self.assertEqual(decoded.bit_errors, 0)

    def test_all_one_flip_and_erasure_masks_never_accept_wrong_marker(self):
        # Exhaust every subset with >=4 observed phases and every allowed
        # single-bit error. Wrong decoded identities are forbidden even
        # if the decoder must reject most heavily erased sequences.
        accepted = 0
        ambiguous = 0
        for marker, code in self.patterns.items():
            for mask in range(1 << self.codes.period):
                phases = [
                    phase for phase in range(self.codes.period)
                    if mask & (1 << phase)
                ]
                if len(phases) < 4:
                    continue
                for flipped in [None] + phases:
                    samples = [
                        LightSample(
                            1100 + phase,
                            1 - code[phase] if phase == flipped else code[phase],
                        )
                        for phase in phases
                    ]
                    try:
                        decoded = decode_temporal_marker(
                            samples, self.codes, trigger_epoch=1100,
                            min_valid_samples=4,
                        )
                    except ValueError:
                        ambiguous += 1
                        continue
                    self.assertEqual(
                        decoded.marker_id, marker,
                        msg=f"wrong identity source={marker} mask={mask} "
                            f"flipped={flipped}",
                    )
                    self.assertEqual(
                        decoded.erased_samples, self.codes.period - len(phases)
                    )
                    accepted += 1
        self.assertGreater(accepted, 0)
        self.assertGreater(ambiguous, 0)

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

    def test_erasure_and_bit_flip_must_not_change_marker_identity(self):
        # With all phases, codes 10/11 are Hamming-distance 4. After
        # erasing distinguishing phases 3, 4, 5, a flip at phase 2 turns
        # the five observed bits of marker 10 into an *exact* match for
        # marker 11. Both codes remain inside the allowed one-bit budget.
        # A legacy best-vs-runner-up margin of 1 silently returned 11.
        true_pattern = self.patterns[10]
        observed_phases = {0, 1, 2, 6, 7}
        samples = [
            LightSample(
                700 + phase,
                (1 - bit if phase == 2 else bit)
                if phase in observed_phases else None,
            )
            for phase, bit in enumerate(true_pattern)
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous temporal"):
            decode_temporal_marker(
                samples, self.codes,
                trigger_epoch=700, min_valid_samples=5,
            )

    def test_unique_code_inside_error_budget_still_decodes(self):
        true_pattern = self.patterns[12]
        samples = [
            LightSample(
                800 + phase,
                None if phase in (1, 6)
                else (1 - bit if phase == 0 else bit),
            )
            for phase, bit in enumerate(true_pattern)
        ]
        result = decode_temporal_marker(
            samples, self.codes, trigger_epoch=800, min_valid_samples=6
        )
        self.assertEqual(result.marker_id, 12)
        self.assertEqual(result.bit_errors, 1)
        self.assertGreaterEqual(result.competing_error_margin, 2)

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

    def test_synchronized_group_retains_noncontiguous_camera_ids(self):
        frames = [
            FrameStamp(2, 900, 5_000_000, "rig-exposure"),
            FrameStamp(5, 900, 5_000_010, "rig-exposure"),
            FrameStamp(9, 900, 5_000_020, "rig-exposure"),
        ]
        observations = [
            StampedIdentifiedDetection(frames[2], 7, (330., 240.)),
            StampedIdentifiedDetection(frames[0], 8, (110., 230.)),
            StampedIdentifiedDetection(frames[1], 7, (325., 238.)),
            StampedIdentifiedDetection(frames[0], 7, (320., 235.)),
            StampedIdentifiedDetection(frames[1], 8, (112., 233.)),
            StampedIdentifiedDetection(frames[2], 99, (400., 250.)),
        ]
        batch, grouped = group_synchronized_detections(
            observations, frames, [2, 5, 9],
            max_exposure_skew_ns=40,
            previous_trigger_id=899, allowed_marker_ids=[7, 8, 99],
        )
        self.assertEqual(batch.trigger_id, 900)
        self.assertEqual(batch.camera_ids, (2, 5, 9))
        self.assertEqual(tuple(grouped.observations_by_marker_id), (7, 8))
        self.assertEqual(grouped.excluded_marker_ids, (99,))
        self.assertIsNone(grouped.observations_by_marker_id[8][2])
        np.testing.assert_array_equal(
            grouped.observations_by_marker_id[7][0],
            np.array([320., 235.]),
        )

    def test_synchronized_group_rejects_borrowed_frame_metadata(self):
        frames = [
            FrameStamp(0, 200, 1_000_000, "rig-exposure"),
            FrameStamp(1, 200, 1_000_004, "rig-exposure"),
        ]
        invalid = [
            FrameStamp(0, 199, 1_000_000, "rig-exposure"),
            FrameStamp(1, 201, 1_000_004, "rig-exposure"),
            FrameStamp(1, 200, 1_000_005, "rig-exposure"),
            FrameStamp(1, 200, 1_000_004, "host-dequeue"),
        ]
        for wrong in invalid:
            with self.subTest(wrong=repr(wrong)):
                detections = [
                    StampedIdentifiedDetection(frames[0], 2, (110., 200.)),
                    StampedIdentifiedDetection(wrong, 2, (115., 203.)),
                ]
                with self.assertRaisesRegex(ValueError, "synchronized trigger batch"):
                    group_synchronized_detections(
                        detections, frames, [0, 1],
                        max_exposure_skew_ns=10,
                    )
        valid = [
            StampedIdentifiedDetection(frames[0], 2, (110., 200.)),
            StampedIdentifiedDetection(frames[1], 2, (115., 203.)),
        ]
        batch, grouped = group_synchronized_detections(
            valid, frames, [0, 1],
            max_exposure_skew_ns=10, allowed_marker_ids=[2],
        )
        self.assertEqual(batch.trigger_id, 200)
        self.assertEqual(tuple(grouped.observations_by_marker_id), (2,))

    def test_synchronized_group_rejects_reused_camera_marker(self):
        frames = [
            FrameStamp(0, 100, 10_000, "rig-exposure"),
            FrameStamp(1, 100, 10_001, "rig-exposure"),
        ]
        repeated = [
            StampedIdentifiedDetection(frames[0], 5, (110., 200.)),
            StampedIdentifiedDetection(frames[0], 5, (111., 200.)),
            StampedIdentifiedDetection(frames[1], 5, (115., 203.)),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate decoded marker"):
            group_synchronized_detections(
                repeated, frames, [0, 1], max_exposure_skew_ns=5,
            )

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
