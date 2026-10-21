"""Offline measurements must expose rejection and statistical uncertainty."""
import unittest

from offline_capture_acceptance import (
    LabeledTemporalTrial,
    RecordedFrameBatch,
    replay_labeled_temporal_trials,
    replay_trigger_batches,
)
from temporal_pattern import FrameStamp, LightSample, TemporalCodebook


class TriggerReplayTests(unittest.TestCase):
    def frames(self, trigger, *, offset=0):
        return (
            FrameStamp(2, trigger, 1_000_000 + trigger * 100 + offset, "rig-clock"),
            FrameStamp(5, trigger, 1_000_006 + trigger * 100 + offset, "rig-clock"),
        )

    def test_reference_mismatch_duplicate_and_skew_are_reported_not_hidden(self):
        recordings = (
            RecordedFrameBatch(10, self.frames(10)),
            RecordedFrameBatch(11, self.frames(10)),  # wrong independent ref
            RecordedFrameBatch(11, self.frames(11)),  # duplicate reference
            RecordedFrameBatch(12, self.frames(12)),
            RecordedFrameBatch(14, self.frames(14)),  # real capture gap
            RecordedFrameBatch(15, (
                FrameStamp(2, 15, 10_000, "rig-clock"),
                FrameStamp(5, 15, 10_100, "rig-clock"),
            )),
        )
        report = replay_trigger_batches(
            recordings, expected_camera_ids=[2, 5],
            max_exposure_skew_ns=10,
        )
        self.assertEqual(report.total_batches, 6)
        self.assertEqual(report.admitted_batches, 2)
        self.assertEqual(report.rejected_batches, 4)
        self.assertEqual(report.reference_gaps, 1)
        self.assertEqual(report.maximum_declared_skew_ns, 6)
        self.assertEqual(
            report.rejection_reasons["independent_reference_trigger_mismatch"], 1
        )
        self.assertEqual(report.rejection_reasons["reference_not_monotonic"], 1)
        self.assertEqual(report.rejection_reasons["reference_trigger_gap"], 1)
        self.assertFalse(report.hardware_provenance_verified)

    def test_invalid_reference_camera_configuration_fails_before_replay(self):
        recording = (RecordedFrameBatch(10, self.frames(10)),)
        for camera_ids in ([], [2, 2], [True, 5]):
            with self.subTest(camera_ids=camera_ids):
                with self.assertRaises(ValueError):
                    replay_trigger_batches(
                        recording, expected_camera_ids=camera_ids,
                        max_exposure_skew_ns=10,
                    )

    def test_ref_gap_can_be_reported_and_admitted_when_explicitly_allowed(self):
        recordings = (
            RecordedFrameBatch(3, self.frames(3)),
            RecordedFrameBatch(5, self.frames(5)),
        )
        report = replay_trigger_batches(
            recordings, expected_camera_ids=[2, 5],
            max_exposure_skew_ns=10, allow_reference_gaps=True,
        )
        self.assertEqual(report.admitted_batches, 2)
        self.assertEqual(report.reference_gaps, 1)
        self.assertFalse(report.hardware_provenance_verified)


class TemporalReplayTests(unittest.TestCase):
    def setUp(self):
        self.codebook = TemporalCodebook({
            10: (0, 0, 0, 0, 1, 1, 1, 1),
            11: (0, 0, 1, 1, 0, 0, 1, 1),
            12: (0, 1, 0, 1, 0, 1, 0, 1),
            13: (0, 1, 1, 0, 1, 0, 0, 1),
        }, max_bit_errors=1)

    def trial(self, marker=10, flips=(), erased=(), *, epoch=1000):
        samples = tuple(
            LightSample(
                epoch + phase,
                None if phase in erased
                else (1 - value if phase in flips else value),
            )
            for phase, value in enumerate(self.codebook.patterns[marker])
        )
        return LabeledTemporalTrial(marker, epoch, samples)

    def test_zero_wrong_accepts_still_have_nonzero_uncertainty(self):
        report = replay_labeled_temporal_trials(
            [self.trial(marker=10 + k % 4) for k in range(100)],
            self.codebook,
        )
        self.assertEqual(report.correct_accepts, 100)
        self.assertEqual(report.wrong_accepts, 0)
        self.assertEqual(report.rejected, 0)
        self.assertGreater(report.wrong_accept_one_sided_upper_95, 0.01)
        self.assertLess(report.wrong_accept_one_sided_upper_95, 0.04)
        self.assertFalse(report.hardware_provenance_verified)

    def test_outside_declared_one_flip_budget_can_be_wrongly_accepted(self):
        # Model marker 10, but three corruptions on positions where marker
        # 11 differs leave only one mismatch to codeword 11. This violates
        # the declared one-bit noise model: log the *wrong* accepted ID.
        corrupt = self.trial(marker=10, flips=(2, 3, 4))
        report = replay_labeled_temporal_trials(
            [corrupt, self.trial(marker=11)],
            self.codebook,
        )
        self.assertEqual(report.wrong_accepts, 1)
        self.assertEqual(report.correct_accepts, 1)
        self.assertEqual(report.acceptance_fraction, 1.0)
        self.assertEqual(report.conditional_false_accept_fraction, 0.5)
        self.assertGreater(report.wrong_accept_one_sided_upper_95, 0.5)

    def test_heavily_erased_code_is_rejected_not_misclassified(self):
        report = replay_labeled_temporal_trials(
            [self.trial(marker=10, erased=(1, 2, 3, 4, 5, 6, 7))],
            self.codebook,
        )
        self.assertEqual(report.rejected, 1)
        self.assertEqual(report.wrong_accepts, 0)
        self.assertIsNone(report.conditional_false_accept_fraction)

    def test_invalid_decoder_configuration_fails_before_statistical_summary(self):
        valid = [self.trial()]
        for invalid in (0, 9, True, "4", 10**1000):
            with self.subTest(min_valid_samples=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "min_valid_samples"):
                    replay_labeled_temporal_trials(
                        valid, self.codebook,
                        min_valid_samples=invalid,
                    )

    def test_malformed_record_fails_replay_not_counted_as_rejection(self):
        duplicate = LabeledTemporalTrial(
            10, 1000, (
                LightSample(1000, 0), LightSample(1000, 0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            replay_labeled_temporal_trials([duplicate], self.codebook)
        invalid_value = LabeledTemporalTrial(
            10, 1000, (LightSample(1000, 0.5),),
        )
        with self.assertRaisesRegex(ValueError, "binary"):
            replay_labeled_temporal_trials([invalid_value], self.codebook)


if __name__ == "__main__":
    unittest.main()
