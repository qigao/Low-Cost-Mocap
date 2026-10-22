"""Admission regression: ID must come from the same light trace and frame."""
import unittest

from light_intensity import (
    BinaryIntensityCalibration, ROIContrast, StampedTrackROI,
)
from light_tracking import TrackEvidence
from marker_admission import admit_track_marker_at_frame
from temporal_pattern import (
    FrameStamp, TemporalCodebook, group_synchronized_detections,
)


class MarkerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.codes = TemporalCodebook({
            10: (0, 0, 0, 0, 1, 1, 1, 1),
            11: (0, 0, 1, 1, 0, 0, 1, 1),
            12: (0, 1, 0, 1, 0, 1, 0, 1),
            13: (0, 1, 1, 0, 1, 0, 0, 1),
        }, max_bit_errors=1)
        self.calibration = BinaryIntensityCalibration(0.0, 120.0)
        self.epoch = 1000
        self.source_id = 13

    def make_trace(self, camera=0, track=3, *, start=1000):
        pattern = self.codes.patterns[self.source_id]
        trace = []
        for phase, bit in enumerate(pattern):
            trigger = start + phase
            stamp = FrameStamp(
                camera, trigger, 5_000_000 + phase * 1000 + camera * 10,
                "validated-hardware-clock",
            )
            observed = (25.0 + camera * 3, 20.0) if bit else None
            evidence = TrackEvidence(
                track, camera, trigger, (25.0 + camera * 3, 20.0),
                observed, 0 if bit else 1,
            )
            roi = ROIContrast(
                120.0 if bit else 0.0, 0.0, 120.0 if bit else 0.0, 9, 28,
            )
            trace.append(StampedTrackROI(
                stamp, evidence, roi, predicted_roi_qualified=not bool(bit)
            ))
        return trace

    def admit(self, trace, *, camera=0, track=3, target=None):
        return admit_track_marker_at_frame(
            trace, self.codes, self.calibration,
            camera_id=camera, local_track_id=track,
            target_frame=trace[-1].frame_stamp if target is None else target,
            trigger_epoch=self.epoch,
        )

    def test_decoded_id_is_bound_to_actual_final_frame_centroid(self):
        trace0 = self.make_trace(camera=0, track=3)
        trace1 = self.make_trace(camera=1, track=8)
        first = self.admit(trace0)
        second = self.admit(trace1, camera=1, track=8)
        self.assertEqual((first.detection.marker_id, second.detection.marker_id), (13, 13))
        self.assertEqual(first.detection.xy, (25.0, 20.0))
        self.assertEqual(second.detection.xy, (28.0, 20.0))
        self.assertEqual(first.decoded_cycle_index, 0)
        self.assertEqual(first.local_track_id, 3)
        self.assertEqual(first.bit_errors, 0)
        batch, group = group_synchronized_detections(
            [second.detection, first.detection],
            [trace0[-1].frame_stamp, trace1[-1].frame_stamp],
            [0, 1], max_exposure_skew_ns=20,
        )
        self.assertEqual(batch.trigger_id, 1007)
        self.assertEqual(group.excluded_marker_ids, ())
        self.assertEqual(tuple(group.observations_by_marker_id), (13,))

    def test_over_budget_unique_decode_is_diagnostic_only(self):
        trace = self.make_trace()
        for phase in (0, 1):
            item = trace[phase]
            trace[phase] = StampedTrackROI(
                item.frame_stamp, item.track_evidence, None,
                predicted_roi_qualified=True,
            )

        with self.assertRaisesRegex(ValueError, "guaranteed erasure budget"):
            self.admit(trace)

        diagnostic = admit_track_marker_at_frame(
            trace, self.codes, self.calibration,
            camera_id=0, local_track_id=3,
            target_frame=trace[-1].frame_stamp,
            trigger_epoch=self.epoch,
            allow_degraded_erasure_decode=True,
        )
        self.assertEqual(diagnostic.detection.marker_id, self.source_id)
        self.assertEqual(diagnostic.erased_samples, 2)
        self.assertEqual(diagnostic.guaranteed_erasure_budget, 1)
        self.assertFalse(diagnostic.within_guaranteed_erasure_budget)

    def test_rejects_old_decode_with_a_new_frame(self):
        old = self.make_trace()
        next_cycle = FrameStamp(
            0, 1008, 5_008_000, "validated-hardware-clock"
        )
        with self.assertRaisesRegex(ValueError, "latest trace exposure"):
            self.admit(old, target=next_cycle)
        new_sample = StampedTrackROI(
            next_cycle,
            TrackEvidence(3, 0, 1008, (25., 20.), (25., 20.), 0),
            ROIContrast(120., 0., 120., 9, 28),
        )
        # An entire old cycle plus the next frame exceeds the strict
        # per-decode capacity before the decoder can inspect cycle IDs.
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.admit(old + [new_sample])
        # With fewer records than the cap, the cross-cycle check must
        # independently reject the same old-ID/new-frame splice.
        with self.assertRaisesRegex(ValueError, "different code cycles"):
            self.admit(old[-1:] + [new_sample])

    def test_rejects_identity_splice_across_local_tracks(self):
        trace = self.make_trace()
        original = trace[3]
        trace[3] = StampedTrackROI(
            original.frame_stamp,
            TrackEvidence(4, 0, original.frame_stamp.trigger_id,
                          (25., 20.), (25., 20.), 0),
            original.roi,
        )
        with self.assertRaisesRegex(ValueError, "mixed camera or local track"):
            self.admit(trace)

    def test_rejects_spliced_frame_or_camera(self):
        trace = self.make_trace()
        original = trace[2]
        for wrong in (
            FrameStamp(1, 1002, 5_002_000, "validated-hardware-clock"),
            FrameStamp(0, 1003, 5_002_000, "validated-hardware-clock"),
            FrameStamp(0, 1002, 5_002_000, "different-clock"),
        ):
            changed = list(trace)
            changed[2] = StampedTrackROI(
                wrong, original.track_evidence, original.roi
            )
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                self.admit(changed)

    def test_last_frame_must_have_real_on_detection(self):
        trace = self.make_trace()
        final = trace[-1]
        lost = list(trace)
        lost[-1] = StampedTrackROI(
            final.frame_stamp,
            TrackEvidence(3, 0, 1007, (25., 20.), None, 1),
            final.roi, predicted_roi_qualified=True,
        )
        with self.assertRaisesRegex(ValueError, "actually observed"):
            self.admit(lost)
        unlit = list(trace)
        unlit[-1] = StampedTrackROI(
            final.frame_stamp, final.track_evidence,
            ROIContrast(0., 0., 0., 9, 28),
        )
        with self.assertRaisesRegex(ValueError, "calibrated ON bit"):
            self.admit(unlit)

    def test_rejects_duplicate_frame_and_unqualified_missing_roi(self):
        trace = self.make_trace()
        repeated = trace[:-1] + [trace[-2]]
        with self.assertRaisesRegex(ValueError, "repeated trigger"):
            self.admit(repeated)
        unqualified = list(trace)
        first = unqualified[0]
        unqualified[0] = StampedTrackROI(
            first.frame_stamp, first.track_evidence,
            first.roi, predicted_roi_qualified=False,
        )
        with self.assertRaisesRegex(ValueError, "independent motion qualification"):
            self.admit(unqualified)


if __name__ == "__main__":
    unittest.main()
