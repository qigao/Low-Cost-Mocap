import unittest

import numpy as np

from light_tracking import BoundedLightTracker


class BoundedLightTrackerTests(unittest.TestCase):
    def test_track_identity_survives_off_phase_and_detector_permutation(self):
        tracker = BoundedLightTracker(
            camera_id=2, max_motion_px_per_trigger=4.0,
            max_missing_triggers=3,
        )
        initial = tracker.update(50, [[10., 20.], [38., 25.]])
        self.assertEqual([x.track_id for x in initial], [0, 1])
        missing = tracker.update(51, [[39., 25.]])
        self.assertIsNone(missing[0].observed_xy)
        self.assertEqual(missing[0].missed_triggers, 1)
        self.assertEqual(missing[1].track_id, 1)
        # The detector sees lights in reverse order after one off frame.
        recovered = tracker.update(52, [[40., 25.], [11., 20.]])
        self.assertEqual(recovered[0].track_id, 0)
        self.assertEqual(recovered[1].track_id, 1)
        np.testing.assert_allclose(recovered[0].observed_xy, [11., 20.])
        np.testing.assert_allclose(recovered[1].observed_xy, [40., 25.])
        self.assertEqual(tracker.active_track_ids, (0, 1))

    def test_strict_rejection_of_crossing_ambiguity_without_mutating_state(self):
        tracker = BoundedLightTracker(
            camera_id=0, max_motion_px_per_trigger=4.0,
            ambiguity_margin_fraction=0.05,
            minimum_detection_separation_px=0.0,
        )
        tracker.update(100, [[10., 20.], [12., 20.]])
        with self.assertRaisesRegex(ValueError, "ambiguous light-track"):
            tracker.update(101, [[10.99, 20.], [11.01, 20.]])
        self.assertEqual(tracker.last_trigger_id, 100)
        self.assertEqual(tracker.active_track_ids, (0, 1))
        outputs = tracker.update(101, [[10., 20.], [12., 20.]])
        self.assertEqual(tuple(x.track_id for x in outputs), (0, 1))

    def test_temporally_dark_track_cannot_steal_bright_neighbor(self):
        tracker = BoundedLightTracker(
            camera_id=4, max_motion_px_per_trigger=8.,
            minimum_detection_separation_px=0.0,
        )
        tracker.update(100, [(20., 20.), (30., 20.)])
        # After rapid motion, the lone bright point is closer to track 0.
        # The active registry says track 0 must be OFF on this phase,
        # therefore its ID is not allowed to consume this centroid.
        outputs = tracker.update(
            101, [(24., 20.)], forbidden_track_ids=[0]
        )
        self.assertIsNone(outputs[0].observed_xy)
        self.assertEqual(outputs[0].missed_triggers, 1)
        self.assertEqual(outputs[1].observed_xy, (24., 20.))
        self.assertEqual(tracker.active_track_ids, (0, 1))

    def test_staging_does_not_commit_until_caller_accepts_clone(self):
        base = BoundedLightTracker(camera_id=2)
        base.update(100, [(20., 20.)])
        stage = base.staged_copy()
        stage.update(101, [(21., 20.)])
        self.assertEqual(base.last_trigger_id, 100)
        self.assertEqual(stage.last_trigger_id, 101)
        self.assertEqual(base.active_track_ids, (0,))
        with self.assertRaisesRegex(ValueError, "stale"):
            stage.update(101, [(21., 20.)])
        # A retry from the original state remains valid.
        valid = base.staged_copy().update(101, [(20., 20.)])
        self.assertEqual(valid[0].track_id, 0)
        self.assertEqual(base.last_trigger_id, 100)

    def test_forbidden_policy_malformed_ids_fail_closed(self):
        tracker = BoundedLightTracker(camera_id=0)
        tracker.update(10, [(20., 20.)])
        for ids in ([True], ["0"], [-1], [0, 0]):
            with self.subTest(ids=repr(ids)):
                with self.assertRaises(ValueError):
                    tracker.update(11, [(21., 20.)], forbidden_track_ids=ids)
                self.assertEqual(tracker.last_trigger_id, 10)

    def test_trigger_gap_rejected_and_missing_tracks_expire(self):
        tracker = BoundedLightTracker(
            camera_id=0, max_missing_triggers=2
        )
        tracker.update(11, [[10., 10.]])
        with self.assertRaisesRegex(ValueError, "missing tracker trigger"):
            tracker.update(13, [])
        self.assertEqual(tracker.last_trigger_id, 11)
        self.assertEqual(tracker.update(12, [])[0].missed_triggers, 1)
        self.assertEqual(tracker.update(13, [])[0].missed_triggers, 2)
        self.assertEqual(tracker.update(14, []), ())
        self.assertEqual(tracker.active_track_ids, ())

    def test_reacquires_after_exact_allowed_off_gap(self):
        tracker = BoundedLightTracker(
            camera_id=0, max_missing_triggers=2,
            max_motion_px_per_trigger=4.0,
        )
        tracker.update(30, [[10., 10.]])
        tracker.update(31, [])
        tracker.update(32, [])
        restored = tracker.update(33, [[10., 10.]])
        self.assertEqual(restored[0].track_id, 0)
        self.assertEqual(restored[0].missed_triggers, 0)

    def test_expired_track_does_not_consume_capacity_for_new_marker(self):
        tracker = BoundedLightTracker(
            camera_id=0, max_tracks=1,
            max_missing_triggers=1,
        )
        tracker.update(30, [[10., 10.]])
        tracker.update(31, [])
        output = tracker.update(32, [[30., 30.]])
        self.assertEqual(tuple(item.track_id for item in output), (1,))
        self.assertEqual(tracker.active_track_ids, (1,))

    def test_unmatched_bright_spot_born_as_separate_local_track(self):
        tracker = BoundedLightTracker(
            camera_id=1, max_tracks=3, max_motion_px_per_trigger=4.0,
        )
        tracker.update(100, [[10., 10.]])
        outputs = tracker.update(101, [[10., 10.], [40., 40.]])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[1].track_id, 1)
        self.assertEqual(outputs[1].camera_id, 1)

    def test_duplicate_detections_and_capacity_fail_closed(self):
        tracker = BoundedLightTracker(camera_id=0, max_tracks=2)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            tracker.update(7, [[10., 10.], [10.2, 10.]])
        self.assertIsNone(tracker.last_trigger_id)
        tracker.update(7, [[10., 10.], [30., 30.]])
        with self.assertRaisesRegex(ValueError, "capacity"):
            tracker.update(8, [[10., 10.], [30., 30.], [50., 50.]])
        self.assertEqual(tracker.last_trigger_id, 7)

    def test_reject_boolean_string_nan_and_non_monotonic_inputs(self):
        tracker = BoundedLightTracker(camera_id=0)
        for invalid in ([True, 5], ["20", "30"], [np.nan, 10]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tracker.update(1, [invalid])
        tracker.update(1, [[10., 10.]])
        with self.assertRaisesRegex(ValueError, "stale"):
            tracker.update(1, [[10., 10.]])
        self.assertEqual(tracker.last_trigger_id, 1)

    def test_transactional_rollback_on_ambiguous_new_birth(self):
        tracker = BoundedLightTracker(
            camera_id=0, minimum_detection_separation_px=0.5,
            max_motion_px_per_trigger=3.0,
        )
        tracker.update(50, [[20., 20.]])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            tracker.update(51, [[20., 20.], [20.1, 20.]])
        self.assertEqual(tracker.last_trigger_id, 50)
        outputs = tracker.update(51, [[20., 20.]])
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
