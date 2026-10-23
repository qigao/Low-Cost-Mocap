import unittest
from dataclasses import replace

from localization_contract import LocalizationQuality
from outdoor_sensor_contract import RTKSolutionStatus
from position_measurement import position_measurement_from_localization
from rtk_position_measurement import position_measurement_from_rtk
from test_outdoor_localization_replay import (
    T0,
    attitude_at,
    build_replay,
    rtk_sample,
    tag_pose,
    imu_sample,
)


class OutdoorReplayFaultTests(unittest.TestCase):
    def test_degraded_tag_and_float_rtk_are_rejected_without_state_change(self):
        owner, calibration, camera_clock, rtk_clock, rtk_cal = build_replay()
        before = owner.state
        degraded = replace(
            tag_pose(timestamp_ns=T0 + 25_000_000, sequence=11),
            quality=LocalizationQuality.DEGRADED,
        )
        with self.assertRaisesRegex(ValueError, "DEGRADED localization"):
            position_measurement_from_localization(
                degraded, camera_clock, expected_target_clock_domain="imu-clock"
            )

        floating = replace(rtk_sample(), status=RTKSolutionStatus.FLOAT)
        with self.assertRaises(ValueError):
            position_measurement_from_rtk(
                floating,
                rtk_clock,
                rtk_cal,
                attitude_at(
                    calibration,
                    camera_clock,
                    rtk_clock.map_timestamp(T0 + 25_000_000),
                ),
            )
        self.assertEqual(owner.state, before)

    def test_changed_tag_map_revision_rolls_back_delayed_update(self):
        owner, _calibration, camera_clock, _rtk_clock, _rtk_cal = build_replay()
        before = owner.state
        retained_before = owner.retained_frame_count
        changed = replace(
            tag_pose(
                timestamp_ns=T0 + 25_000_000,
                sequence=11,
                position=(1.0001, 2.0, 3.0),
            ),
            map_revision_sha256="d" * 64,
        )
        measurement = position_measurement_from_localization(
            changed, camera_clock, expected_target_clock_domain="imu-clock"
        )
        with self.assertRaisesRegex(ValueError, "source revision mismatch"):
            owner.ingest_position_measurement(measurement, max_position_nis=100.0)
        self.assertEqual(owner.state, before)
        self.assertEqual(owner.retained_frame_count, retained_before)

    def test_lost_attitude_does_not_consume_imu_frame_and_tracking_can_reacquire(self):
        owner, calibration, camera_clock, _rtk_clock, _rtk_cal = build_replay()
        timestamp = camera_clock.map_timestamp(T0 + 60_000_000)
        sample = imu_sample(calibration, 6, timestamp)
        before = owner.state
        retained_before = owner.retained_frame_count
        lost = replace(
            attitude_at(calibration, camera_clock, timestamp),
            status="LOST",
            source="LOST",
            quaternion_world_from_body_xyzw=None,
            attitude_variance_rad2=None,
        )
        with self.assertRaises(ValueError):
            owner.ingest_imu(sample, lost)
        self.assertEqual(owner.state, before)
        self.assertEqual(owner.retained_frame_count, retained_before)

        tracking = replace(
            attitude_at(calibration, camera_clock, timestamp),
            status="TRACKING",
            source="LOCALIZATION_CORRECTED",
            last_localization_timestamp_ns=timestamp,
            last_source_sequence_number=12,
            localization_age_ns=0,
        )
        recovered = owner.ingest_imu(sample, tracking)
        self.assertEqual(recovered.timestamp_ns, timestamp)
        self.assertNotEqual(recovered.status, "LOST")
        self.assertEqual(owner.retained_frame_count, retained_before + 1)


if __name__ == "__main__":
    unittest.main()
