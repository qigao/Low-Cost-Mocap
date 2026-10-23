import unittest
from dataclasses import replace

from outdoor_sensor_contract import RTKSolutionStatus
from rtk_coordinates import enu_to_ecef
from rtk_position_measurement import position_measurement_from_rtk
from test_outdoor_localization_replay import (
    T0,
    attitude_at,
    build_replay,
    imu_sample,
    rtk_sample,
)


class OutdoorRecoveryReplayTests(unittest.TestCase):
    def test_delayed_rtk_reacquisition_recovers_expired_translation_transactionally(self):
        owner, calibration, camera_clock, rtk_clock, rtk_cal = build_replay()
        # No absolute-position corrections arrive during the following 110 ms.
        for sequence in range(6, 12):
            timestamp = camera_clock.map_timestamp(T0 + sequence * 10_000_000)
            state = owner.ingest_imu(
                imu_sample(calibration, sequence, timestamp),
                attitude_at(calibration, camera_clock, timestamp),
            )
        self.assertEqual(state.status, "LOST")
        self.assertIsNone(state.position_world_from_body_m)
        self.assertEqual(state.timestamp_ns, camera_clock.map_timestamp(T0 + 110_000_000))

        # A valid FIXED RTK observation acquired at 95 ms arrives after the
        # current state has expired. Replay must insert it at acquisition time,
        # not arrival time, and propagate through the retained frames again.
        recovered_epoch = T0 + 95_000_000
        rtk_template = rtk_sample()
        fixed_delayed = replace(
            rtk_template,
            sequence_number=8,
            solution_timestamp_ns=recovered_epoch,
            status=RTKSolutionStatus.FIXED,
            position_ecef_m=enu_to_ecef((1.0001, 2.0, 3.0), rtk_cal.enu_origin),
        )
        update = position_measurement_from_rtk(
            fixed_delayed,
            rtk_clock,
            rtk_cal,
            attitude_at(
                calibration,
                camera_clock,
                rtk_clock.map_timestamp(recovered_epoch),
            ),
        )
        recovered = owner.ingest_position_measurement(update, max_position_nis=100.0)
        self.assertEqual(recovered.status, "DEGRADED")
        self.assertEqual(recovered.timestamp_ns, camera_clock.map_timestamp(T0 + 110_000_000))
        self.assertEqual(recovered.last_position_measurement_source, "RTK")
        self.assertEqual(recovered.last_position_update_timestamp_ns, update.timestamp_ns)
        self.assertIsNotNone(recovered.position_world_from_body_m)


if __name__ == "__main__":
    unittest.main()
