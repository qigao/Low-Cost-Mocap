import unittest
from dataclasses import replace

from localization_replay import (
    LocalizationReplayConfig,
    LocalizationReplayEvent,
    replay_localization_events,
)
from position_measurement import position_measurement_from_localization
from rtk_position_measurement import (
    position_measurement_from_rtk,
    position_source_contract_from_rtk,
)
from rtk_coordinates import enu_to_ecef
from test_outdoor_localization_replay import (
    T0,
    attitude_at,
    clock_mapping,
    gravity,
    imu_calibration,
    imu_sample,
    rtk_calibration,
    rtk_sample,
    tag_pose,
)
from translational_propagation import translational_seed_from_localization


def replay_fixture():
    camera_clock = clock_mapping("camera-clock")
    rtk_clock = clock_mapping("rtk-clock")
    calibration = imu_calibration()
    rtk_cal = replace(rtk_calibration(), map_frame_id="site-map")
    seed = translational_seed_from_localization(
        tag_pose(),
        camera_clock,
        imu_id="imu-1",
        expected_imu_clock_domain="imu-clock",
        initial_velocity_world_m_s=(0.0, 0.0, 0.0),
        velocity_stddev_m_s=(0.1, 0.1, 0.1),
        residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
        residual_accel_bias_stddev_m_s2=(0.02, 0.02, 0.02),
    )
    contract = position_source_contract_from_rtk(
        rtk_cal, rtk_clock, expected_target_clock_domain="imu-clock"
    )
    config = LocalizationReplayConfig(
        seed=seed,
        imu_calibration=calibration,
        gravity_model=gravity(),
        position_source_contracts=(contract,),
    )
    events = []
    for sequence in range(6):
        raw_timestamp = T0 + sequence * 10_000_000
        timestamp = camera_clock.map_timestamp(raw_timestamp)
        events.append(LocalizationReplayEvent(
            event_index=len(events),
            kind="imu",
            sample=imu_sample(calibration, sequence, timestamp),
            attitude=attitude_at(calibration, camera_clock, timestamp),
        ))

    tag_measurement = position_measurement_from_localization(
        tag_pose(
            timestamp=T0 + 25_000_000,
            sequence=11,
            position=(1.0001, 2.0, 3.0),
        ),
        camera_clock,
        expected_target_clock_domain="imu-clock",
    )
    rtk_template = rtk_sample()
    rtk_value = replace(
        rtk_template,
        sequence_number=1,
        solution_timestamp_ns=T0 + 25_000_000,
        position_ecef_m=enu_to_ecef((1.0002, 2.0, 3.0), rtk_cal.enu_origin),
    )
    rtk_attitude = attitude_at(
        calibration,
        camera_clock,
        rtk_clock.map_timestamp(T0 + 25_000_000),
    )
    rtk_measurement = position_measurement_from_rtk(
        rtk_value, rtk_clock, rtk_cal, rtk_attitude
    )
    return config, events, tag_measurement, rtk_measurement


class LocalizationReplayTests(unittest.TestCase):
    def test_replay_digests_are_stable_and_bind_config_and_events(self):
        config, imu_events, tag_update, rtk_update = replay_fixture()
        events = tuple(imu_events + [
            LocalizationReplayEvent(6, "position", measurement=rtk_update, max_position_nis=100.0),
            LocalizationReplayEvent(7, "position", measurement=tag_update, max_position_nis=100.0),
        ])
        first = replay_localization_events(config, events)
        second = replay_localization_events(config, events)
        self.assertEqual(first.input_sha256, second.input_sha256)
        self.assertEqual(first.output_sha256, second.output_sha256)
        self.assertEqual(first.states, second.states)
        self.assertEqual(first.event_count, 8)
        self.assertFalse(first.hardware_provenance_verified)
        self.assertEqual(first.states[-1].last_position_measurement_source, "RTK")
        self.assertEqual(
            first.states[-1].last_position_update_timestamp_ns,
            tag_update.timestamp_ns,
        )

    def test_same_epoch_tag_rtk_arrival_order_does_not_change_final_state(self):
        config, imu_events, tag_update, rtk_update = replay_fixture()
        tag_first = tuple(imu_events + [
            LocalizationReplayEvent(6, "position", measurement=tag_update, max_position_nis=100.0),
            LocalizationReplayEvent(7, "position", measurement=rtk_update, max_position_nis=100.0),
        ])
        rtk_first = tuple(imu_events + [
            LocalizationReplayEvent(6, "position", measurement=rtk_update, max_position_nis=100.0),
            LocalizationReplayEvent(7, "position", measurement=tag_update, max_position_nis=100.0),
        ])
        first = replay_localization_events(config, tag_first)
        second = replay_localization_events(config, rtk_first)
        self.assertEqual(first.states[-1], second.states[-1])
        self.assertNotEqual(first.input_sha256, second.input_sha256)

    def test_replay_rejects_noncontiguous_events_and_capacity_overflow(self):
        config, imu_events, _tag_update, _rtk_update = replay_fixture()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replay_localization_events(config, (
                imu_events[0],
                replace(imu_events[1], event_index=9),
            ))
        with self.assertRaisesRegex(ValueError, "capacity"):
            replay_localization_events(config, tuple(imu_events), max_events=3)

    def test_stale_position_source_revision_aborts_replay(self):
        config, imu_events, tag_update, _rtk_update = replay_fixture()
        stale = replace(tag_update, source_revision_sha256="e" * 64)
        events = tuple(imu_events + [
            LocalizationReplayEvent(6, "position", measurement=stale, max_position_nis=100.0)
        ])
        with self.assertRaisesRegex(ValueError, "source revision mismatch"):
            replay_localization_events(config, events)


if __name__ == "__main__":
    unittest.main()
