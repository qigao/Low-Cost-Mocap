import unittest
from dataclasses import replace

import numpy as np

from delayed_position_replay import BoundedDelayedPositionReplay
from imu_calibration import CorrectedIMUVectors, IMUCalibration
from localization_contract import LocalizationQuality
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import (
    MetricPositionMeasurement,
    PositionMeasurementSource,
    PositionSourceContract,
)
from translational_eskf import ConditionedTranslationalESKF
from translational_propagation import GravityModel, TranslationalSeed


MAP_SHA = "a" * 64
LOC_CAL_SHA = "b" * 64
IMU_SHA_PLACEHOLDER = "c" * 64
CLOCK_SHA = "d" * 64
RTK_SOURCE_SHA = "e" * 64
RTK_CAL_SHA = "f" * 64
RTK_CLOCK_SHA = "1" * 64
T0 = 1_000_000_000


def imu_calibration():
    return IMUCalibration(
        imu_id="imu-1",
        revision="imu-cal-v1",
        imu_clock_domain="imu-clock",
        rotation_body_from_imu=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_body_from_imu_m=(0.0, 0.0, 0.0),
        gyro_bias_rad_s=(0.0, 0.0, 0.0),
        accel_bias_m_s2=(0.0, 0.0, 0.0),
        gyro_noise_density_rad_s_sqrt_hz=(0.01, 0.01, 0.01),
        accel_noise_density_m_s2_sqrt_hz=(0.03, 0.03, 0.03),
        gyro_bias_random_walk_rad_s_per_sqrt_s=(0.001, 0.001, 0.001),
        accel_bias_random_walk_m_s2_per_sqrt_s=(0.01, 0.01, 0.01),
        calibration_method="synthetic-test",
    )


def gravity():
    return GravityModel(
        frame_id="site-map",
        gravity_world_m_s2=(0.0, 0.0, -9.80665),
        model_id="site-gravity-v1",
    )


def seed():
    return TranslationalSeed(
        body_id="rigid-body",
        imu_id="imu-1",
        map_frame_id="site-map",
        timestamp_ns=T0,
        source_sequence_number=10,
        clock_domain="imu-clock",
        position_world_from_body_m=(1.0, 2.0, 3.0),
        velocity_world_from_body_m_s=(0.0, 0.0, 0.0),
        residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
        position_stddev_m=(0.05, 0.05, 0.05),
        velocity_stddev_m_s=(0.3, 0.3, 0.3),
        residual_accel_bias_stddev_m_s2=(0.3, 0.3, 0.3),
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
        clock_mapping_sha256=CLOCK_SHA,
    )


def sample(calibration, sequence, timestamp, force=(0.7, 0.0, 9.80665)):
    return CorrectedIMUVectors(
        imu_id="imu-1",
        sequence_number=sequence,
        sensor_timestamp_ns=timestamp,
        clock_domain="imu-clock",
        calibration_revision=calibration.revision,
        calibration_sha256=calibration.calibration_sha256,
        gyro_body_rad_s=(0.0, 0.0, 0.0),
        specific_force_body_m_s2=force,
    )


def attitude(calibration, timestamp):
    return LocalizationIMUAttitudeState(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=(0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=1e-4,
        status="TRACKING" if timestamp == T0 else "DEGRADED",
        source=(
            "LOCALIZATION_CORRECTED"
            if timestamp == T0
            else "IMU_PROPAGATED"
        ),
        last_localization_timestamp_ns=T0,
        last_source_sequence_number=10,
        localization_age_ns=timestamp - T0,
        clock_mapping_sha256=CLOCK_SHA,
        imu_calibration_sha256=calibration.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )


def localization_measurement(timestamp, sequence, *, position=(1.0, 2.0, 3.0), stddev=(0.02, 0.02, 0.02)):
    return MetricPositionMeasurement(
        source=PositionMeasurementSource.LOCALIZATION,
        body_id="rigid-body",
        map_frame_id="site-map",
        timestamp_ns=timestamp,
        source_sequence_number=sequence,
        clock_domain="imu-clock",
        position_world_from_body_m=position,
        position_stddev_m=stddev,
        quality=LocalizationQuality.TRACKING,
        source_revision_sha256=MAP_SHA,
        measurement_calibration_sha256=LOC_CAL_SHA,
        clock_mapping_sha256=CLOCK_SHA,
    )


def rtk_measurement(timestamp, sequence, *, position=(1.0, 2.0, 3.0), stddev=(0.03, 0.03, 0.05)):
    return MetricPositionMeasurement(
        source=PositionMeasurementSource.RTK,
        body_id="rigid-body",
        map_frame_id="site-map",
        timestamp_ns=timestamp,
        source_sequence_number=sequence,
        clock_domain="imu-clock",
        position_world_from_body_m=position,
        position_stddev_m=stddev,
        quality=LocalizationQuality.TRACKING,
        source_revision_sha256=RTK_SOURCE_SHA,
        measurement_calibration_sha256=RTK_CAL_SHA,
        clock_mapping_sha256=RTK_CLOCK_SHA,
    )


def rtk_source_contract():
    return PositionSourceContract(
        source=PositionMeasurementSource.RTK,
        source_revision_sha256=RTK_SOURCE_SHA,
        measurement_calibration_sha256=RTK_CAL_SHA,
        clock_mapping_sha256=RTK_CLOCK_SHA,
    )


def filter_kwargs(**changes):
    values = dict(
        max_sample_gap_ns=20_000_000,
        max_step_ns=5_000_000,
        max_substeps=64,
        max_position_dead_reckoning_ns=100_000_000,
        max_attitude_variance_rad2=0.05,
        max_residual_accel_bias_m_s2=2.0,
        body_origin_tolerance_m=1e-9,
    )
    values.update(changes)
    return values


def new_filter(**changes):
    calibration = imu_calibration()
    return (
        ConditionedTranslationalESKF(
            seed(),
            calibration,
            gravity(),
            **filter_kwargs(**changes),
            position_source_contracts=(rtk_source_contract(),),
        ),
        calibration,
    )


def assert_state_close(testcase, left, right, atol=1e-12):
    testcase.assertEqual(left.status, right.status)
    testcase.assertEqual(left.source, right.source)
    testcase.assertEqual(left.timestamp_ns, right.timestamp_ns)
    testcase.assertEqual(left.imu_sequence_number, right.imu_sequence_number)
    testcase.assertEqual(
        left.last_position_measurement_source,
        right.last_position_measurement_source,
    )
    testcase.assertEqual(
        left.last_position_source_sequence_number,
        right.last_position_source_sequence_number,
    )
    testcase.assertEqual(
        left.last_position_update_timestamp_ns,
        right.last_position_update_timestamp_ns,
    )
    if left.position_world_from_body_m is None:
        testcase.assertIsNone(right.position_world_from_body_m)
        testcase.assertIsNone(right.velocity_world_from_body_m_s)
        testcase.assertIsNone(right.covariance_p_v_bias)
        return
    np.testing.assert_allclose(
        left.position_world_from_body_m,
        right.position_world_from_body_m,
        atol=atol,
        rtol=0,
    )
    np.testing.assert_allclose(
        left.velocity_world_from_body_m_s,
        right.velocity_world_from_body_m_s,
        atol=atol,
        rtol=0,
    )
    np.testing.assert_allclose(
        left.residual_accel_bias_body_m_s2,
        right.residual_accel_bias_body_m_s2,
        atol=atol,
        rtol=0,
    )
    np.testing.assert_allclose(
        left.covariance_p_v_bias,
        right.covariance_p_v_bias,
        atol=atol,
        rtol=0,
    )


class TranslationalESKFSnapshotTests(unittest.TestCase):
    def test_snapshot_restore_reproduces_future_state_exactly(self):
        estimator, calibration = new_filter()
        estimator.advance(sample(calibration, 0, T0), attitude(calibration, T0))
        t1 = T0 + 10_000_000
        estimator.advance(sample(calibration, 1, t1), attitude(calibration, t1))
        estimator.ingest_position_measurement(
            localization_measurement(t1, 11),
            max_position_nis=100.0,
        )
        checkpoint = estimator.snapshot()

        restored, same_calibration = new_filter()
        restored.restore(checkpoint)
        self.assertEqual(restored.state, estimator.state)
        self.assertEqual(
            restored.estimator_fingerprint_sha256,
            estimator.estimator_fingerprint_sha256,
        )

        t2 = T0 + 20_000_000
        expected = estimator.advance(
            sample(calibration, 2, t2),
            attitude(calibration, t2),
        )
        actual = restored.advance(
            sample(same_calibration, 2, t2),
            attitude(same_calibration, t2),
        )
        assert_state_close(self, expected, actual)

    def test_snapshot_rejects_other_estimator_configuration(self):
        estimator, calibration = new_filter()
        estimator.advance(sample(calibration, 0, T0), attitude(calibration, T0))
        checkpoint = estimator.snapshot()
        other, _ = new_filter(max_position_dead_reckoning_ns=200_000_000)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            other.restore(checkpoint)

    def test_snapshot_fingerprint_includes_position_source_contracts(self):
        estimator, calibration = new_filter()
        estimator.advance(
            sample(calibration, 0, T0),
            attitude(calibration, T0),
        )
        checkpoint = estimator.snapshot()
        other_contract = PositionSourceContract(
            source=PositionMeasurementSource.RTK,
            source_revision_sha256=RTK_SOURCE_SHA,
            measurement_calibration_sha256="0" * 64,
            clock_mapping_sha256=RTK_CLOCK_SHA,
        )
        other = ConditionedTranslationalESKF(
            seed(),
            calibration,
            gravity(),
            **filter_kwargs(),
            position_source_contracts=(other_contract,),
        )
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            other.restore(checkpoint)

    def test_snapshot_lifecycle_invariants_fail_closed(self):
        estimator, calibration = new_filter()
        estimator.advance(sample(calibration, 0, T0), attitude(calibration, T0))
        checkpoint = estimator.snapshot()
        with self.assertRaisesRegex(ValueError, "expired flag"):
            replace(checkpoint, expired=True)
        with self.assertRaisesRegex(ValueError, "absent or present together"):
            replace(checkpoint, last_attitude=None)


class DelayedPositionReplayTests(unittest.TestCase):
    def build_wrapper(self, **kwargs):
        calibration = imu_calibration()
        owner = BoundedDelayedPositionReplay(
            seed(), calibration, gravity(),
            **filter_kwargs(**kwargs),
            max_history_frames=32,
            max_history_ns=200_000_000,
            position_source_contracts=(rtk_source_contract(),),
        )
        return owner, calibration

    def ingest_frames(self, owner, calibration, count):
        states = []
        for sequence in range(count):
            timestamp = T0 + sequence * 10_000_000
            states.append(owner.ingest_imu(
                sample(calibration, sequence, timestamp),
                attitude(calibration, timestamp),
            ))
        return states

    def test_delayed_update_matches_in_order_reference_replay(self):
        reference, calibration = new_filter()
        measurement = localization_measurement(
            T0 + 20_000_000,
            11,
            position=(1.0001, 2.0, 3.0),
        )
        expected = None
        for sequence in range(6):
            timestamp = T0 + sequence * 10_000_000
            expected = reference.advance(
                sample(calibration, sequence, timestamp),
                attitude(calibration, timestamp),
            )
            if sequence == 2:
                expected = reference.ingest_position_measurement(
                    measurement, max_position_nis=100.0
                )

        owner, owner_calibration = self.build_wrapper()
        self.ingest_frames(owner, owner_calibration, 6)
        actual = owner.ingest_position_measurement(
            measurement,
            max_position_nis=100.0,
        )
        assert_state_close(self, expected, actual)

    def test_delayed_position_can_recover_current_lost_state_within_window(self):
        owner, calibration = self.build_wrapper(
            max_position_dead_reckoning_ns=25_000_000
        )
        states = self.ingest_frames(owner, calibration, 4)
        self.assertEqual(states[-1].status, "LOST")

        recovered = owner.ingest_position_measurement(
            localization_measurement(
                T0 + 20_000_000,
                11,
                position=(1.0001, 2.0, 3.0),
            ),
            max_position_nis=100.0,
        )
        self.assertEqual(recovered.timestamp_ns, T0 + 30_000_000)
        self.assertEqual(recovered.status, "DEGRADED")
        self.assertIsNotNone(recovered.position_world_from_body_m)
        self.assertEqual(recovered.position_age_ns, 10_000_000)

    def test_measurement_older_than_pruned_window_is_rejected(self):
        calibration = imu_calibration()
        owner = BoundedDelayedPositionReplay(
            seed(), calibration, gravity(),
            **filter_kwargs(),
            max_history_frames=3,
            max_history_ns=200_000_000,
        )
        self.ingest_frames(owner, calibration, 5)
        self.assertEqual(owner.retained_frame_count, 3)
        self.assertEqual(
            owner.retained_timestamp_range_ns,
            (T0 + 20_000_000, T0 + 40_000_000),
        )
        with self.assertRaisesRegex(ValueError, "outside retained"):
            owner.ingest_position_measurement(
                localization_measurement(T0 + 10_000_000, 11)
            )

    def test_history_time_window_prunes_old_exact_frames(self):
        calibration = imu_calibration()
        owner = BoundedDelayedPositionReplay(
            seed(), calibration, gravity(),
            **filter_kwargs(),
            max_history_frames=32,
            max_history_ns=15_000_000,
        )
        self.ingest_frames(owner, calibration, 4)
        self.assertEqual(
            owner.retained_timestamp_range_ns,
            (T0 + 20_000_000, T0 + 30_000_000),
        )
        with self.assertRaisesRegex(ValueError, "outside retained"):
            owner.ingest_position_measurement(
                localization_measurement(T0 + 10_000_000, 11)
            )

    def test_failed_delayed_nis_replay_is_transactional(self):
        owner, calibration = self.build_wrapper()
        self.ingest_frames(owner, calibration, 5)
        before = owner.state
        with self.assertRaisesRegex(ValueError, "NIS"):
            owner.ingest_position_measurement(
                localization_measurement(
                    T0 + 20_000_000,
                    11,
                    position=(100.0, 100.0, 100.0),
                    stddev=(0.001, 0.001, 0.001),
                ),
                max_position_nis=9.0,
            )
        self.assertEqual(owner.state, before)

        # The rejected sequence was not committed and can be replaced by a
        # valid measurement with the same source sequence.
        accepted = owner.ingest_position_measurement(
            localization_measurement(
                T0 + 20_000_000,
                11,
                position=(1.0001, 2.0, 3.0),
            ),
            max_position_nis=100.0,
        )
        self.assertEqual(accepted.last_position_source_sequence_number, 11)

    def test_delayed_source_sequence_must_agree_with_timestamp_order(self):
        owner, calibration = self.build_wrapper()
        self.ingest_frames(owner, calibration, 6)
        owner.ingest_position_measurement(
            localization_measurement(T0 + 40_000_000, 13),
            max_position_nis=100.0,
        )
        accepted = owner.ingest_position_measurement(
            localization_measurement(T0 + 30_000_000, 12),
            max_position_nis=100.0,
        )
        self.assertEqual(accepted.timestamp_ns, T0 + 50_000_000)

        before = owner.state
        with self.assertRaisesRegex(ValueError, "timestamp order"):
            owner.ingest_position_measurement(
                localization_measurement(T0 + 20_000_000, 14),
                max_position_nis=100.0,
            )
        self.assertEqual(owner.state, before)

    def test_same_timestamp_localization_and_rtk_have_canonical_arrival_independent_order(self):
        timestamp = T0 + 20_000_000
        localization = localization_measurement(
            timestamp, 11, position=(1.0001, 2.0, 3.0)
        )
        rtk = rtk_measurement(
            timestamp, 1, position=(1.0002, 2.0, 3.0)
        )

        first, cal1 = self.build_wrapper()
        self.ingest_frames(first, cal1, 5)
        first.ingest_position_measurement(rtk, max_position_nis=100.0)
        state_a = first.ingest_position_measurement(
            localization, max_position_nis=100.0
        )

        second, cal2 = self.build_wrapper()
        self.ingest_frames(second, cal2, 5)
        second.ingest_position_measurement(
            localization, max_position_nis=100.0
        )
        state_b = second.ingest_position_measurement(
            rtk, max_position_nis=100.0
        )
        assert_state_close(self, state_a, state_b)

    def test_duplicate_same_source_at_same_timestamp_is_rejected(self):
        owner, calibration = self.build_wrapper()
        self.ingest_frames(owner, calibration, 4)
        owner.ingest_position_measurement(
            localization_measurement(T0 + 20_000_000, 11),
            max_position_nis=100.0,
        )
        with self.assertRaisesRegex(ValueError, "one replay timestamp"):
            owner.ingest_position_measurement(
                localization_measurement(T0 + 20_000_000, 12),
                max_position_nis=100.0,
            )

    def test_rtk_source_calibration_and_clock_revisions_are_pinned(self):
        owner, calibration = self.build_wrapper()
        self.ingest_frames(owner, calibration, 5)
        timestamp = T0 + 20_000_000
        before = owner.state
        bad_measurements = (
            (
                replace(
                    rtk_measurement(timestamp, 1),
                    source_revision_sha256="9" * 64,
                ),
                "source revision mismatch",
            ),
            (
                replace(
                    rtk_measurement(timestamp, 1),
                    measurement_calibration_sha256="8" * 64,
                ),
                "position calibration mismatch",
            ),
            (
                replace(
                    rtk_measurement(timestamp, 1),
                    clock_mapping_sha256="7" * 64,
                ),
                "clock mapping mismatch",
            ),
        )
        for measurement, message in bad_measurements:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                owner.ingest_position_measurement(
                    measurement, max_position_nis=100.0
                )
            self.assertEqual(owner.state, before)

        accepted = owner.ingest_position_measurement(
            rtk_measurement(timestamp, 1),
            max_position_nis=100.0,
        )
        self.assertEqual(
            accepted.last_position_measurement_source,
            PositionMeasurementSource.RTK.value,
        )

    def test_rtk_updates_require_an_explicit_source_contract(self):
        calibration = imu_calibration()
        owner = BoundedDelayedPositionReplay(
            seed(),
            calibration,
            gravity(),
            **filter_kwargs(),
            max_history_frames=32,
            max_history_ns=200_000_000,
        )
        self.ingest_frames(owner, calibration, 5)
        before = owner.state
        with self.assertRaisesRegex(
            ValueError, "RTK position source contract is not configured"
        ):
            owner.ingest_position_measurement(
                rtk_measurement(T0 + 20_000_000, 1),
                max_position_nis=100.0,
            )
        self.assertEqual(owner.state, before)

    def test_position_timestamp_must_match_retained_imu_frame_exactly(self):
        owner, calibration = self.build_wrapper()
        self.ingest_frames(owner, calibration, 4)
        with self.assertRaisesRegex(ValueError, "exact IMU replay frames"):
            owner.ingest_position_measurement(
                localization_measurement(T0 + 25_000_000, 11)
            )


if __name__ == "__main__":
    unittest.main()
