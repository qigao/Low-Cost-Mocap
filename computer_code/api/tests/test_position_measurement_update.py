import unittest

import numpy as np

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from imu_contract import ClockAnchor, fit_clock_mapping
from localization_contract import (
    LocalizationQuality,
    LocalizationTopology,
    MapPoseObservation,
    PoseSource,
    ScaleReference,
)
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import (
    MetricPositionMeasurement,
    PositionMeasurementSource,
    position_measurement_from_localization,
)
from translational_propagation import (
    BoundedIMUTranslationPropagator,
    GravityModel,
    translational_seed_from_localization,
)


MAP_SHA = "b" * 64
LOC_CAL_SHA = "c" * 64


def mapping():
    return fit_clock_mapping(
        [
            ClockAnchor(1_000_000_000, 1_000_001_000, event_id="sync-1"),
            ClockAnchor(2_000_000_000, 2_000_001_000, event_id="sync-2"),
            ClockAnchor(3_000_000_000, 3_000_001_000, event_id="sync-3"),
        ],
        source_clock_domain="camera-clock",
        target_clock_domain="imu-clock",
        max_drift_ppm=10,
        max_residual_ns=10,
        max_extrapolation_ns=20_000_000,
        min_anchor_span_ns=100_000_000,
    )


def localization_pose(timestamp=1_000_000_000, sequence=10, **changes):
    values = dict(
        source=PoseSource.FIDUCIAL_LOCALIZATION,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id="rigid-body",
        map_frame_id="site-map",
        timestamp_ns=timestamp,
        source_sequence_number=sequence,
        clock_domain="camera-clock",
        position_xyz=(1.0, 2.0, 3.0),
        position_unit="meters",
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_stddev=(0.02, 0.03, 0.04),
        orientation_stddev_rad=(0.01, 0.01, 0.02),
        map_revision_sha256=MAP_SHA,
        calibration_sha256=LOC_CAL_SHA,
        quality=LocalizationQuality.TRACKING,
        inlier_count=3,
        metric_scale_qualified=True,
        scale_reference=ScaleReference.CALIBRATED_TAG_MAP,
        relocalized=False,
        reprojection_rms_px=0.3,
    )
    values.update(changes)
    return MapPoseObservation(**values)


def imu_calibration():
    return IMUCalibration(
        imu_id="imu-1",
        revision="imu-cal-v1",
        imu_clock_domain="imu-clock",
        rotation_body_from_imu=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_body_from_imu_m=(0.0, 0.0, 0.0),
        gyro_bias_rad_s=(0.0, 0.0, 0.0),
        accel_bias_m_s2=(0.0, 0.0, 0.0),
        gyro_noise_density_rad_s_sqrt_hz=(0.01, 0.01, 0.01),
        accel_noise_density_m_s2_sqrt_hz=(0.02, 0.02, 0.02),
        gyro_bias_random_walk_rad_s_per_sqrt_s=(0.001, 0.001, 0.001),
        accel_bias_random_walk_m_s2_per_sqrt_s=(0.002, 0.002, 0.002),
        calibration_method="synthetic-test",
    )


def seed():
    return translational_seed_from_localization(
        localization_pose(),
        mapping(),
        imu_id="imu-1",
        expected_imu_clock_domain="imu-clock",
        initial_velocity_world_m_s=(0.0, 0.0, 0.0),
        velocity_stddev_m_s=(0.1, 0.1, 0.1),
        residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
        residual_accel_bias_stddev_m_s2=(0.02, 0.02, 0.02),
    )


def gravity():
    return GravityModel(
        frame_id="site-map",
        gravity_world_m_s2=(0.0, 0.0, -9.80665),
        model_id="site-gravity-v1",
    )


def sample(calibration, sequence, timestamp, force=(0.0, 0.0, 9.80665)):
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
    t0 = 1_000_001_000
    tracking = timestamp == t0
    return LocalizationIMUAttitudeState(
        body_id="rigid-body",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=(0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=1e-4,
        status="TRACKING" if tracking else "DEGRADED",
        source="LOCALIZATION_CORRECTED" if tracking else "IMU_PROPAGATED",
        last_localization_timestamp_ns=t0,
        last_source_sequence_number=10,
        localization_age_ns=timestamp - t0,
        clock_mapping_sha256=mapping().mapping_sha256,
        imu_calibration_sha256=calibration.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )


def setup_propagator(*, max_dead_reckoning_ns=100_000_000):
    calibration = imu_calibration()
    propagator = BoundedIMUTranslationPropagator(
        seed(),
        calibration,
        gravity(),
        max_position_dead_reckoning_ns=max_dead_reckoning_ns,
    )
    t0 = 1_000_001_000
    propagator.advance(sample(calibration, 0, t0), attitude(calibration, t0))
    return propagator, calibration


class PositionMeasurementAdapterTests(unittest.TestCase):
    def test_tracking_localization_maps_into_common_position_contract(self):
        measurement = position_measurement_from_localization(
            localization_pose(timestamp=1_010_000_000, sequence=11),
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        self.assertEqual(measurement.source, PositionMeasurementSource.LOCALIZATION)
        self.assertEqual(measurement.timestamp_ns, 1_010_001_000)
        self.assertEqual(measurement.source_sequence_number, 11)
        self.assertEqual(measurement.position_world_from_body_m, (1.0, 2.0, 3.0))
        self.assertEqual(measurement.position_stddev_m, (0.02, 0.03, 0.04))
        self.assertEqual(measurement.clock_mapping_sha256, mapping().mapping_sha256)

    def test_degraded_localization_requires_opt_in_and_inflates_variance(self):
        degraded = localization_pose(
            timestamp=1_010_000_000,
            sequence=11,
            quality=LocalizationQuality.DEGRADED,
        )
        with self.assertRaisesRegex(ValueError, "not enabled"):
            position_measurement_from_localization(
                degraded,
                mapping(),
                expected_target_clock_domain="imu-clock",
            )
        measurement = position_measurement_from_localization(
            degraded,
            mapping(),
            expected_target_clock_domain="imu-clock",
            allow_degraded=True,
            degraded_variance_scale=4.0,
        )
        self.assertEqual(measurement.quality, LocalizationQuality.DEGRADED)
        self.assertEqual(measurement.position_stddev_m, (0.04, 0.06, 0.08))

    def test_unqualified_or_wrong_clock_localization_fails_closed(self):
        unscaled = localization_pose(
            position_unit="map_units",
            metric_scale_qualified=False,
            scale_reference=ScaleReference.NONE,
        )
        with self.assertRaisesRegex(ValueError, "qualified metric"):
            position_measurement_from_localization(
                unscaled,
                mapping(),
                expected_target_clock_domain="imu-clock",
            )
        with self.assertRaisesRegex(ValueError, "target"):
            position_measurement_from_localization(
                localization_pose(),
                mapping(),
                expected_target_clock_domain="other-clock",
            )


class PositionCorrectionTests(unittest.TestCase):
    def test_current_time_localization_corrects_position_not_velocity_or_bias(self):
        propagator, calibration = setup_propagator()
        t1 = 1_010_001_000
        propagated = propagator.advance(
            sample(calibration, 1, t1, force=(1.0, 0.0, 9.80665)),
            attitude(calibration, t1),
        )
        before_position = np.asarray(propagated.position_world_from_body_m)
        before_variance = np.asarray(propagated.position_variance_m2)
        before_velocity = propagated.velocity_world_from_body_m_s
        before_velocity_variance = propagated.velocity_variance_m2_s2
        before_bias = propagated.residual_accel_bias_body_m_s2
        before_bias_variance = propagated.residual_accel_bias_variance_m2_s4

        observation = localization_pose(
            timestamp=1_010_000_000,
            sequence=11,
            position_xyz=(1.0, 2.0, 3.0),
            position_stddev=(0.005, 0.005, 0.005),
        )
        measurement = position_measurement_from_localization(
            observation,
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        corrected = propagator.ingest_position_measurement(measurement)

        after_position = np.asarray(corrected.position_world_from_body_m)
        self.assertLess(
            np.linalg.norm(after_position - np.asarray((1.0, 2.0, 3.0))),
            np.linalg.norm(before_position - np.asarray((1.0, 2.0, 3.0))),
        )
        self.assertTrue(
            np.all(np.asarray(corrected.position_variance_m2) < before_variance)
        )
        self.assertEqual(corrected.velocity_world_from_body_m_s, before_velocity)
        self.assertEqual(corrected.velocity_variance_m2_s2, before_velocity_variance)
        self.assertEqual(corrected.residual_accel_bias_body_m_s2, before_bias)
        self.assertEqual(
            corrected.residual_accel_bias_variance_m2_s4,
            before_bias_variance,
        )
        self.assertEqual(corrected.source, "POSITION_CORRECTED")
        self.assertEqual(corrected.status, "TRACKING")
        self.assertEqual(corrected.position_age_ns, 0)
        self.assertEqual(corrected.last_position_source_sequence_number, 11)
        self.assertEqual(
            corrected.last_position_measurement_source,
            PositionMeasurementSource.LOCALIZATION.value,
        )

    def test_position_correction_resets_dead_reckoning_horizon(self):
        propagator, calibration = setup_propagator(max_dead_reckoning_ns=15_000_000)
        t1 = 1_010_001_000
        propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        correction = position_measurement_from_localization(
            localization_pose(timestamp=1_010_000_000, sequence=11),
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        propagator.ingest_position_measurement(correction)

        t2 = 1_020_001_000
        still_valid = propagator.advance(
            sample(calibration, 2, t2),
            attitude(calibration, t2),
        )
        self.assertEqual(still_valid.status, "DEGRADED")
        self.assertEqual(still_valid.position_age_ns, 10_000_000)

        t3 = 1_030_001_000
        lost = propagator.advance(
            sample(calibration, 3, t3),
            attitude(calibration, t3),
        )
        self.assertEqual(lost.status, "LOST")
        self.assertIsNone(lost.position_world_from_body_m)

    def test_nis_rejection_is_transactional(self):
        propagator, calibration = setup_propagator()
        t1 = 1_010_001_000
        before = propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        measurement = position_measurement_from_localization(
            localization_pose(
                timestamp=1_010_000_000,
                sequence=11,
                position_xyz=(100.0, 100.0, 100.0),
                position_stddev=(0.001, 0.001, 0.001),
            ),
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        with self.assertRaisesRegex(ValueError, "NIS"):
            propagator.ingest_position_measurement(
                measurement, max_position_nis=9.0
            )
        self.assertEqual(propagator.state, before)

    def test_timestamp_and_per_source_sequence_are_strict(self):
        propagator, calibration = setup_propagator()
        t1 = 1_010_001_000
        propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        with self.assertRaisesRegex(ValueError, "align exactly"):
            propagator.ingest_position_measurement(
                position_measurement_from_localization(
                    localization_pose(timestamp=1_000_000_000, sequence=11),
                    mapping(),
                    expected_target_clock_domain="imu-clock",
                )
            )
        first = position_measurement_from_localization(
            localization_pose(timestamp=1_010_000_000, sequence=11),
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        propagator.ingest_position_measurement(first)
        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            propagator.ingest_position_measurement(first)

    def test_degraded_correction_remains_degraded(self):
        propagator, calibration = setup_propagator()
        t1 = 1_010_001_000
        propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        degraded = position_measurement_from_localization(
            localization_pose(
                timestamp=1_010_000_000,
                sequence=11,
                quality=LocalizationQuality.DEGRADED,
            ),
            mapping(),
            expected_target_clock_domain="imu-clock",
            allow_degraded=True,
        )
        corrected = propagator.ingest_position_measurement(degraded)
        self.assertEqual(corrected.status, "DEGRADED")
        self.assertEqual(corrected.source, "POSITION_CORRECTED")

    def test_rtk_source_has_independent_sequence_domain(self):
        propagator, calibration = setup_propagator()
        t1 = 1_010_001_000
        propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        rtk = MetricPositionMeasurement(
            source=PositionMeasurementSource.RTK,
            body_id="rigid-body",
            map_frame_id="site-map",
            timestamp_ns=t1,
            source_sequence_number=1,
            clock_domain="imu-clock",
            position_world_from_body_m=(1.0, 2.0, 3.0),
            position_stddev_m=(0.02, 0.02, 0.04),
            quality=LocalizationQuality.TRACKING,
            source_revision_sha256="d" * 64,
            measurement_calibration_sha256="e" * 64,
            clock_mapping_sha256=mapping().mapping_sha256,
        )
        corrected = propagator.ingest_position_measurement(rtk)
        self.assertEqual(
            corrected.last_position_measurement_source,
            PositionMeasurementSource.RTK.value,
        )
        self.assertEqual(corrected.last_position_source_sequence_number, 1)

    def test_correction_after_expiry_requires_full_state_reseed(self):
        propagator, calibration = setup_propagator(max_dead_reckoning_ns=5_000_000)
        t1 = 1_010_001_000
        lost = propagator.advance(
            sample(calibration, 1, t1),
            attitude(calibration, t1),
        )
        self.assertEqual(lost.status, "LOST")
        measurement = position_measurement_from_localization(
            localization_pose(timestamp=1_010_000_000, sequence=11),
            mapping(),
            expected_target_clock_domain="imu-clock",
        )
        with self.assertRaisesRegex(ValueError, "full-state reseed"):
            propagator.ingest_position_measurement(measurement)


if __name__ == "__main__":
    unittest.main()
