import unittest
from dataclasses import replace

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


def localization_pose(**changes):
    values = dict(
        source=PoseSource.FIDUCIAL_LOCALIZATION,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id="rigid-body",
        map_frame_id="site-map",
        timestamp_ns=1_000_000_000,
        source_sequence_number=10,
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


def imu_calibration(**changes):
    values = dict(
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
        accel_noise_density_m_s2_sqrt_hz=(0.02, 0.03, 0.04),
        gyro_bias_random_walk_rad_s_per_sqrt_s=(0.001, 0.001, 0.001),
        accel_bias_random_walk_m_s2_per_sqrt_s=(0.002, 0.003, 0.004),
        calibration_method="synthetic-test",
        valid_from_timestamp_ns=0,
        valid_until_timestamp_ns=None,
    )
    values.update(changes)
    return IMUCalibration(**values)


def gravity(**changes):
    values = dict(
        frame_id="site-map",
        gravity_world_m_s2=(0.0, 0.0, -9.80665),
        model_id="constant-site-gravity-v1",
    )
    values.update(changes)
    return GravityModel(**values)


def seed(**changes):
    values = dict(
        observation=localization_pose(),
        mapping=mapping(),
        imu_id="imu-1",
        expected_imu_clock_domain="imu-clock",
        initial_velocity_world_m_s=(0.0, 0.0, 0.0),
        velocity_stddev_m_s=(0.1, 0.1, 0.1),
        residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
        residual_accel_bias_stddev_m_s2=(0.02, 0.02, 0.02),
    )
    values.update(changes)
    return translational_seed_from_localization(**values)


def corrected_sample(calibration, sequence, timestamp, force=(0.0, 0.0, 9.80665)):
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


def attitude(calibration, timestamp, *, variance=1e-4, status=None):
    seed_time = 1_000_001_000
    if status is None:
        status = "TRACKING" if timestamp == seed_time else "DEGRADED"
    lost = status == "LOST"
    return LocalizationIMUAttitudeState(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=None if lost else (0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=None if lost else variance,
        status=status,
        source=(
            "LOCALIZATION_CORRECTED"
            if status == "TRACKING"
            else "IMU_PROPAGATED"
            if status == "DEGRADED"
            else "LOST"
        ),
        last_localization_timestamp_ns=seed_time if status != "LOST" else None,
        last_source_sequence_number=10 if status != "LOST" else None,
        localization_age_ns=(timestamp - seed_time) if status != "LOST" else None,
        clock_mapping_sha256=mapping().mapping_sha256,
        imu_calibration_sha256=calibration.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )


class TranslationSeedTests(unittest.TestCase):
    def test_seed_maps_localization_time_and_preserves_metric_state(self):
        result = seed(
            initial_velocity_world_m_s=(1.0, 0.0, 0.0),
            velocity_stddev_m_s=(0.2, 0.3, 0.4),
            residual_accel_bias_body_m_s2=(0.01, -0.02, 0.03),
        )
        self.assertEqual(result.timestamp_ns, 1_000_001_000)
        self.assertEqual(result.clock_domain, "imu-clock")
        self.assertEqual(result.source_sequence_number, 10)
        self.assertEqual(result.position_world_from_body_m, (1.0, 2.0, 3.0))
        self.assertEqual(result.velocity_world_from_body_m_s, (1.0, 0.0, 0.0))
        self.assertEqual(result.position_stddev_m, (0.02, 0.03, 0.04))
        self.assertEqual(result.localization_source_revision_sha256, MAP_SHA)

    def test_seed_rejects_unqualified_scale_and_clock_mismatch(self):
        unscaled = localization_pose(
            position_unit="map_units",
            metric_scale_qualified=False,
            scale_reference=ScaleReference.NONE,
        )
        with self.assertRaisesRegex(ValueError, "qualified metric"):
            seed(observation=unscaled)
        with self.assertRaisesRegex(ValueError, "TRACKING localization"):
            seed(observation=localization_pose(quality=LocalizationQuality.DEGRADED))
        bad_mapping = fit_clock_mapping(
            [
                ClockAnchor(1_000_000_000, 1_000_001_000, event_id="a"),
                ClockAnchor(2_000_000_000, 2_000_001_000, event_id="b"),
                ClockAnchor(3_000_000_000, 3_000_001_000, event_id="c"),
            ],
            source_clock_domain="other-camera-clock",
            target_clock_domain="imu-clock",
            max_drift_ppm=10,
            max_residual_ns=10,
            max_extrapolation_ns=20_000_000,
            min_anchor_span_ns=100_000_000,
        )
        with self.assertRaisesRegex(ValueError, "source"):
            seed(mapping=bad_mapping)


class GravityModelTests(unittest.TestCase):
    def test_explicit_map_gravity_has_deterministic_revision(self):
        first = gravity()
        second = gravity()
        self.assertEqual(first.revision_sha256, second.revision_sha256)
        self.assertNotEqual(
            first.revision_sha256,
            gravity(gravity_world_m_s2=(0.0, -9.80665, 0.0)).revision_sha256,
        )

    def test_implausible_or_zero_gravity_fails_closed(self):
        for vector in ((0.0, 0.0, 0.0), (0.0, 0.0, -100.0)):
            with self.subTest(vector=vector), self.assertRaisesRegex(ValueError, "terrestrial"):
                gravity(gravity_world_m_s2=vector)


class BoundedTranslationPropagationTests(unittest.TestCase):
    def test_static_specific_force_cancels_explicit_gravity(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(), calibration, gravity()
        )
        t0 = 1_000_001_000
        seeded = propagator.advance(
            corrected_sample(calibration, 0, t0),
            attitude(calibration, t0),
        )
        self.assertEqual(seeded.status, "TRACKING")
        self.assertEqual(seeded.source, "POSITION_SEEDED")

        propagated = propagator.advance(
            corrected_sample(calibration, 1, t0 + 10_000_000),
            attitude(calibration, t0 + 10_000_000),
        )
        self.assertEqual(propagated.status, "DEGRADED")
        np.testing.assert_allclose(
            propagated.position_world_from_body_m,
            (1.0, 2.0, 3.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            propagated.velocity_world_from_body_m_s,
            (0.0, 0.0, 0.0),
            atol=1e-12,
        )

    def test_constant_world_acceleration_integrates_position_and_velocity(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(), calibration, gravity()
        )
        t0 = 1_000_001_000
        force = (1.0, 0.0, 9.80665)
        propagator.advance(
            corrected_sample(calibration, 0, t0, force),
            attitude(calibration, t0),
        )
        state = propagator.advance(
            corrected_sample(calibration, 1, t0 + 10_000_000, force),
            attitude(calibration, t0 + 10_000_000),
        )
        np.testing.assert_allclose(
            state.velocity_world_from_body_m_s,
            (0.01, 0.0, 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            state.position_world_from_body_m,
            (1.00005, 2.0, 3.0),
            atol=1e-12,
        )

    def test_uncertainty_grows_during_dead_reckoning(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(), calibration, gravity()
        )
        t0 = 1_000_001_000
        first = propagator.advance(
            corrected_sample(calibration, 0, t0),
            attitude(calibration, t0),
        )
        second = propagator.advance(
            corrected_sample(calibration, 1, t0 + 10_000_000),
            attitude(calibration, t0 + 10_000_000),
        )
        self.assertTrue(
            all(b > a for a, b in zip(first.position_variance_m2, second.position_variance_m2))
        )
        self.assertTrue(
            all(b > a for a, b in zip(first.velocity_variance_m2_s2, second.velocity_variance_m2_s2))
        )
        self.assertTrue(
            all(
                b > a
                for a, b in zip(
                    first.residual_accel_bias_variance_m2_s4,
                    second.residual_accel_bias_variance_m2_s4,
                )
            )
        )

    def test_dead_reckoning_expiry_hides_translation_and_requires_reseed(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(),
            calibration,
            gravity(),
            max_position_dead_reckoning_ns=15_000_000,
        )
        t0 = 1_000_001_000
        propagator.advance(
            corrected_sample(calibration, 0, t0),
            attitude(calibration, t0),
        )
        propagator.advance(
            corrected_sample(calibration, 1, t0 + 10_000_000),
            attitude(calibration, t0 + 10_000_000),
        )
        lost = propagator.advance(
            corrected_sample(calibration, 2, t0 + 20_000_000),
            attitude(calibration, t0 + 20_000_000),
        )
        self.assertEqual(lost.status, "LOST")
        self.assertIsNone(lost.position_world_from_body_m)
        self.assertIsNone(lost.velocity_world_from_body_m_s)
        with self.assertRaisesRegex(ValueError, "explicit position reseed"):
            propagator.advance(
                corrected_sample(calibration, 3, t0 + 30_000_000),
                attitude(calibration, t0 + 30_000_000),
            )

    def test_nonzero_imu_lever_arm_is_rejected_until_compensation_exists(self):
        calibration = imu_calibration(
            translation_body_from_imu_m=(0.1, 0.0, 0.0)
        )
        with self.assertRaisesRegex(ValueError, "lever arm"):
            BoundedIMUTranslationPropagator(
                seed(), calibration, gravity()
            )

    def test_attitude_loss_uncertainty_and_revision_mismatch_fail_closed(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(), calibration, gravity(), max_attitude_variance_rad2=0.01
        )
        t0 = 1_000_001_000
        with self.assertRaisesRegex(ValueError, "publishable attitude"):
            propagator.advance(
                corrected_sample(calibration, 0, t0),
                attitude(calibration, t0, status="LOST"),
            )
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            propagator.advance(
                corrected_sample(calibration, 0, t0),
                attitude(calibration, t0, variance=0.02),
            )
        wrong = replace(
            attitude(calibration, t0),
            localization_source_revision_sha256="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "source revision"):
            propagator.advance(
                corrected_sample(calibration, 0, t0),
                wrong,
            )

    def test_order_gap_and_failed_advance_do_not_mutate_state(self):
        calibration = imu_calibration()
        propagator = BoundedIMUTranslationPropagator(
            seed(),
            calibration,
            gravity(),
            max_sample_gap_ns=5_000_000,
            max_step_ns=5_000_000,
        )
        t0 = 1_000_001_000
        first = propagator.advance(
            corrected_sample(calibration, 0, t0),
            attitude(calibration, t0),
        )
        with self.assertRaisesRegex(ValueError, "gap"):
            propagator.advance(
                corrected_sample(calibration, 1, t0 + 10_000_000),
                attitude(calibration, t0 + 10_000_000),
            )
        self.assertEqual(propagator.state, first)
        with self.assertRaisesRegex(ValueError, "out of order"):
            # Use an aligned timestamp but duplicate sequence.
            propagator.advance(
                corrected_sample(calibration, 0, t0 + 1_000_000),
                attitude(calibration, t0 + 1_000_000),
            )
        self.assertEqual(propagator.state, first)

    def test_gravity_frame_and_first_sample_alignment_are_strict(self):
        calibration = imu_calibration()
        with self.assertRaisesRegex(ValueError, "gravity frame"):
            BoundedIMUTranslationPropagator(
                seed(),
                calibration,
                gravity(frame_id="other-map"),
            )
        propagator = BoundedIMUTranslationPropagator(
            seed(), calibration, gravity()
        )
        with self.assertRaisesRegex(ValueError, "align exactly"):
            propagator.advance(
                corrected_sample(calibration, 0, 1_000_002_000),
                attitude(calibration, 1_000_002_000),
            )


if __name__ == "__main__":
    unittest.main()
