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
from position_measurement import position_measurement_from_localization
from translational_eskf import ConditionedTranslationalESKF
from translational_propagation import GravityModel, translational_seed_from_localization


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
        position_stddev=(0.05, 0.05, 0.05),
        orientation_stddev_rad=(0.01, 0.01, 0.01),
        map_revision_sha256=MAP_SHA,
        calibration_sha256=LOC_CAL_SHA,
        quality=LocalizationQuality.TRACKING,
        inlier_count=3,
        metric_scale_qualified=True,
        scale_reference=ScaleReference.CALIBRATED_TAG_MAP,
        reprojection_rms_px=0.2,
    )
    values.update(changes)
    return MapPoseObservation(**values)


def imu_calibration(**changes):
    values = dict(
        imu_id="imu-1",
        revision="imu-cal-v1",
        imu_clock_domain="imu-clock",
        rotation_body_from_imu=((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)),
        translation_body_from_imu_m=(0.0,0.0,0.0),
        gyro_bias_rad_s=(0.0,0.0,0.0),
        accel_bias_m_s2=(0.0,0.0,0.0),
        gyro_noise_density_rad_s_sqrt_hz=(0.01,0.01,0.01),
        accel_noise_density_m_s2_sqrt_hz=(0.03,0.03,0.03),
        gyro_bias_random_walk_rad_s_per_sqrt_s=(0.001,0.001,0.001),
        accel_bias_random_walk_m_s2_per_sqrt_s=(0.01,0.01,0.01),
        calibration_method="synthetic-test",
    )
    values.update(changes)
    return IMUCalibration(**values)


def seed(*, velocity_stddev=(0.3,0.3,0.3), bias_stddev=(0.3,0.3,0.3)):
    return translational_seed_from_localization(
        localization_pose(),
        mapping(),
        imu_id="imu-1",
        expected_imu_clock_domain="imu-clock",
        initial_velocity_world_m_s=(0.0,0.0,0.0),
        velocity_stddev_m_s=velocity_stddev,
        residual_accel_bias_body_m_s2=(0.0,0.0,0.0),
        residual_accel_bias_stddev_m_s2=bias_stddev,
    )


def gravity():
    return GravityModel(
        frame_id="site-map",
        gravity_world_m_s2=(0.0,0.0,-9.80665),
        model_id="site-gravity-v1",
    )


def sample(cal, seq, timestamp, force=(0.0,0.0,9.80665)):
    return CorrectedIMUVectors(
        imu_id="imu-1",
        sequence_number=seq,
        sensor_timestamp_ns=timestamp,
        clock_domain="imu-clock",
        calibration_revision=cal.revision,
        calibration_sha256=cal.calibration_sha256,
        gyro_body_rad_s=(0.0,0.0,0.0),
        specific_force_body_m_s2=force,
    )


def attitude(cal, timestamp, *, variance=1e-4):
    t0=1_000_001_000
    return LocalizationIMUAttitudeState(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=(0.0,0.0,0.0,1.0),
        attitude_variance_rad2=variance,
        status="TRACKING" if timestamp==t0 else "DEGRADED",
        source="LOCALIZATION_CORRECTED" if timestamp==t0 else "IMU_PROPAGATED",
        last_localization_timestamp_ns=t0,
        last_source_sequence_number=10,
        localization_age_ns=timestamp-t0,
        clock_mapping_sha256=mapping().mapping_sha256,
        imu_calibration_sha256=cal.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )


def estimator(**kwargs):
    cal=imu_calibration()
    est=ConditionedTranslationalESKF(seed(),cal,gravity(),**kwargs)
    return est,cal


class ConditionedTranslationalESKFTests(unittest.TestCase):
    def test_static_gravity_cancellation_and_full_covariance_growth(self):
        est,cal=estimator()
        t0=1_000_001_000
        first=est.advance(sample(cal,0,t0),attitude(cal,t0))
        second=est.advance(
            sample(cal,1,t0+10_000_000),
            attitude(cal,t0+10_000_000),
        )
        np.testing.assert_allclose(second.position_world_from_body_m,(1.0,2.0,3.0),atol=1e-12)
        np.testing.assert_allclose(second.velocity_world_from_body_m_s,(0.0,0.0,0.0),atol=1e-12)
        P=np.asarray(second.covariance_p_v_bias)
        self.assertEqual(P.shape,(9,9))
        self.assertTrue(np.allclose(P,P.T,atol=1e-12))
        self.assertGreater(float(np.linalg.norm(P[0:3,3:6])),0.0)
        self.assertGreater(float(np.linalg.norm(P[0:3,6:9])),0.0)
        self.assertTrue(np.all(np.asarray(second.position_stddev_m) >= np.asarray(first.position_stddev_m)))

    def test_constant_acceleration_nominal_state_matches_kinematics(self):
        est,cal=estimator()
        t0=1_000_001_000
        force=(1.0,0.0,9.80665)
        est.advance(sample(cal,0,t0,force),attitude(cal,t0))
        state=est.advance(
            sample(cal,1,t0+10_000_000,force),
            attitude(cal,t0+10_000_000),
        )
        np.testing.assert_allclose(state.velocity_world_from_body_m_s,(0.01,0.0,0.0),atol=1e-10)
        np.testing.assert_allclose(state.position_world_from_body_m,(1.00005,2.0,3.0),atol=1e-10)

    def test_position_update_uses_cross_covariance_to_correct_velocity_and_bias(self):
        est,cal=estimator()
        t0=1_000_001_000
        force=(1.0,0.0,9.80665)
        est.advance(sample(cal,0,t0,force),attitude(cal,t0))
        state=None
        for seq in range(1,6):
            ts=t0+seq*10_000_000
            state=est.advance(sample(cal,seq,ts,force),attitude(cal,ts))
        before_v=np.asarray(state.velocity_world_from_body_m_s)
        before_b=np.asarray(state.residual_accel_bias_body_m_s2)
        before_P=np.asarray(state.covariance_p_v_bias)

        observation=localization_pose(
            timestamp=1_050_000_000,
            sequence=11,
            position_xyz=(1.0,2.0,3.0),
            position_stddev=(0.005,0.005,0.005),
        )
        measurement=position_measurement_from_localization(
            observation,mapping(),expected_target_clock_domain="imu-clock"
        )
        corrected=est.ingest_position_measurement(measurement,max_position_nis=100.0)
        after_v=np.asarray(corrected.velocity_world_from_body_m_s)
        after_b=np.asarray(corrected.residual_accel_bias_body_m_s2)
        after_P=np.asarray(corrected.covariance_p_v_bias)

        self.assertGreater(float(np.linalg.norm(after_v-before_v)),1e-8)
        self.assertGreater(float(np.linalg.norm(after_b-before_b)),1e-10)
        self.assertTrue(np.all(np.diag(after_P)[0:3] < np.diag(before_P)[0:3]))
        self.assertTrue(np.any(np.diag(after_P)[3:6] < np.diag(before_P)[3:6]))
        self.assertTrue(np.any(np.diag(after_P)[6:9] < np.diag(before_P)[6:9]))
        self.assertEqual(corrected.status,"TRACKING")
        self.assertEqual(corrected.position_age_ns,0)

    def test_nis_rejection_is_transactional(self):
        est,cal=estimator()
        t0=1_000_001_000
        est.advance(sample(cal,0,t0),attitude(cal,t0))
        before=est.advance(sample(cal,1,t0+10_000_000),attitude(cal,t0+10_000_000))
        measurement=position_measurement_from_localization(
            localization_pose(
                timestamp=1_010_000_000,
                sequence=11,
                position_xyz=(100.0,100.0,100.0),
                position_stddev=(0.001,0.001,0.001),
            ),
            mapping(),expected_target_clock_domain="imu-clock"
        )
        with self.assertRaisesRegex(ValueError,"NIS"):
            est.ingest_position_measurement(measurement,max_position_nis=9.0)
        self.assertEqual(est.state,before)

    def test_position_update_bias_limit_is_transactional(self):
        est,cal=estimator(max_residual_accel_bias_m_s2=1e-8)
        t0=1_000_001_000
        force=(1.0,0.0,9.80665)
        est.advance(sample(cal,0,t0,force),attitude(cal,t0))
        for seq in range(1,6):
            ts=t0+seq*10_000_000
            est.advance(sample(cal,seq,ts,force),attitude(cal,ts))
        before=est.state
        measurement=position_measurement_from_localization(
            localization_pose(
                timestamp=1_050_000_000,
                sequence=11,
                position_xyz=(1.0,2.0,3.0),
                position_stddev=(0.005,0.005,0.005),
            ),
            mapping(),expected_target_clock_domain="imu-clock"
        )
        with self.assertRaisesRegex(ValueError,"bias limit"):
            est.ingest_position_measurement(measurement,max_position_nis=100.0)
        self.assertEqual(est.state,before)

    def test_dead_reckoning_expiry_hides_state(self):
        est,cal=estimator(max_position_dead_reckoning_ns=15_000_000)
        t0=1_000_001_000
        est.advance(sample(cal,0,t0),attitude(cal,t0))
        est.advance(sample(cal,1,t0+10_000_000),attitude(cal,t0+10_000_000))
        lost=est.advance(sample(cal,2,t0+20_000_000),attitude(cal,t0+20_000_000))
        self.assertEqual(lost.status,"LOST")
        self.assertIsNone(lost.position_world_from_body_m)
        self.assertIsNone(lost.velocity_world_from_body_m_s)
        self.assertIsNone(lost.covariance_p_v_bias)

    def test_current_time_and_map_frame_are_strict(self):
        est,cal=estimator()
        t0=1_000_001_000
        est.advance(sample(cal,0,t0),attitude(cal,t0))
        est.advance(sample(cal,1,t0+10_000_000),attitude(cal,t0+10_000_000))
        old=position_measurement_from_localization(
            localization_pose(timestamp=1_000_000_000,sequence=11),
            mapping(),expected_target_clock_domain="imu-clock"
        )
        with self.assertRaisesRegex(ValueError,"align exactly"):
            est.ingest_position_measurement(old)
        with self.assertRaisesRegex(ValueError,"map frame"):
            est.advance(
                sample(cal,2,t0+20_000_000),
                LocalizationIMUAttitudeState(
                    **{
                        **attitude(cal,t0+20_000_000).__dict__,
                        "map_frame_id":"other-map",
                    }
                ),
            )

    def test_nonzero_imu_lever_arm_remains_blocked(self):
        cal=imu_calibration(translation_body_from_imu_m=(0.1,0.0,0.0))
        with self.assertRaisesRegex(ValueError,"lever arm"):
            ConditionedTranslationalESKF(seed(),cal,gravity())


if __name__=="__main__":
    unittest.main()
