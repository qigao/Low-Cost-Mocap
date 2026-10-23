import unittest

from delayed_position_replay import BoundedDelayedPositionReplay
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
from outdoor_sensor_contract import RTKSample, RTKSolutionStatus
from position_measurement import (
    PositionMeasurementSource,
    PositionSourceContract,
    position_measurement_from_localization,
)
from rtk_coordinates import ENUOrigin, enu_to_ecef
from rtk_position_measurement import (
    RTKPositionCalibration,
    position_measurement_from_rtk,
    position_source_contract_from_rtk,
)
from translational_propagation import (
    GravityModel,
    translational_seed_from_localization,
)


T0 = 1_000_000_000
MAP_SHA = "b" * 64
LOC_CAL_SHA = "c" * 64


def clock_mapping(source_domain):
    # Five ppm drift plus a 1 us epoch offset, independently pinned for each
    # provider clock. Both mappings target the same estimator clock.
    return fit_clock_mapping(
        [
            ClockAnchor(T0, T0 + 1_000, event_id=f"{source_domain}-sync-1"),
            ClockAnchor(T0 + 1_000_000_000, T0 + 1_000_001_005,
                        event_id=f"{source_domain}-sync-2"),
            ClockAnchor(T0 + 2_000_000_000, T0 + 2_000_001_010,
                        event_id=f"{source_domain}-sync-3"),
        ],
        source_clock_domain=source_domain,
        target_clock_domain="imu-clock",
        max_drift_ppm=10,
        max_residual_ns=10,
        max_extrapolation_ns=100_000_000,
        min_anchor_span_ns=100_000_000,
    )


def tag_pose(timestamp_ns=T0, sequence=10, position=(1.0, 2.0, 3.0)):
    return MapPoseObservation(
        source=PoseSource.FIDUCIAL_LOCALIZATION,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id="rigid-body",
        map_frame_id="site-map",
        timestamp_ns=timestamp_ns,
        source_sequence_number=sequence,
        clock_domain="camera-clock",
        position_xyz=position,
        position_unit="meters",
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_stddev=(0.02, 0.02, 0.03),
        orientation_stddev_rad=(0.01, 0.01, 0.01),
        map_revision_sha256=MAP_SHA,
        calibration_sha256=LOC_CAL_SHA,
        quality=LocalizationQuality.TRACKING,
        inlier_count=3,
        metric_scale_qualified=True,
        scale_reference=ScaleReference.CALIBRATED_TAG_MAP,
        relocalized=False,
        reprojection_rms_px=0.25,
    )


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
        calibration_method="synthetic-integration-test",
    )


def gravity():
    return GravityModel(
        frame_id="site-map",
        gravity_world_m_s2=(0.0, 0.0, -9.80665),
        model_id="site-gravity-v1",
    )


def imu_sample(calibration, sequence, timestamp_ns):
    return CorrectedIMUVectors(
        imu_id="imu-1",
        sequence_number=sequence,
        sensor_timestamp_ns=timestamp_ns,
        clock_domain="imu-clock",
        calibration_revision=calibration.revision,
        calibration_sha256=calibration.calibration_sha256,
        gyro_body_rad_s=(0.0, 0.0, 0.0),
        specific_force_body_m_s2=(0.0, 0.0, 9.80665),
    )


def attitude_at(calibration, localization_mapping, timestamp_ns):
    seed_timestamp = localization_mapping.map_timestamp(T0)
    is_seed = timestamp_ns == seed_timestamp
    return LocalizationIMUAttitudeState(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        timestamp_ns=timestamp_ns,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=(0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=0.0004,
        status="TRACKING" if is_seed else "DEGRADED",
        source="LOCALIZATION_CORRECTED" if is_seed else "IMU_PROPAGATED",
        last_localization_timestamp_ns=seed_timestamp,
        last_source_sequence_number=10,
        localization_age_ns=timestamp_ns - seed_timestamp,
        clock_mapping_sha256=localization_mapping.mapping_sha256,
        imu_calibration_sha256=calibration.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )


def rtk_calibration():
    return RTKPositionCalibration(
        receiver_id="rtk-1",
        body_id="rigid-body",
        map_frame_id="site-map",
        enu_origin=ENUOrigin(0.0, 0.0, 0.0),
        antenna_lever_arm_body_m=(0.0, 0.0, 0.0),
        horizontal_accuracy_to_axis_stddev_scale=1.0,
        vertical_accuracy_to_stddev_scale=1.0,
        max_correction_age_ms=1000.0,
        max_horizontal_accuracy_m=0.1,
        max_vertical_accuracy_m=0.2,
        min_satellites=8,
    )


def rtk_sample():
    calibration = rtk_calibration()
    return RTKSample(
        receiver_id="rtk-1",
        sequence_number=7,
        solution_timestamp_ns=T0 + 25_000_000,
        clock_domain="rtk-clock",
        status=RTKSolutionStatus.FIXED,
        position_ecef_m=enu_to_ecef((1.0002, 2.0, 3.0), calibration.enu_origin),
        horizontal_accuracy_m=0.02,
        vertical_accuracy_m=0.04,
        correction_age_ms=100.0,
        satellites_used=12,
        datum="WGS84",
        antenna_reference_point="antenna_phase_center",
    )


def build_replay():
    camera_clock = clock_mapping("camera-clock")
    rtk_clock = clock_mapping("rtk-clock")
    initial_pose = tag_pose()
    seed = translational_seed_from_localization(
        initial_pose,
        camera_clock,
        imu_id="imu-1",
        expected_imu_clock_domain="imu-clock",
        initial_velocity_world_m_s=(0.0, 0.0, 0.0),
        velocity_stddev_m_s=(0.1, 0.1, 0.1),
        residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
        residual_accel_bias_stddev_m_s2=(0.02, 0.02, 0.02),
    )
    calibration = imu_calibration()
    rtk_cal = rtk_calibration()
    localization_contract = PositionSourceContract(
        source=PositionMeasurementSource.LOCALIZATION,
        source_revision_sha256=MAP_SHA,
        measurement_calibration_sha256=LOC_CAL_SHA,
        clock_mapping_sha256=camera_clock.mapping_sha256,
    )
    rtk_contract = position_source_contract_from_rtk(
        rtk_cal, rtk_clock, expected_target_clock_domain="imu-clock"
    )
    owner = BoundedDelayedPositionReplay(
        seed,
        calibration,
        gravity(),
        max_history_frames=32,
        max_history_ns=200_000_000,
        position_source_contracts=(localization_contract, rtk_contract),
    )
    for sequence in range(6):
        source_timestamp = T0 + sequence * 10_000_000
        timestamp_ns = camera_clock.map_timestamp(source_timestamp)
        owner.ingest_imu(
            imu_sample(calibration, sequence, timestamp_ns),
            attitude_at(calibration, camera_clock, timestamp_ns),
        )
    return owner, calibration, camera_clock, rtk_clock, rtk_cal


class OutdoorLocalizationReplayTests(unittest.TestCase):
    def test_tag_and_rtk_adapters_replay_same_interior_epoch_arrival_independently(self):
        tag_observation = tag_pose(
            timestamp_ns=T0 + 25_000_000,
            sequence=11,
            position=(1.0001, 2.0, 3.0),
        )
        owner_a, calibration_a, camera_clock_a, rtk_clock_a, rtk_cal_a = build_replay()
        tag_update_a = position_measurement_from_localization(
            tag_observation,
            camera_clock_a,
            expected_target_clock_domain="imu-clock",
        )
        rtk_update_a = position_measurement_from_rtk(
            rtk_sample(),
            rtk_clock_a,
            rtk_cal_a,
            attitude_at(
                calibration_a,
                camera_clock_a,
                rtk_clock_a.map_timestamp(T0 + 25_000_000),
            ),
        )
        self.assertEqual(tag_update_a.timestamp_ns, rtk_update_a.timestamp_ns)
        owner_a.ingest_position_measurement(rtk_update_a, max_position_nis=100.0)
        state_a = owner_a.ingest_position_measurement(
            tag_update_a, max_position_nis=100.0
        )

        owner_b, calibration_b, camera_clock_b, rtk_clock_b, rtk_cal_b = build_replay()
        tag_update_b = position_measurement_from_localization(
            tag_observation,
            camera_clock_b,
            expected_target_clock_domain="imu-clock",
        )
        rtk_update_b = position_measurement_from_rtk(
            rtk_sample(),
            rtk_clock_b,
            rtk_cal_b,
            attitude_at(
                calibration_b,
                camera_clock_b,
                rtk_clock_b.map_timestamp(T0 + 25_000_000),
            ),
        )
        owner_b.ingest_position_measurement(tag_update_b, max_position_nis=100.0)
        state_b = owner_b.ingest_position_measurement(
            rtk_update_b, max_position_nis=100.0
        )

        self.assertEqual(state_a, state_b)
        self.assertEqual(state_a.timestamp_ns, camera_clock_a.map_timestamp(T0 + 50_000_000))
        self.assertEqual(state_a.last_position_update_timestamp_ns, tag_update_a.timestamp_ns)
        self.assertEqual(state_a.last_position_measurement_source, "RTK")


if __name__ == "__main__":
    unittest.main()
