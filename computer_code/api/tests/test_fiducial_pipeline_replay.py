import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialObservation, FiducialTagMap, TagAnchor
from delayed_position_replay import BoundedDelayedPositionReplay
from imu_calibration import CorrectedIMUVectors, IMUCalibration
from imu_contract import ClockAnchor, fit_clock_mapping
from localization_imu_fusion import LocalizationIMUAttitudeState
from outdoor_sensor_contract import RTKSample, RTKSolutionStatus
from position_measurement import position_measurement_from_localization
from rtk_coordinates import ENUOrigin, enu_to_ecef
from rtk_position_measurement import (
    RTKPositionCalibration,
    position_measurement_from_rtk,
    position_source_contract_from_rtk,
)
from translational_propagation import GravityModel, translational_seed_from_localization
from fiducial_localization import (
    CameraBodyExtrinsic,
    FiducialUncertaintyModel,
    map_body_observation_from_fiducials,
)
from fiducial_multitag_pose import solve_multi_tag_camera_pose
from fiducial_pose import (
    FiducialCameraCalibration,
    solve_single_tag_pose,
)
from localization_contract import LocalizationQuality


CAMERA_ID = 2
CAMERA_SERIAL = "CAM-02"
CALIBRATION_SHA = "a" * 64
T0 = 8_000_000
K = np.asarray(((500.0, 0.0, 320.0), (0.0, 510.0, 240.0), (0.0, 0.0, 1.0)))
SIZE_M = 0.16


def make_map(*, moved_tag_id=None, moved_by=(0.0, 0.0, 0.0)):
    anchors = []
    positions = {
        1: (0.0, 0.0, 0.0),
        2: (0.6, 0.0, 0.0),
        3: (0.0, 0.6, 0.0),
        4: (0.6, 0.6, 0.0),
    }
    for tag_id, position in positions.items():
        xyz = np.asarray(position, dtype=float)
        if tag_id == moved_tag_id:
            xyz = xyz + np.asarray(moved_by, dtype=float)
        anchors.append(TagAnchor(
            tag_family="tag36h11",
            tag_id=tag_id,
            map_frame_id="site-map",
            position_xyz_m=tuple(xyz),
            orientation_map_from_tag_xyzw=(0.0, 0.0, 0.0, 1.0),
            physical_size_m=SIZE_M,
            position_stddev_m=(0.002, 0.002, 0.003),
            orientation_stddev_rad=(0.003, 0.003, 0.004),
            reference_method="surveyed",
        ))
    return FiducialTagMap("indoor_dense", "site-map", tuple(anchors))


def camera_calibration():
    return FiducialCameraCalibration(
        camera_id=CAMERA_ID,
        camera_serial=CAMERA_SERIAL,
        intrinsic_matrix=tuple(tuple(float(v) for v in row) for row in K),
        distortion_model="none",
        distortion_coefficients=(),
        calibration_sha256=CALIBRATION_SHA,
    )


def tag_corners(size=SIZE_M):
    h = size * 0.5
    return np.asarray(((-h, h, 0.0), (h, h, 0.0), (h, -h, 0.0), (-h, -h, 0.0)))


def render_observation(anchor, active_map, camera_position, rotation_map_from_camera, *, timestamp_ns=T0, noise_px=0.0, tag_id_override=None):
    rotation_map_from_tag = Rotation.from_quat(anchor.orientation_map_from_tag_xyzw).as_matrix()
    translation_map_from_tag = np.asarray(anchor.position_xyz_m, dtype=float)
    rotation_camera_from_tag = rotation_map_from_camera.T @ rotation_map_from_tag
    translation_camera_from_tag = rotation_map_from_camera.T @ (
        translation_map_from_tag - camera_position
    )
    points_camera = (rotation_camera_from_tag @ tag_corners().T).T + translation_camera_from_tag
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("synthetic tag must be in front of camera")
    normalized = points_camera[:, :2] / points_camera[:, 2, None]
    homogeneous = (K @ np.column_stack((normalized, np.ones(4))).T).T
    corners = homogeneous[:, :2] / homogeneous[:, 2, None]
    if noise_px:
        # Fixed deterministic pattern; no random seed or nondeterministic test data.
        corners = corners + np.asarray(((noise_px, -noise_px), (-noise_px, noise_px),
                                        (noise_px, noise_px), (-noise_px, -noise_px)))
    return FiducialObservation(
        tag_family="tag36h11",
        tag_id=anchor.tag_id if tag_id_override is None else tag_id_override,
        camera_id=CAMERA_ID,
        camera_serial=CAMERA_SERIAL,
        frame_id=900 + (timestamp_ns - T0) // 10_000_000,
        trigger_id=42 + (timestamp_ns - T0) // 10_000_000,
        exposure_timestamp_ns=timestamp_ns,
        clock_domain="capture-ptp-A",
        image_corners_px=tuple(tuple(float(v) for v in p) for p in corners),
        corrected_bit_errors=0,
        corner_rms_px=0.2,
        calibration_sha256=CALIBRATION_SHA,
        map_revision_sha256=active_map.revision_sha256,
    )


def localize_frame(active_map, *, timestamp_ns=T0, camera_position=(0.3, 0.25, 1.4), noise_px=0.0, map_override=None, wrong_id=None):
    # Camera looks down onto a planar tag field. The body pose is known only to
    # this synthetic generator; the localization algorithm receives image corners.
    camera_position = np.asarray(camera_position, dtype=float)
    rotation_map_from_camera = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    calibration = camera_calibration()
    observations = []
    for anchor in active_map.tags[:3]:
        observation = render_observation(
            anchor, active_map, camera_position, rotation_map_from_camera,
            timestamp_ns=timestamp_ns,
            noise_px=noise_px,
            tag_id_override=wrong_id if wrong_id is not None and anchor.tag_id == 3 else None,
        )
        observations.append(solve_single_tag_pose(
            observation,
            active_map if map_override is None else map_override,
            calibration,
            ambiguity_rms_tolerance_px=0.05,
        ))
    camera_pose = solve_multi_tag_camera_pose(observations, active_map)
    body_observation = map_body_observation_from_fiducials(
        camera_pose,
        active_map,
        CameraBodyExtrinsic(
            camera_id=CAMERA_ID,
            camera_serial=CAMERA_SERIAL,
            camera_frame_id="camera-frame",
            body_frame_id="rigid-body",
            translation_body_from_camera_m=(0.0, 0.0, 0.0),
            orientation_body_from_camera_xyzw=(0.0, 0.0, 0.0, 1.0),
            reference_method="calibrated",
        ),
        FiducialUncertaintyModel(
            model_id="pipeline-test-v1",
            base_position_stddev_m=(0.01, 0.01, 0.015),
            base_orientation_stddev_rad=(0.01, 0.01, 0.01),
            translation_residual_gain=1.5,
            orientation_residual_gain=1.5,
        ),
    )
    return camera_pose, body_observation


class FiducialPipelineReplayTests(unittest.TestCase):
    def test_image_corners_recover_metric_body_pose_end_to_end(self):
        active_map = make_map()
        camera_pose, body = localize_frame(active_map)
        self.assertEqual(camera_pose.inlier_tag_keys, (
            ("tag36h11", 1), ("tag36h11", 2), ("tag36h11", 3)
        ))
        self.assertEqual(body.quality, LocalizationQuality.TRACKING)
        np.testing.assert_allclose(body.position_xyz, (0.3, 0.25, 1.4), atol=1e-5)
        actual_rotation = Rotation.from_quat(body.orientation_xyzw).as_matrix()
        expected_rotation = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        angle = Rotation.from_matrix(expected_rotation.T @ actual_rotation).magnitude()
        self.assertLess(angle, 1e-5)
        self.assertTrue(body.metric_scale_qualified)

    def test_deterministic_corner_noise_is_bounded_and_repeatable(self):
        active_map = make_map()
        first_pose, first_body = localize_frame(active_map, noise_px=0.2)
        second_pose, second_body = localize_frame(active_map, noise_px=0.2)
        self.assertEqual(first_pose, second_pose)
        self.assertEqual(first_body, second_body)
        self.assertLess(first_pose.mean_single_tag_reprojection_rms_px, 2.0)

    def test_wrong_tag_id_fails_closed(self):
        active_map = make_map()
        with self.assertRaises(ValueError):
            localize_frame(active_map, wrong_id=99)

    def test_stale_tag_map_revision_is_rejected(self):
        active_map = make_map()
        moved_map = make_map(moved_tag_id=2, moved_by=(0.1, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "map revision"):
            localize_frame(active_map, map_override=moved_map)

    def test_image_to_body_pose_to_delayed_position_replay(self):
        active_map = make_map()
        _, initial_pose = localize_frame(active_map)
        camera_mapping = fit_clock_mapping(
            [
                ClockAnchor(T0, T0 + 1_000, event_id="camera-sync-1"),
                ClockAnchor(T0 + 1_000_000_000, T0 + 1_000_001_000, event_id="camera-sync-2"),
                ClockAnchor(T0 + 2_000_000_000, T0 + 2_000_001_000, event_id="camera-sync-3"),
            ],
            source_clock_domain="capture-ptp-A",
            target_clock_domain="imu-clock",
            max_drift_ppm=10,
            max_residual_ns=10,
            max_extrapolation_ns=100_000_000,
            min_anchor_span_ns=100_000_000,
        )
        seed = translational_seed_from_localization(
            initial_pose,
            camera_mapping,
            imu_id="imu-1",
            expected_imu_clock_domain="imu-clock",
            initial_velocity_world_m_s=(0.0, 0.0, 0.0),
            velocity_stddev_m_s=(0.1, 0.1, 0.1),
            residual_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
            residual_accel_bias_stddev_m_s2=(0.02, 0.02, 0.02),
        )
        imu_cal = IMUCalibration(
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
            calibration_method="synthetic-pipeline-test",
        )
        rtk_mapping = fit_clock_mapping(
            [
                ClockAnchor(T0, T0 + 1_000, event_id="rtk-sync-1"),
                ClockAnchor(T0 + 1_000_000_000, T0 + 1_000_001_000, event_id="rtk-sync-2"),
                ClockAnchor(T0 + 2_000_000_000, T0 + 2_000_001_000, event_id="rtk-sync-3"),
            ],
            source_clock_domain="rtk-clock",
            target_clock_domain="imu-clock",
            max_drift_ppm=10,
            max_residual_ns=10,
            max_extrapolation_ns=100_000_000,
            min_anchor_span_ns=100_000_000,
        )
        rtk_origin = ENUOrigin(0.0, 0.0, 0.0)
        rtk_cal = RTKPositionCalibration(
            receiver_id="rtk-1",
            body_id="rigid-body",
            map_frame_id="site-map",
            enu_origin=rtk_origin,
            antenna_lever_arm_body_m=(0.0, 0.0, 0.0),
            horizontal_accuracy_to_axis_stddev_scale=1.0,
            vertical_accuracy_to_stddev_scale=1.0,
            max_correction_age_ms=1000.0,
            max_horizontal_accuracy_m=0.1,
            max_vertical_accuracy_m=0.2,
            min_satellites=8,
        )
        rtk_contract = position_source_contract_from_rtk(
            rtk_cal, rtk_mapping, expected_target_clock_domain="imu-clock"
        )
        owner = BoundedDelayedPositionReplay(
            seed,
            imu_cal,
            GravityModel(
                frame_id="site-map",
                gravity_world_m_s2=(0.0, 0.0, -9.80665),
                model_id="site-gravity-v1",
            ),
            max_history_frames=32,
            max_history_ns=200_000_000,
            position_source_contracts=(rtk_contract,),
        )
        seed_timestamp = camera_mapping.map_timestamp(T0)

        def attitude_at(timestamp_ns, *, tracking=False, sequence=900):
            return LocalizationIMUAttitudeState(
                body_id="rigid-body",
                map_frame_id="site-map",
                imu_id="imu-1",
                timestamp_ns=timestamp_ns,
                clock_domain="imu-clock",
                quaternion_world_from_body_xyzw=initial_pose.orientation_xyzw,
                attitude_variance_rad2=max(initial_pose.orientation_stddev_rad) ** 2,
                status="TRACKING" if tracking else "DEGRADED",
                source="LOCALIZATION_CORRECTED" if tracking else "IMU_PROPAGATED",
                last_localization_timestamp_ns=seed_timestamp,
                last_source_sequence_number=sequence,
                localization_age_ns=timestamp_ns - seed_timestamp,
                clock_mapping_sha256=camera_mapping.mapping_sha256,
                imu_calibration_sha256=imu_cal.calibration_sha256,
                localization_source_revision_sha256=initial_pose.map_revision_sha256,
                localization_calibration_sha256=initial_pose.calibration_sha256,
            )

        for sequence in range(6):
            raw_timestamp = T0 + sequence * 10_000_000
            timestamp = camera_mapping.map_timestamp(raw_timestamp)
            sample = CorrectedIMUVectors(
                imu_id="imu-1",
                sequence_number=sequence,
                sensor_timestamp_ns=timestamp,
                clock_domain="imu-clock",
                calibration_revision=imu_cal.revision,
                calibration_sha256=imu_cal.calibration_sha256,
                gyro_body_rad_s=(0.0, 0.0, 0.0),
                specific_force_body_m_s2=(0.0, 0.0, 9.80665),
            )
            owner.ingest_imu(
                sample,
                attitude_at(timestamp, tracking=(sequence == 0)),
            )

        # Build a real synthetic image observation at an epoch between IMU
        # frames; only the image corners reach the PnP/consensus pipeline.
        _, delayed_pose = localize_frame(
            active_map,
            timestamp_ns=T0 + 25_000_000,
            camera_position=(0.31, 0.25, 1.4),
            noise_px=0.1,
        )
        update = position_measurement_from_localization(
            delayed_pose,
            camera_mapping,
            expected_target_clock_domain="imu-clock",
        )
        self.assertEqual(update.timestamp_ns, camera_mapping.map_timestamp(T0 + 25_000_000))
        rtk_timestamp = T0 + 25_000_000
        rtk_sample = RTKSample(
            receiver_id="rtk-1",
            sequence_number=7,
            solution_timestamp_ns=rtk_timestamp,
            clock_domain="rtk-clock",
            status=RTKSolutionStatus.FIXED,
            position_ecef_m=enu_to_ecef((0.31, 0.25, 1.4), rtk_origin),
            horizontal_accuracy_m=0.02,
            vertical_accuracy_m=0.03,
            correction_age_ms=100.0,
            satellites_used=12,
            datum="WGS84",
            antenna_reference_point="antenna_phase_center",
        )
        mapped_epoch = rtk_mapping.map_timestamp(rtk_timestamp)
        rtk_attitude = LocalizationIMUAttitudeState(
            body_id="rigid-body",
            map_frame_id="site-map",
            imu_id="imu-1",
            timestamp_ns=mapped_epoch,
            clock_domain="imu-clock",
            quaternion_world_from_body_xyzw=delayed_pose.orientation_xyzw,
            attitude_variance_rad2=max(delayed_pose.orientation_stddev_rad) ** 2,
            status="TRACKING",
            source="LOCALIZATION_CORRECTED",
            last_localization_timestamp_ns=mapped_epoch,
            last_source_sequence_number=delayed_pose.source_sequence_number,
            localization_age_ns=0,
            clock_mapping_sha256=camera_mapping.mapping_sha256,
            imu_calibration_sha256=imu_cal.calibration_sha256,
            localization_source_revision_sha256=delayed_pose.map_revision_sha256,
            localization_calibration_sha256=delayed_pose.calibration_sha256,
        )
        rtk_update = position_measurement_from_rtk(
            rtk_sample, rtk_mapping, rtk_cal, rtk_attitude
        )
        self.assertEqual(rtk_update.timestamp_ns, update.timestamp_ns)
        owner.ingest_position_measurement(rtk_update, max_position_nis=100.0)
        final_state = owner.ingest_position_measurement(update, max_position_nis=100.0)
        self.assertEqual(final_state.timestamp_ns, camera_mapping.map_timestamp(T0 + 50_000_000))
        self.assertEqual(final_state.last_position_update_timestamp_ns, update.timestamp_ns)
        self.assertEqual(final_state.last_position_measurement_source, "RTK")
        self.assertIsNotNone(final_state.position_world_from_body_m)

    def test_anchor_perturbation_changes_map_revision(self):
        nominal = make_map()
        perturbed = make_map(moved_tag_id=2, moved_by=(0.005, 0.0, 0.0))
        self.assertNotEqual(nominal.revision_sha256, perturbed.revision_sha256)


if __name__ == "__main__":
    unittest.main()
