"""Single deterministic software fault matrix spanning vision, clocks, RTK and replay."""
import hashlib
import json
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_multitag_pose import solve_multi_tag_camera_pose
from fiducial_pose import solve_single_tag_pose
from localization_contract import LocalizationQuality
from outdoor_sensor_contract import RTKSolutionStatus
from rtk_coordinates import enu_to_ecef
from rtk_position_measurement import position_measurement_from_rtk
from test_fiducial_pipeline_replay import (
    camera_calibration,
    localize_frame,
    make_map,
    render_observation,
)
from test_localization_imu_fusion import observation as attitude_observation, setup_pipeline
from test_outdoor_localization_replay import (
    T0,
    attitude_at,
    build_replay,
    clock_mapping,
    imu_sample,
    rtk_sample,
)


class IntegratedLocalizationFaultMatrixTests(unittest.TestCase):
    def test_combined_fault_matrix_is_deterministic_and_fails_closed(self):
        matrix = []
        active_map = make_map()

        # Raw synthetic image corners -> single-tag PnP -> multi-tag consensus
        # -> camera/body pose, with repeatable image noise.
        _, nominal_pose = localize_frame(active_map)
        self.assertEqual(nominal_pose.quality, LocalizationQuality.TRACKING)
        matrix.append(("nominal_image_to_pose", "accepted"))

        _, noisy_a = localize_frame(active_map, noise_px=0.2)
        _, noisy_b = localize_frame(active_map, noise_px=0.2)
        self.assertEqual(noisy_a, noisy_b)
        matrix.append(("deterministic_corner_noise", "repeatable"))

        with self.assertRaises(ValueError):
            localize_frame(active_map, wrong_id=99)
        matrix.append(("unknown_tag_id", "rejected"))

        moved_map = make_map(moved_tag_id=2, moved_by=(0.01, 0.0, 0.0))
        self.assertNotEqual(active_map.revision_sha256, moved_map.revision_sha256)
        with self.assertRaisesRegex(ValueError, "map revision"):
            localize_frame(active_map, map_override=moved_map)

        # Survey-bias experiment: image corners are generated from the true
        # map, but the solver is supplied a consistently shifted surveyed map.
        # The map digest is internally consistent, so this is a bias experiment,
        # not a stale-revision rejection test.
        biased_map = make_map(global_offset=(0.01, 0.0, 0.0))
        camera_position = np.asarray((0.3, 0.25, 1.4))
        rotation_map_from_camera = Rotation.from_euler(
            "x", 180.0, degrees=True
        ).as_matrix()
        calibration = camera_calibration()
        biased_measurements = []
        for anchor in active_map.tags[:3]:
            raw = render_observation(
                anchor, biased_map, camera_position, rotation_map_from_camera
            )
            biased_measurements.append(
                solve_single_tag_pose(raw, biased_map, calibration, ambiguity_rms_tolerance_px=0.05)
            )
        biased_pose = solve_multi_tag_camera_pose(biased_measurements, biased_map)
        self.assertAlmostEqual(
            biased_pose.position_map_from_camera_m[0] - 0.3,
            0.01,
            delta=0.002,
        )
        matrix.append(("stale_map_revision", "rejected"))
        matrix.append(("global_anchor_survey_bias_10mm", "10mm_global_pose_bias_detected"))

        # Verify a declared 5 ppm camera clock drift remains mapped to the
        # estimator clock by its versioned mapping rather than host arrival time.
        camera_clock = clock_mapping("matrix-camera-clock")
        self.assertEqual(
            camera_clock.map_timestamp(T0 + 1_000_000_000),
            T0 + 1_000_001_005,
        )
        matrix.append(("clock_offset_and_5ppm_drift", "mapped"))

        # A FIXED RTK update is accepted at acquisition time; a subsequent
        # FLOAT solution is rejected before it can mutate the estimator.
        owner, calibration, camera_clock, rtk_clock, rtk_cal = build_replay()
        fixed_template = rtk_sample()
        fixed_epoch = T0 + 25_000_000
        fixed = replace(
            fixed_template,
            sequence_number=7,
            solution_timestamp_ns=fixed_epoch,
            status=RTKSolutionStatus.FIXED,
            position_ecef_m=enu_to_ecef((1.0001, 2.0, 3.0), rtk_cal.enu_origin),
        )
        fixed_attitude = attitude_at(
            calibration, camera_clock, rtk_clock.map_timestamp(fixed_epoch)
        )
        fixed_update = position_measurement_from_rtk(
            fixed, rtk_clock, rtk_cal, fixed_attitude
        )
        accepted = owner.ingest_position_measurement(fixed_update, max_position_nis=100.0)
        self.assertEqual(accepted.last_position_measurement_source, "RTK")
        before_float = owner.state
        float_epoch = T0 + 35_000_000
        floating = replace(
            fixed,
            sequence_number=8,
            solution_timestamp_ns=float_epoch,
            status=RTKSolutionStatus.FLOAT,
        )
        with self.assertRaises(ValueError):
            position_measurement_from_rtk(
                floating,
                rtk_clock,
                rtk_cal,
                attitude_at(
                    calibration, camera_clock, rtk_clock.map_timestamp(float_epoch)
                ),
            )
        self.assertEqual(owner.state, before_float)
        matrix.append(("RTK_FIXED_to_FLOAT", "fixed_accepted_float_rejected"))

        # A gap beyond the configured IMU sample horizon must leave state and
        # retained history unchanged.
        gap_owner, gap_cal, gap_clock, _, _ = build_replay()
        before_gap = gap_owner.state
        before_count = gap_owner.retained_frame_count
        gap_timestamp = gap_clock.map_timestamp(T0 + 80_000_000)
        with self.assertRaises(ValueError):
            gap_owner.ingest_imu(
                imu_sample(gap_cal, 6, gap_timestamp),
                attitude_at(gap_cal, gap_clock, gap_timestamp),
            )
        self.assertEqual(gap_owner.state, before_gap)
        self.assertEqual(gap_owner.retained_frame_count, before_count)
        matrix.append(("IMU_gap", "rejected_transactionally"))

        # Prolonged absolute-position loss expires translation. A delayed FIXED
        # RTK observation then reacquires at its original epoch and replays to
        # the latest retained IMU frame.
        recovery, rec_cal, rec_camera_clock, rec_rtk_clock, rec_rtk_cal = build_replay()
        for sequence in range(6, 12):
            timestamp = rec_camera_clock.map_timestamp(T0 + sequence * 10_000_000)
            lost_state = recovery.ingest_imu(
                imu_sample(rec_cal, sequence, timestamp),
                attitude_at(rec_cal, rec_camera_clock, timestamp),
            )
        self.assertEqual(lost_state.status, "LOST")
        self.assertIsNone(lost_state.position_world_from_body_m)
        recovery_epoch = T0 + 95_000_000
        template = rtk_sample()
        delayed_fixed = replace(
            template,
            sequence_number=8,
            solution_timestamp_ns=recovery_epoch,
            status=RTKSolutionStatus.FIXED,
            position_ecef_m=enu_to_ecef((1.0001, 2.0, 3.0), rec_rtk_cal.enu_origin),
        )
        delayed_update = position_measurement_from_rtk(
            delayed_fixed,
            rec_rtk_clock,
            rec_rtk_cal,
            attitude_at(
                rec_cal,
                rec_camera_clock,
                rec_rtk_clock.map_timestamp(recovery_epoch),
            ),
        )
        recovered = recovery.ingest_position_measurement(
            delayed_update, max_position_nis=100.0
        )
        self.assertEqual(recovered.status, "DEGRADED")
        self.assertEqual(recovered.timestamp_ns, rec_camera_clock.map_timestamp(T0 + 110_000_000))
        self.assertEqual(recovered.last_position_update_timestamp_ns, delayed_update.timestamp_ns)
        self.assertIsNotNone(recovered.position_world_from_body_m)
        matrix.append(("prolonged_loss_and_delayed_RTK_recovery", "recovered_at_acquisition_epoch"))

        # Visual/IMU attitude path independently verifies that LOST is sticky
        # until an explicit localization reseed is authorized.
        fusion, attitude_states = setup_pipeline(dropout=15_000_000)
        fusion.ingest_imu(attitude_states[0])
        fusion.ingest_imu(attitude_states[1])
        fusion.ingest_localization(attitude_observation(1_010_000_000, 10))
        fusion.ingest_imu(attitude_states[2])
        fusion.ingest_imu(attitude_states[3])
        attitude_lost = fusion.ingest_imu(attitude_states[4])
        self.assertEqual(attitude_lost.status, "LOST")
        with self.assertRaisesRegex(ValueError, "explicit reseed"):
            fusion.ingest_localization(attitude_observation(1_040_000_000, 11))
        attitude_recovered = fusion.ingest_localization(
            attitude_observation(1_040_000_000, 11),
            allow_reseed_after_lost=True,
        )
        self.assertEqual(attitude_recovered.status, "TRACKING")
        matrix.append(("visual_reacquisition_after_LOST", "explicit_reseed_required"))

        report = {
            "schema_version": 1,
            "qualification": "synthetic_software_fault_matrix_only",
            "scenarios": [
                {"id": name, "result": result} for name, result in matrix
            ],
            "scenario_count": len(matrix),
            "hardware_provenance_verified": False,
        }
        canonical = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(report["scenario_count"], 10)
        self.assertFalse(report["hardware_provenance_verified"])
        self.assertEqual(
            [item["id"] for item in report["scenarios"]],
            [
                "nominal_image_to_pose",
                "deterministic_corner_noise",
                "unknown_tag_id",
                "stale_map_revision",
                "global_anchor_survey_bias_10mm",
                "clock_offset_and_5ppm_drift",
                "RTK_FIXED_to_FLOAT",
                "IMU_gap",
                "prolonged_loss_and_delayed_RTK_recovery",
                "visual_reacquisition_after_LOST",
            ],
        )


if __name__ == "__main__":
    unittest.main()
