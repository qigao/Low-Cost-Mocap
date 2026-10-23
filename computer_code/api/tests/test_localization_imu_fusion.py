import math
import unittest
from dataclasses import replace

from imu_calibration import CorrectedIMUVectors
from imu_contract import ClockAnchor, fit_clock_mapping
from imu_propagation import BoundedIMUAttitudePropagator
from localization_contract import (
    LocalizationQuality,
    LocalizationTopology,
    MapPoseObservation,
    PoseSource,
    ScaleReference,
)
from localization_imu_fusion import (
    LocalizationIMUAttitudeFusion,
    localization_attitude_measurement,
)


IMU_SHA = "a" * 64
MAP_SHA = "b" * 64
LOCALIZATION_CAL_SHA = "c" * 64


def mapping():
    return fit_clock_mapping(
        [
            ClockAnchor(
                1_000_000_000,
                1_000_001_000,
                event_id="sync-1",
            ),
            ClockAnchor(
                2_000_000_000,
                2_000_001_000,
                event_id="sync-2",
            ),
            ClockAnchor(
                3_000_000_000,
                3_000_001_000,
                event_id="sync-3",
            ),
        ],
        source_clock_domain="camera-clock",
        target_clock_domain="imu-clock",
        max_drift_ppm=10,
        max_residual_ns=10,
        max_extrapolation_ns=20_000_000,
        min_anchor_span_ns=100_000_000,
    )


def imu_sample(seq, timestamp, gyro=(0.0, 0.0, 1.0)):
    return CorrectedIMUVectors(
        imu_id="imu-1",
        sequence_number=seq,
        sensor_timestamp_ns=timestamp,
        clock_domain="imu-clock",
        calibration_revision="imu-cal-v1",
        calibration_sha256=IMU_SHA,
        gyro_body_rad_s=gyro,
        specific_force_body_m_s2=(0.0, 0.0, 9.81),
    )


def setup_pipeline(*, dropout=30_000_000):
    propagator = BoundedIMUAttitudePropagator(
        imu_id="imu-1",
        clock_domain="imu-clock",
        calibration_revision="imu-cal-v1",
        calibration_sha256=IMU_SHA,
        max_sample_gap_ns=20_000_000,
        max_step_ns=5_000_000,
        max_substeps=8,
    )
    fusion = LocalizationIMUAttitudeFusion(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        imu_clock_domain="imu-clock",
        imu_calibration_revision="imu-cal-v1",
        imu_calibration_sha256=IMU_SHA,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOCALIZATION_CAL_SHA,
        clock_mapping=mapping(),
        gyro_noise_density_rad_s_sqrt_hz=(0.01, 0.01, 0.01),
        default_attitude_variance_rad2=0.0025,
        max_dropout_ns=dropout,
    )
    states = [
        propagator.advance(
            imu_sample(seq, 1_000_001_000 + seq * 10_000_000)
        )
        for seq in range(6)
    ]
    return fusion, states


def observation(
    source_timestamp,
    sequence,
    *,
    quality=LocalizationQuality.TRACKING,
    orientation=(0.0, 0.0, 0.0, 1.0),
    orientation_stddev=(0.01, 0.02, 0.015),
    body_id="rigid-body",
    map_sha=MAP_SHA,
    calibration_sha=LOCALIZATION_CAL_SHA,
):
    return MapPoseObservation(
        source=PoseSource.FIDUCIAL_LOCALIZATION,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id=body_id,
        map_frame_id="site-map",
        timestamp_ns=source_timestamp,
        source_sequence_number=sequence,
        clock_domain="camera-clock",
        position_xyz=(1.0, 2.0, 3.0),
        position_unit="meters",
        orientation_xyzw=orientation,
        position_stddev=(0.02, 0.02, 0.03),
        orientation_stddev_rad=orientation_stddev,
        map_revision_sha256=map_sha,
        calibration_sha256=calibration_sha,
        quality=quality,
        inlier_count=3,
        metric_scale_qualified=True,
        scale_reference=ScaleReference.CALIBRATED_TAG_MAP,
        relocalized=False,
        reprojection_rms_px=0.3,
    )


class LocalizationIMUFusionTests(unittest.TestCase):
    def test_tracking_localization_seeds_existing_bounded_attitude_core(self):
        fusion, states = setup_pipeline()
        self.assertEqual(fusion.ingest_imu(states[0]).status, "LOST")
        fusion.ingest_imu(states[1])
        result = fusion.ingest_localization(
            observation(1_010_000_000, 10)
        )
        self.assertEqual(result.status, "TRACKING")
        self.assertEqual(result.source, "LOCALIZATION_CORRECTED")
        self.assertEqual(result.last_source_sequence_number, 10)
        self.assertEqual(
            result.last_localization_timestamp_ns, 1_010_001_000
        )
        self.assertEqual(result.localization_age_ns, 0)
        self.assertIsNone(result.global_position_m)
        self.assertFalse(result.global_position_observable)

    def test_localization_orientation_stddev_drives_measurement_variance(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        result = fusion.ingest_localization(
            observation(
                1_010_000_000,
                10,
                orientation_stddev=(0.01, 0.02, 0.015),
            )
        )
        self.assertAlmostEqual(
            result.attitude_variance_rad2,
            0.02**2,
            places=12,
        )

    def test_degraded_pose_is_rejected_by_default_and_can_be_explicitly_inflated(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        degraded = observation(
            1_010_000_000,
            10,
            quality=LocalizationQuality.DEGRADED,
            orientation_stddev=(0.01, 0.02, 0.015),
        )
        with self.assertRaisesRegex(
            ValueError, "DEGRADED localization correction is not enabled"
        ):
            fusion.ingest_localization(degraded)

        result = fusion.ingest_localization(
            degraded,
            allow_degraded=True,
            degraded_variance_scale=4.0,
        )
        self.assertAlmostEqual(
            result.attitude_variance_rad2,
            0.02**2 * 4.0,
            places=12,
        )

    def test_source_sequence_and_timestamp_both_must_advance(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        fusion.ingest_localization(observation(1_010_000_000, 10))
        fusion.ingest_imu(states[2])

        with self.assertRaisesRegex(ValueError, "duplicate/out of order"):
            fusion.ingest_localization(
                observation(1_020_000_000, 10)
            )
        with self.assertRaisesRegex(ValueError, "duplicate/out of order"):
            fusion.ingest_localization(
                observation(1_010_000_000, 11)
            )

    def test_revision_body_and_clock_mismatch_fail_closed(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])

        with self.assertRaisesRegex(ValueError, "identity/calibration"):
            fusion.ingest_localization(
                observation(
                    1_010_000_000,
                    10,
                    map_sha="d" * 64,
                )
            )
        with self.assertRaisesRegex(ValueError, "identity/calibration"):
            fusion.ingest_localization(
                observation(
                    1_010_000_000,
                    10,
                    calibration_sha="e" * 64,
                )
            )
        with self.assertRaisesRegex(ValueError, "body mismatch"):
            fusion.ingest_localization(
                observation(
                    1_010_000_000,
                    10,
                    body_id="other-body",
                )
            )
        wrong_clock = replace(
            observation(1_010_000_000, 10),
            clock_domain="other-clock",
        )
        with self.assertRaisesRegex(ValueError, "clock mapping source"):
            fusion.ingest_localization(wrong_clock)

    def test_imu_propagation_and_dropout_keep_existing_fail_closed_semantics(self):
        fusion, states = setup_pipeline(dropout=15_000_000)
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        fusion.ingest_localization(observation(1_010_000_000, 10))
        propagated = fusion.ingest_imu(states[2])
        self.assertEqual(propagated.status, "DEGRADED")
        self.assertEqual(propagated.source, "IMU_PROPAGATED")
        fusion.ingest_imu(states[3])
        lost = fusion.ingest_imu(states[4])
        self.assertEqual(lost.status, "LOST")
        self.assertIsNone(lost.quaternion_world_from_body_xyzw)

        with self.assertRaisesRegex(ValueError, "explicit reseed"):
            fusion.ingest_localization(
                observation(1_040_000_000, 11)
            )
        recovered = fusion.ingest_localization(
            observation(1_040_000_000, 11),
            allow_reseed_after_lost=True,
        )
        self.assertEqual(recovered.status, "TRACKING")

    def test_measurement_adapter_has_positive_variance_floor(self):
        obs = observation(
            1_010_000_000,
            10,
            orientation_stddev=(0.0, 0.0, 0.0),
        )
        adapted = localization_attitude_measurement(
            obs, variance_floor_rad2=1e-7
        )
        self.assertEqual(adapted.frame_id, 10)
        self.assertEqual(adapted.pattern_sha256, MAP_SHA)
        self.assertEqual(
            adapted.calibration_sha256, LOCALIZATION_CAL_SHA
        )
        self.assertAlmostEqual(
            adapted.attitude_variance_rad2, 1e-7
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            localization_attitude_measurement(
                obs, variance_floor_rad2=0.0
            )

    def test_optical_core_optional_variance_contract_rejects_nonpositive(self):
        from optical_imu_fusion import OpticalAttitudeMeasurement

        with self.assertRaisesRegex(
            ValueError, "attitude variance must be positive"
        ):
            OpticalAttitudeMeasurement(
                "rigid-body",
                1_010_000_000,
                "camera-clock",
                (0.0, 0.0, 0.0, 1.0),
                10,
                MAP_SHA,
                LOCALIZATION_CAL_SHA,
                attitude_variance_rad2=0.0,
            )


if __name__ == "__main__":
    unittest.main()
