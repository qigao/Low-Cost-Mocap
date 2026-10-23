import math
import unittest
from dataclasses import replace

import numpy as np

from imu_contract import ClockAnchor, fit_clock_mapping
from localization_imu_fusion import LocalizationIMUAttitudeState
from outdoor_sensor_contract import RTKSample, RTKSolutionStatus
from position_measurement import PositionMeasurementSource
from rtk_coordinates import ENUOrigin, enu_to_ecef
from rtk_position_measurement import (
    RTKPositionCalibration,
    position_measurement_from_rtk,
)


def mapping():
    return fit_clock_mapping(
        [
            ClockAnchor(1_000_000_000, 1_000_001_000, event_id="rtk-sync-1"),
            ClockAnchor(2_000_000_000, 2_000_001_000, event_id="rtk-sync-2"),
            ClockAnchor(3_000_000_000, 3_000_001_000, event_id="rtk-sync-3"),
        ],
        source_clock_domain="rtk-clock",
        target_clock_domain="imu-clock",
        max_drift_ppm=10,
        max_residual_ns=10,
        max_extrapolation_ns=20_000_000,
        min_anchor_span_ns=100_000_000,
    )


def origin():
    return ENUOrigin(0.0, 0.0, 0.0)


def calibration(**changes):
    values = dict(
        receiver_id="rtk-1",
        body_id="rigid-body",
        map_frame_id="site-ENU",
        enu_origin=origin(),
        antenna_lever_arm_body_m=(0.5, -0.2, 0.1),
        horizontal_accuracy_to_axis_stddev_scale=1.0,
        vertical_accuracy_to_stddev_scale=1.0,
        max_correction_age_ms=1000.0,
        max_horizontal_accuracy_m=0.1,
        max_vertical_accuracy_m=0.2,
        min_satellites=8,
    )
    values.update(changes)
    return RTKPositionCalibration(**values)


def sample(**changes):
    cal = calibration()
    body = np.asarray((10.0, 20.0, 5.0))
    lever = np.asarray(cal.antenna_lever_arm_body_m)
    antenna = body + lever
    values = dict(
        receiver_id="rtk-1",
        sequence_number=7,
        solution_timestamp_ns=1_000_000_000,
        clock_domain="rtk-clock",
        status=RTKSolutionStatus.FIXED,
        position_ecef_m=enu_to_ecef(antenna, cal.enu_origin),
        horizontal_accuracy_m=0.02,
        vertical_accuracy_m=0.04,
        correction_age_ms=100.0,
        satellites_used=12,
        datum="WGS84",
        antenna_reference_point="antenna_phase_center",
    )
    values.update(changes)
    return RTKSample(**values)


def attitude(**changes):
    values = dict(
        body_id="rigid-body",
        map_frame_id="site-ENU",
        imu_id="imu-1",
        timestamp_ns=1_000_001_000,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=(0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=0.0004,
        status="TRACKING",
        source="LOCALIZATION_CORRECTED",
        last_localization_timestamp_ns=1_000_001_000,
        last_source_sequence_number=10,
        localization_age_ns=0,
        clock_mapping_sha256="a" * 64,
        imu_calibration_sha256="b" * 64,
        localization_source_revision_sha256="c" * 64,
        localization_calibration_sha256="d" * 64,
    )
    values.update(changes)
    return LocalizationIMUAttitudeState(**values)


class RTKPositionCalibrationTests(unittest.TestCase):
    def test_revisions_are_deterministic_and_sensitive(self):
        first = calibration()
        second = calibration()
        self.assertEqual(first.source_revision_sha256, second.source_revision_sha256)
        self.assertEqual(first.calibration_sha256, second.calibration_sha256)
        moved_origin = calibration(
            enu_origin=ENUOrigin(0.0, 0.0001, 0.0)
        )
        self.assertNotEqual(first.source_revision_sha256, moved_origin.source_revision_sha256)
        changed_lever = calibration(antenna_lever_arm_body_m=(0.6, -0.2, 0.1))
        self.assertNotEqual(first.calibration_sha256, changed_lever.calibration_sha256)

    def test_accuracy_semantics_and_satellite_gate_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            calibration(horizontal_accuracy_to_axis_stddev_scale=0.0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            calibration(min_satellites=0)


class RTKPositionMeasurementTests(unittest.TestCase):
    def test_fixed_solution_converts_to_body_origin_in_site_enu(self):
        cal = calibration()
        result = position_measurement_from_rtk(
            sample(), mapping(), cal, attitude()
        )
        self.assertEqual(result.source, PositionMeasurementSource.RTK)
        self.assertEqual(result.body_id, "rigid-body")
        self.assertEqual(result.map_frame_id, "site-ENU")
        self.assertEqual(result.timestamp_ns, 1_000_001_000)
        self.assertEqual(result.source_sequence_number, 7)
        np.testing.assert_allclose(
            result.position_world_from_body_m,
            (10.0, 20.0, 5.0),
            atol=1e-7,
        )
        self.assertEqual(result.source_revision_sha256, cal.source_revision_sha256)
        self.assertEqual(result.measurement_calibration_sha256, cal.calibration_sha256)
        self.assertEqual(result.clock_mapping_sha256, mapping().mapping_sha256)

    def test_uncertainty_includes_receiver_accuracy_and_lever_attitude_term(self):
        cal = calibration()
        result = position_measurement_from_rtk(
            sample(), mapping(), cal, attitude()
        )
        lever_variance = (
            np.dot(cal.antenna_lever_arm_body_m, cal.antenna_lever_arm_body_m)
            * 0.0004
        )
        expected_h = math.sqrt(0.02**2 + lever_variance)
        expected_v = math.sqrt(0.04**2 + lever_variance)
        np.testing.assert_allclose(
            result.position_stddev_m,
            (expected_h, expected_h, expected_v),
            atol=1e-12,
        )

    def test_float_stale_and_low_quality_rtk_do_not_become_position_updates(self):
        cases = (
            ({"status": RTKSolutionStatus.FLOAT}, "global-position gates"),
            ({"correction_age_ms": 1500.0}, "global-position gates"),
            ({"horizontal_accuracy_m": 0.2}, "global-position gates"),
            ({"vertical_accuracy_m": 0.3}, "global-position gates"),
            ({"satellites_used": 4}, "global-position gates"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                position_measurement_from_rtk(
                    sample(**changes), mapping(), calibration(), attitude()
                )

    def test_receiver_antenna_datum_and_frame_identity_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "receiver identity"):
            position_measurement_from_rtk(
                sample(receiver_id="other"),
                mapping(), calibration(), attitude(),
            )
        with self.assertRaisesRegex(ValueError, "antenna phase center"):
            position_measurement_from_rtk(
                sample(antenna_reference_point="vehicle-origin"),
                mapping(), calibration(), attitude(),
            )
        with self.assertRaisesRegex(ValueError, "datum"):
            position_measurement_from_rtk(
                sample(datum="OTHER"),
                mapping(), calibration(), attitude(),
            )
        with self.assertRaisesRegex(ValueError, "map frame"):
            position_measurement_from_rtk(
                sample(),
                mapping(), calibration(), attitude(map_frame_id="other-map"),
            )

    def test_rtk_solution_must_align_to_current_publishable_attitude(self):
        with self.assertRaisesRegex(ValueError, "align exactly"):
            position_measurement_from_rtk(
                sample(solution_timestamp_ns=1_001_000_000),
                mapping(), calibration(), attitude(),
            )
        with self.assertRaisesRegex(ValueError, "publishable attitude"):
            position_measurement_from_rtk(
                sample(),
                mapping(), calibration(),
                attitude(
                    status="LOST",
                    source="LOST",
                    quaternion_world_from_body_xyzw=None,
                    attitude_variance_rad2=None,
                    last_localization_timestamp_ns=None,
                    last_source_sequence_number=None,
                    localization_age_ns=None,
                ),
            )

    def test_clock_mapping_domains_are_strict(self):
        wrong_source = fit_clock_mapping(
            [
                ClockAnchor(1_000_000_000, 1_000_001_000, event_id="a"),
                ClockAnchor(2_000_000_000, 2_000_001_000, event_id="b"),
                ClockAnchor(3_000_000_000, 3_000_001_000, event_id="c"),
            ],
            source_clock_domain="other-rtk-clock",
            target_clock_domain="imu-clock",
            max_drift_ppm=10,
            max_residual_ns=10,
            max_extrapolation_ns=20_000_000,
            min_anchor_span_ns=100_000_000,
        )
        with self.assertRaisesRegex(ValueError, "source domain"):
            position_measurement_from_rtk(
                sample(), wrong_source, calibration(), attitude()
            )

        wrong_target = fit_clock_mapping(
            [
                ClockAnchor(1_000_000_000, 1_000_001_000, event_id="a"),
                ClockAnchor(2_000_000_000, 2_000_001_000, event_id="b"),
                ClockAnchor(3_000_000_000, 3_000_001_000, event_id="c"),
            ],
            source_clock_domain="rtk-clock",
            target_clock_domain="other-clock",
            max_drift_ppm=10,
            max_residual_ns=10,
            max_extrapolation_ns=20_000_000,
            min_anchor_span_ns=100_000_000,
        )
        with self.assertRaisesRegex(ValueError, "target domain"):
            position_measurement_from_rtk(
                sample(), wrong_target, calibration(), attitude()
            )


if __name__ == "__main__":
    unittest.main()
