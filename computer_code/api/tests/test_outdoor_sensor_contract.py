import unittest
from dataclasses import replace

from outdoor_sensor_contract import (
    CameraFrameMetadata,
    CameraSyncProfile,
    RTKSample,
    RTKSolutionStatus,
    validate_camera_batch,
    validate_camera_frame,
)


class CameraSyncContractTests(unittest.TestCase):
    def setUp(self):
        self.profiles = (
            CameraSyncProfile(
                camera_id=2, device_serial="CAM-02", transport="gige_vision",
                shutter_type="global", trigger_mode="external_line",
                timestamp_semantics="exposure_midpoint", clock_domain="capture-ptp-A",
                max_exposure_skew_ns=100_000,
            ),
            CameraSyncProfile(
                camera_id=5, device_serial="CAM-05", transport="gige_vision",
                shutter_type="global", trigger_mode="external_line",
                timestamp_semantics="exposure_midpoint", clock_domain="capture-ptp-A",
                max_exposure_skew_ns=100_000,
            ),
        )
        self.frames = (
            CameraFrameMetadata(
                camera_id=2, device_serial="CAM-02", frame_counter=900,
                trigger_id=42, acquisition_timestamp_ns=8_000_000,
                clock_domain="capture-ptp-A", timestamp_semantics="exposure_midpoint",
                hardware_triggered=True, ptp_locked=True, host_receive_timestamp_ns=8_004_000,
            ),
            CameraFrameMetadata(
                camera_id=5, device_serial="CAM-05", frame_counter=1200,
                trigger_id=42, acquisition_timestamp_ns=8_000_050,
                clock_domain="capture-ptp-A", timestamp_semantics="exposure_midpoint",
                hardware_triggered=True, ptp_locked=True, host_receive_timestamp_ns=8_002_000,
            ),
        )

    def test_valid_batch_uses_exposure_not_host_receive_timestamps(self):
        batch = validate_camera_batch(
            self.frames, self.profiles, expected_trigger_id=42,
            previous_frame_counters={2: 899, 5: 1199},
        )
        self.assertEqual(batch.exposure_skew_ns, 50)
        self.assertEqual(batch.camera_ids, (2, 5))
        self.assertEqual(batch.clock_domain, "capture-ptp-A")

    def test_host_receive_or_dequeue_time_cannot_be_exposure_timestamp(self):
        profile = self.profiles[0]
        for semantics in ("host_receive", "host_dequeue", "unknown"):
            with self.subTest(semantics=semantics):
                frame = replace(self.frames[0], timestamp_semantics=semantics)
                with self.assertRaises(ValueError):
                    validate_camera_frame(frame, profile, expected_trigger_id=42)

    def test_camera_identity_trigger_clock_and_counter_must_match(self):
        cases = (
            ({"device_serial": "OTHER"}, "camera identity"),
            ({"trigger_id": 41}, "trigger_id"),
            ({"clock_domain": "host-clock"}, "clock domain"),
            ({"frame_counter": 899}, "stale or repeated"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                frame = replace(self.frames[0], **changes)
                with self.assertRaisesRegex(ValueError, message):
                    validate_camera_frame(
                        frame, self.profiles[0], expected_trigger_id=42,
                        previous_frame_counter=899,
                    )

    def test_missing_timestamp_packet_loss_and_untriggered_frame_fail_closed(self):
        cases = (
            ({"acquisition_timestamp_ns": None}, "timestamp is missing"),
            ({"trigger_id": None}, "trigger_id"),
            ({"packet_loss_count": 1}, "packet loss"),
            ({"hardware_triggered": False}, "hardware trigger"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                frame = replace(self.frames[0], **changes)
                with self.assertRaisesRegex(ValueError, message):
                    validate_camera_frame(frame, self.profiles[0], expected_trigger_id=42)

    def test_ptp_scheduled_profile_requires_ptp_lock(self):
        profile = CameraSyncProfile(
            camera_id=2, device_serial="CAM-02", transport="gige_vision",
            shutter_type="global", trigger_mode="ptp_scheduled",
            timestamp_semantics="exposure_start", clock_domain="ptp-A",
            max_exposure_skew_ns=50_000, require_ptp_lock=True,
        )
        frame = replace(
            self.frames[0], timestamp_semantics="exposure_start",
            clock_domain="ptp-A", ptp_locked=False,
        )
        with self.assertRaisesRegex(ValueError, "PTP clock is not locked"):
            validate_camera_frame(frame, profile, expected_trigger_id=42)

    def test_profile_rejects_free_run_when_hardware_trigger_is_required(self):
        with self.assertRaisesRegex(ValueError, "free_run"):
            CameraSyncProfile(
                camera_id=1, device_serial="CAM-01", transport="gige_vision",
                shutter_type="global", trigger_mode="free_run",
                timestamp_semantics="exposure_start", clock_domain="ptp-A",
                max_exposure_skew_ns=1000,
            )

    def test_batch_rejects_skew_mismatch_missing_camera_and_mixed_trigger(self):
        with self.assertRaisesRegex(ValueError, "skew exceeds"):
            validate_camera_batch(
                (self.frames[0], replace(self.frames[1], acquisition_timestamp_ns=9_000_000)),
                self.profiles, expected_trigger_id=42,
            )
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            validate_camera_batch(self.frames[:1], self.profiles, expected_trigger_id=42)
        with self.assertRaisesRegex(ValueError, "trigger_id"):
            validate_camera_batch(
                (self.frames[0], replace(self.frames[1], trigger_id=43)),
                self.profiles, expected_trigger_id=42,
            )


class RTKContractTests(unittest.TestCase):
    def sample(self, status=RTKSolutionStatus.FIXED, **changes):
        values = dict(
            receiver_id="rover-01", sequence_number=700,
            solution_timestamp_ns=10_000_000, clock_domain="gnss-solution-clock",
            status=status, position_ecef_m=(3_912_000.0, 345_000.0, 5_012_000.0),
            horizontal_accuracy_m=0.025, vertical_accuracy_m=0.04,
            correction_age_ms=150.0, satellites_used=18,
            datum="WGS84", antenna_reference_point="antenna_phase_center",
            host_receive_timestamp_ns=10_003_000,
        )
        values.update(changes)
        return RTKSample(**values)

    def test_fixed_fresh_solution_can_pass_explicit_global_position_gate(self):
        self.assertTrue(self.sample().usable_as_global_position(
            max_correction_age_ms=500,
            max_horizontal_accuracy_m=0.05,
            max_vertical_accuracy_m=0.08,
            min_satellites=10,
        ))

    def test_float_stale_or_low_quality_solution_is_not_global_anchor(self):
        for sample in (
            self.sample(RTKSolutionStatus.FLOAT),
            self.sample(correction_age_ms=2000.0),
            self.sample(horizontal_accuracy_m=0.5),
            self.sample(satellites_used=4),
        ):
            with self.subTest(status=sample.status, age=sample.correction_age_ms):
                self.assertFalse(sample.usable_as_global_position(
                    max_correction_age_ms=500,
                    max_horizontal_accuracy_m=0.05,
                    max_vertical_accuracy_m=0.08,
                    min_satellites=10,
                ))

    def test_rtk_solution_rejects_nonfinite_coordinates_and_host_only_clock(self):
        with self.assertRaisesRegex(ValueError, "position_ecef_m"):
            self.sample(position_ecef_m=(1.0, float("nan"), 3.0))
        sample = self.sample()
        self.assertEqual(sample.solution_timestamp_ns, 10_000_000)
        self.assertEqual(sample.host_receive_timestamp_ns, 10_003_000)


if __name__ == "__main__":
    unittest.main()
