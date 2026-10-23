import unittest

from imu_contract import ClockAnchor, fit_clock_mapping
from outdoor_sensor_contract import RTKSample, RTKSolutionStatus
from rtk_clock_alignment import align_rtk_sample_to_camera_clock


class RTKClockAlignmentTests(unittest.TestCase):
    def setUp(self):
        anchors = (
            ClockAnchor(1_000_000_000, 2_000_000_000, event_id="pps-1"),
            ClockAnchor(1_010_000_000, 2_010_000_000, event_id="pps-2"),
            ClockAnchor(1_020_000_000, 2_020_000_000, event_id="pps-3"),
        )
        self.mapping = fit_clock_mapping(
            anchors,
            source_clock_domain="rtk-receiver-clock",
            target_clock_domain="camera-ptp-clock",
            max_drift_ppm=10.0,
            max_residual_ns=1.0,
            max_extrapolation_ns=1_000_000,
            min_anchor_span_ns=10_000_000,
        )
        self.sample = RTKSample(
            receiver_id="rover-01",
            sequence_number=7,
            solution_timestamp_ns=1_015_000_000,
            clock_domain="rtk-receiver-clock",
            status=RTKSolutionStatus.FIXED,
            position_ecef_m=(6_378_137.0, 2.0, 3.0),
            horizontal_accuracy_m=0.02,
            vertical_accuracy_m=0.04,
            correction_age_ms=100.0,
            satellites_used=18,
            host_receive_timestamp_ns=1_017_000_000,
        )

    def test_maps_solution_epoch_and_preserves_original_sample(self):
        aligned = align_rtk_sample_to_camera_clock(
            self.sample, self.mapping,
            expected_camera_clock_domain="camera-ptp-clock",
        )
        self.assertEqual(aligned.mapped_timestamp_ns, 2_015_000_000)
        self.assertEqual(aligned.sample.solution_timestamp_ns, 1_015_000_000)
        self.assertEqual(aligned.sample.host_receive_timestamp_ns, 1_017_000_000)
        self.assertEqual(aligned.target_clock_domain, "camera-ptp-clock")
        self.assertEqual(aligned.clock_mapping_sha256, self.mapping.mapping_sha256)
        self.assertFalse(aligned.hardware_time_alignment_verified)

    def test_rejects_source_and_target_clock_domain_mismatch(self):
        with self.assertRaisesRegex(ValueError, "source domain"):
            align_rtk_sample_to_camera_clock(
                RTKSample(
                    receiver_id="rover-01", sequence_number=8,
                    solution_timestamp_ns=1_015_000_000,
                    clock_domain="wrong-source", status=RTKSolutionStatus.FIXED,
                    position_ecef_m=(6_378_137.0, 2.0, 3.0),
                    horizontal_accuracy_m=0.02, vertical_accuracy_m=0.04,
                    correction_age_ms=100.0, satellites_used=18,
                ),
                self.mapping, expected_camera_clock_domain="camera-ptp-clock",
            )
        with self.assertRaisesRegex(ValueError, "target domain"):
            align_rtk_sample_to_camera_clock(
                self.sample, self.mapping,
                expected_camera_clock_domain="another-camera-clock",
            )

    def test_rejects_solution_epoch_outside_mapping_horizon(self):
        sample = RTKSample(
            receiver_id="rover-01", sequence_number=9,
            solution_timestamp_ns=1_100_000_000,
            clock_domain="rtk-receiver-clock", status=RTKSolutionStatus.FIXED,
            position_ecef_m=(6_378_137.0, 2.0, 3.0),
            horizontal_accuracy_m=0.02, vertical_accuracy_m=0.04,
            correction_age_ms=100.0, satellites_used=18,
        )
        with self.assertRaisesRegex(ValueError, "outside qualified clock-mapping horizon"):
            align_rtk_sample_to_camera_clock(
                sample, self.mapping,
                expected_camera_clock_domain="camera-ptp-clock",
            )


if __name__ == "__main__":
    unittest.main()
