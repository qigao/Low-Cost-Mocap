import unittest

from imu_contract import (
    ClockAnchor,
    IMUSample,
    IMUSampleStreamGuard,
    fit_clock_mapping,
)


def sample(
    sequence=0,
    timestamp=1_000_000_000,
    *,
    imu_id="imu-1",
    clock_domain="imu-clock",
    calibration_revision="cal-v1",
    valid=True,
    saturated=False,
    overrun=False,
):
    return IMUSample(
        imu_id=imu_id,
        sequence_number=sequence,
        sensor_timestamp_ns=timestamp,
        clock_domain=clock_domain,
        gyro_rad_s=(0.1, -0.2, 0.3),
        accel_m_s2=(0.0, 0.0, 9.81),
        calibration_revision=calibration_revision,
        valid=valid,
        saturated=saturated,
        overrun=overrun,
        host_received_timestamp_ns=timestamp + 2_000_000,
    )


class IMUSampleContractTests(unittest.TestCase):
    def test_valid_sample_uses_explicit_si_units_and_separate_host_time(self):
        item = sample()
        item.require_usable()
        self.assertEqual(item.gyro_rad_s, (0.1, -0.2, 0.3))
        self.assertEqual(item.accel_m_s2, (0.0, 0.0, 9.81))
        self.assertEqual(item.sensor_timestamp_ns, 1_000_000_000)
        self.assertNotEqual(item.sensor_timestamp_ns, item.host_received_timestamp_ns)

    def test_rejects_bad_identity_sequence_timestamp_and_clock_domain(self):
        invalid = [
            dict(imu_id=""),
            dict(sequence=True),
            dict(timestamp=-1),
            dict(clock_domain=" "),
            dict(calibration_revision=""),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    sample(**kwargs)

    def test_rejects_malformed_nonfinite_and_boolean_vectors(self):
        for vector in ((1.0, 2.0), (1.0, float("nan"), 2.0), (1.0, True, 2.0), (0.0, float("inf"), 0.0)):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    IMUSample(
                        imu_id="imu-1", sequence_number=0,
                        sensor_timestamp_ns=1, clock_domain="imu-clock",
                        gyro_rad_s=vector, accel_m_s2=(0.0, 0.0, 9.81),
                        calibration_revision="cal-v1",
                    )

    def test_invalid_saturated_and_overrun_samples_are_not_usable(self):
        for item, message in (
            (sample(valid=False), "invalid"),
            (sample(saturated=True), "saturated"),
            (sample(overrun=True), "overrun"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    item.require_usable()

    def test_stream_guard_rejects_duplicate_reordered_gapped_and_wrong_revision(self):
        guard = IMUSampleStreamGuard(
            "imu-1", "imu-clock", "cal-v1", max_sample_gap_ns=10_000_000
        )
        guard.accept(sample(sequence=1, timestamp=1_000_000_000))
        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            guard.accept(sample(sequence=1, timestamp=1_001_000_000))
        with self.assertRaisesRegex(ValueError, "regressed"):
            guard.accept(sample(sequence=2, timestamp=999_000_000))
        with self.assertRaisesRegex(ValueError, "gap exceeds"):
            guard.accept(sample(sequence=2, timestamp=1_020_000_000))
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            guard.accept(sample(sequence=2, timestamp=1_002_000_000, calibration_revision="cal-v2"))
        guard.accept(sample(sequence=2, timestamp=1_002_000_000))
        guard.reset()
        guard.accept(sample(sequence=0, timestamp=500_000_000))

    def test_clock_mapping_fits_known_offset_and_drift(self):
        anchors = [
            ClockAnchor(1_000_000_000, 1_000_100_000, event_id="pulse-1"),
            ClockAnchor(2_000_000_000, 2_000_200_000, event_id="pulse-2"),
            ClockAnchor(3_000_000_000, 3_000_300_000, event_id="pulse-3"),
        ]
        mapping = fit_clock_mapping(
            anchors,
            source_clock_domain="imu-clock",
            target_clock_domain="camera-trigger-clock",
            max_drift_ppm=200,
            max_residual_ns=10,
            max_extrapolation_ns=100_000_000,
            min_anchor_span_ns=10_000_000,
        )
        self.assertEqual(mapping.anchor_count, 3)
        self.assertAlmostEqual(mapping.drift_ppm, 100.0, places=4)
        self.assertLessEqual(mapping.observed_max_residual_ns, 10)
        self.assertEqual(mapping.map_timestamp(1_500_000_000), 1_500_150_000)
        self.assertEqual(len(mapping.mapping_sha256), 64)

    def test_clock_mapping_rejects_too_few_or_host_receive_only_anchors(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            fit_clock_mapping(
                [ClockAnchor(1, 1), ClockAnchor(2, 2)],
                source_clock_domain="a", target_clock_domain="b",
                min_anchor_span_ns=1,
            )
        anchors = [
            ClockAnchor(1_000_000, 1_000_000, anchor_kind="host_receive_time"),
            ClockAnchor(11_000_000, 11_000_000),
            ClockAnchor(21_000_000, 21_000_000),
        ]
        with self.assertRaisesRegex(ValueError, "hardware sync pulses"):
            fit_clock_mapping(
                anchors, source_clock_domain="a", target_clock_domain="b",
                min_anchor_span_ns=1,
            )

    def test_clock_mapping_rejects_nonmonotonic_anchors_and_short_span(self):
        cases = [
            [
                ClockAnchor(1_000_000, 1_000_000),
                ClockAnchor(3_000_000, 3_000_000),
                ClockAnchor(2_000_000, 2_000_000),
            ],
            [
                ClockAnchor(1_000_000, 1_000_000),
                ClockAnchor(2_000_000, 2_000_000),
                ClockAnchor(3_000_000, 3_000_000),
            ],
        ]
        for anchors in cases:
            with self.subTest(anchors=anchors):
                with self.assertRaises(ValueError):
                    fit_clock_mapping(
                        anchors, source_clock_domain="a", target_clock_domain="b",
                        min_anchor_span_ns=10_000_000,
                    )

    def test_clock_mapping_rejects_excessive_drift_and_residual(self):
        drifted = [
            ClockAnchor(1_000_000_000, 1_000_000_000),
            ClockAnchor(2_000_000_000, 2_001_000_000),
            ClockAnchor(3_000_000_000, 3_002_000_000),
        ]
        with self.assertRaisesRegex(ValueError, "drift exceeds"):
            fit_clock_mapping(
                drifted, source_clock_domain="a", target_clock_domain="b",
                max_drift_ppm=100, min_anchor_span_ns=1,
            )
        outlier = [
            ClockAnchor(1_000_000_000, 1_000_000_000),
            ClockAnchor(2_000_000_000, 2_000_000_000),
            ClockAnchor(3_000_000_000, 3_030_000_000),
            ClockAnchor(4_000_000_000, 4_000_000_000),
        ]
        with self.assertRaisesRegex(ValueError, "residual exceeds"):
            fit_clock_mapping(
                outlier, source_clock_domain="a", target_clock_domain="b",
                max_drift_ppm=20_000, max_residual_ns=1_000_000,
                min_anchor_span_ns=1,
            )

    def test_clock_mapping_rejects_extrapolation_and_int64_overflow(self):
        mapping = fit_clock_mapping(
            [
                ClockAnchor(1_000_000_000, 2_000_000_000),
                ClockAnchor(2_000_000_000, 3_000_000_000),
                ClockAnchor(3_000_000_000, 4_000_000_000),
            ],
            source_clock_domain="imu", target_clock_domain="optical",
            max_drift_ppm=1, max_residual_ns=0,
            max_extrapolation_ns=100, min_anchor_span_ns=1,
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            mapping.map_timestamp(3_000_000_101)

        i64_max = (1 << 63) - 1
        overflow_mapping = fit_clock_mapping(
            [
                ClockAnchor(1_000_000_000, i64_max - 20_000_000),
                ClockAnchor(1_010_000_000, i64_max - 10_000_000),
                ClockAnchor(1_020_000_000, i64_max - 10),
            ],
            source_clock_domain="imu", target_clock_domain="optical",
            max_drift_ppm=2_000_000, max_residual_ns=10_000_000,
            max_extrapolation_ns=100, min_anchor_span_ns=1,
        )
        with self.assertRaisesRegex(ValueError, "signed 64-bit"):
            overflow_mapping.map_timestamp(1_020_000_050)


if __name__ == "__main__":
    unittest.main()
