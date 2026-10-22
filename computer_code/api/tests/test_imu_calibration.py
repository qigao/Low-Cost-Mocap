import unittest

from imu_calibration import IMUCalibration, correct_imu_sample
from imu_contract import IMUSample


def sample(
    *,
    imu_id="imu-1",
    sequence=4,
    timestamp=2_000_000,
    clock_domain="imu-clock",
    revision="cal-v1",
    gyro=(1.0, 2.0, 3.0),
    accel=(0.0, 0.0, 9.81),
    valid=True,
    saturated=False,
    overrun=False,
):
    return IMUSample(
        imu_id=imu_id,
        sequence_number=sequence,
        sensor_timestamp_ns=timestamp,
        clock_domain=clock_domain,
        gyro_rad_s=gyro,
        accel_m_s2=accel,
        calibration_revision=revision,
        valid=valid,
        saturated=saturated,
        overrun=overrun,
    )


def calibration(**overrides):
    fields = {
        "imu_id": "imu-1",
        "revision": "cal-v1",
        "imu_clock_domain": "imu-clock",
        "rotation_body_from_imu": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "translation_body_from_imu_m": (0.02, 0.0, 0.01),
        "gyro_bias_rad_s": (0.1, 0.2, 0.3),
        "accel_bias_m_s2": (0.0, 0.0, 0.81),
        "gyro_noise_density_rad_s_sqrt_hz": (0.001, 0.001, 0.002),
        "accel_noise_density_m_s2_sqrt_hz": (0.01, 0.01, 0.02),
        "calibration_method": "bench-six-position-v1",
        "valid_from_timestamp_ns": 1_000_000,
        "valid_until_timestamp_ns": 10_000_000,
    }
    fields.update(overrides)
    return IMUCalibration(**fields)


class IMUCalibrationContractTests(unittest.TestCase):
    def test_calibration_is_immutable_and_has_content_digest(self):
        value = calibration()
        self.assertEqual(len(value.calibration_sha256), 64)
        self.assertEqual(value.calibration_sha256, calibration().calibration_sha256)
        with self.assertRaises(Exception):
            value.revision = "changed"

    def test_bias_correction_and_identity_transform(self):
        value = calibration()
        corrected = correct_imu_sample(sample(), value)
        self.assertEqual(corrected.gyro_body_rad_s, (0.9, 1.8, 2.7))
        self.assertAlmostEqual(corrected.specific_force_body_m_s2[2], 9.0)
        self.assertEqual(corrected.measurement_reference_point, "imu_origin")
        self.assertEqual(corrected.calibration_sha256, value.calibration_sha256)

    def test_rotation_maps_sensor_vectors_into_body_axes(self):
        value = calibration(
            rotation_body_from_imu=(
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            gyro_bias_rad_s=(0.1, 0.0, 0.0),
            accel_bias_m_s2=(0.0, 0.0, 0.81),
        )
        corrected = correct_imu_sample(
            sample(gyro=(1.1, 0.0, 0.0), accel=(0.0, 0.0, 9.81)),
            value,
        )
        self.assertAlmostEqual(corrected.gyro_body_rad_s[0], 0.0)
        self.assertAlmostEqual(corrected.gyro_body_rad_s[1], 1.0)
        self.assertAlmostEqual(corrected.specific_force_body_m_s2[2], 9.0)

    def test_rejects_reflection_nonorthogonal_and_bad_translation(self):
        invalid_rotations = [
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
            ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        ]
        for rotation in invalid_rotations:
            with self.subTest(rotation=rotation):
                with self.assertRaises(ValueError):
                    calibration(rotation_body_from_imu=rotation)
        with self.assertRaisesRegex(ValueError, "finite real"):
            calibration(translation_body_from_imu_m=(0.0, float("nan"), 0.0))

    def test_rejects_negative_noise_and_nonfinite_bias(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            calibration(gyro_noise_density_rad_s_sqrt_hz=(0.1, -0.1, 0.1))
        with self.assertRaisesRegex(ValueError, "finite real"):
            calibration(accel_bias_m_s2=(0.0, float("inf"), 0.0))

    def test_calibration_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            calibration(calibration_sha256="0" * 64)

    def test_sample_device_clock_revision_and_validity_are_enforced(self):
        value = calibration()
        cases = [
            (sample(imu_id="imu-2"), "device ID mismatch"),
            (sample(clock_domain="host-clock"), "clock-domain mismatch"),
            (sample(revision="cal-v2"), "revision mismatch"),
            (sample(timestamp=999_999), "not yet valid"),
            (sample(timestamp=10_000_000), "expired"),
            (sample(saturated=True), "saturated"),
            (sample(overrun=True), "overrun"),
            (sample(valid=False), "invalid"),
        ]
        for item, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    correct_imu_sample(item, value)

    def test_validity_interval_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "after valid_from"):
            calibration(
                valid_from_timestamp_ns=5_000_000,
                valid_until_timestamp_ns=5_000_000,
            )


if __name__ == "__main__":
    unittest.main()
