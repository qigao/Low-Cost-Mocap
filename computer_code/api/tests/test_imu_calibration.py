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
        "gyro_bias_random_walk_rad_s_per_sqrt_s": (0.0001, 0.0001, 0.0002),
        "accel_bias_random_walk_m_s2_per_sqrt_s": (0.001, 0.001, 0.002),
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

    def test_correction_matrices_apply_after_bias_and_before_body_rotation(self):
        value = calibration(
            gyro_correction_matrix=(
                (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.5),
            ),
            accel_correction_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 0.0, 0.5),
            ),
        )
        corrected = correct_imu_sample(sample(), value)
        self.assertEqual(corrected.gyro_body_rad_s, (1.8, 1.8, 1.35))
        self.assertAlmostEqual(corrected.specific_force_body_m_s2[1], 0.0)
        self.assertAlmostEqual(corrected.specific_force_body_m_s2[2], 4.5)

    def test_calibration_json_object_roundtrips_and_rejects_unknown_fields(self):
        value = calibration()
        decoded = IMUCalibration.from_dict(value.to_dict())
        self.assertEqual(decoded, value)
        self.assertEqual(decoded.calibration_sha256, value.calibration_sha256)
        raw = value.to_dict()
        raw["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown IMU calibration fields"):
            IMUCalibration.from_dict(raw)
        raw = value.to_dict()
        raw["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            IMUCalibration.from_dict(raw)

    def test_bias_random_walk_parameters_are_explicit_and_nonnegative(self):
        value = calibration()
        self.assertEqual(value.gyro_bias_random_walk_rad_s_per_sqrt_s, (0.0001, 0.0001, 0.0002))
        self.assertEqual(value.accel_bias_random_walk_m_s2_per_sqrt_s, (0.001, 0.001, 0.002))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            calibration(gyro_bias_random_walk_rad_s_per_sqrt_s=(0.0, -0.1, 0.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            calibration(accel_bias_random_walk_m_s2_per_sqrt_s=(0.0, float("nan"), 0.0))

    def test_correction_matrices_must_be_finite_and_well_conditioned(self):
        invalid_matrices = (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0001)),
            ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    calibration(gyro_correction_matrix=matrix)

    def test_digest_changes_when_calibration_values_change(self):
        baseline = calibration()
        changed = calibration(gyro_correction_matrix=(
            (1.01, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ))
        self.assertNotEqual(baseline.calibration_sha256, changed.calibration_sha256)

    def test_calibration_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            calibration(calibration_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "must be a string"):
            calibration(calibration_sha256=None)

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
