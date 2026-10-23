import math
import unittest
from dataclasses import replace

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from imu_interval_interpolation import interpolate_imu_observation_epoch
from localization_imu_fusion import LocalizationIMUAttitudeState

MAP_SHA = "a" * 64
LOC_CAL_SHA = "b" * 64
CLOCK_SHA = "d" * 64
T0 = 1_000_000_000

def calibration():
    return IMUCalibration(
        imu_id="imu-1", revision="imu-cal-v1", imu_clock_domain="imu-clock",
        rotation_body_from_imu=((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)),
        translation_body_from_imu_m=(0.0,0.0,0.0), gyro_bias_rad_s=(0.0,0.0,0.0),
        accel_bias_m_s2=(0.0,0.0,0.0), gyro_noise_density_rad_s_sqrt_hz=(0.01,)*3,
        accel_noise_density_m_s2_sqrt_hz=(0.03,)*3,
        gyro_bias_random_walk_rad_s_per_sqrt_s=(0.001,)*3,
        accel_bias_random_walk_m_s2_per_sqrt_s=(0.01,)*3, calibration_method="synthetic-test")

def sample(cal, seq, ts, force, gyro=(0.0,0.0,0.0)):
    return CorrectedIMUVectors("imu-1", seq, ts, "imu-clock", cal.revision,
        cal.calibration_sha256, gyro, force)

def attitude(cal, ts, q=(0.0,0.0,0.0,1.0), variance=0.01):
    return LocalizationIMUAttitudeState(
        body_id="body-1", map_frame_id="map-1", imu_id="imu-1", timestamp_ns=ts,
        clock_domain="imu-clock", quaternion_world_from_body_xyzw=q,
        attitude_variance_rad2=variance, status="TRACKING", source="LOCALIZATION_CORRECTED",
        last_localization_timestamp_ns=ts, last_source_sequence_number=1, localization_age_ns=0,
        clock_mapping_sha256=CLOCK_SHA, imu_calibration_sha256=cal.calibration_sha256,
        localization_source_revision_sha256=MAP_SHA, localization_calibration_sha256=LOC_CAL_SHA)

class IMUIntervalInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.cal = calibration()
        self.left = sample(self.cal,10,T0,(0.0,2.0,4.0),gyro=(0.0,2.0,4.0))
        self.right = sample(self.cal,12,T0+20_000_000,(2.0,4.0,6.0),gyro=(2.0,4.0,6.0))
        self.left_att = attitude(self.cal,T0,variance=0.01)
        self.right_att = attitude(self.cal,T0+20_000_000,
            q=(0.0,0.0,math.sqrt(0.5),math.sqrt(0.5)),variance=0.03)

    def test_midpoint_interpolates_vectors_variance_and_slerp_without_fabricated_sequence(self):
        epoch = interpolate_imu_observation_epoch(self.left,self.left_att,self.right,
            self.right_att,T0+10_000_000)
        self.assertEqual((epoch.left_sequence_number,epoch.right_sequence_number),(10,12))
        self.assertEqual(epoch.specific_force_body_m_s2,(1.0,3.0,5.0))
        self.assertEqual(epoch.gyro_body_rad_s,(1.0,3.0,5.0))
        self.assertAlmostEqual(epoch.attitude_variance_rad2,0.02)
        self.assertAlmostEqual(epoch.quaternion_world_from_body_xyzw[2],math.sin(math.pi/8))
        self.assertAlmostEqual(epoch.quaternion_world_from_body_xyzw[3],math.cos(math.pi/8))

    def test_rejects_endpoint_and_outside_epochs(self):
        for ts in (T0,T0+20_000_000,T0-1,T0+20_000_001):
            with self.subTest(timestamp=ts), self.assertRaisesRegex(ValueError,"strictly inside"):
                interpolate_imu_observation_epoch(self.left,self.left_att,self.right,self.right_att,ts)

    def test_rejects_provenance_change_and_lost_attitude(self):
        bad = replace(self.right_att,clock_mapping_sha256="e"*64)
        with self.assertRaisesRegex(ValueError,"clock mapping mismatch"):
            interpolate_imu_observation_epoch(self.left,self.left_att,self.right,bad,T0+5_000_000)
        lost = replace(self.right_att,quaternion_world_from_body_xyzw=None,
            attitude_variance_rad2=None,status="LOST",source="LOST")
        with self.assertRaisesRegex(ValueError,"publishable endpoint"):
            interpolate_imu_observation_epoch(self.left,self.left_att,self.right,lost,T0+5_000_000)

if __name__ == "__main__":
    unittest.main()
