import copy
import unittest

import numpy as np

from localization_imu_fusion import LocalizationIMUAttitudeState
from localization_state_stream import (
    LocalizationStateConsumer,
    localization_state_to_dict,
    validate_localization_state_payload,
)
from translational_eskf import TranslationalErrorState


CLOCK_SHA = "a" * 64
IMU_SHA = "b" * 64
MAP_SHA = "c" * 64
LOC_CAL_SHA = "d" * 64
POSITION_SOURCE_SHA = "e" * 64
POSITION_CAL_SHA = "f" * 64
POSITION_CLOCK_SHA = "1" * 64
GRAVITY_SHA = "2" * 64


def attitude(timestamp=1_020_000_000, *, status="TRACKING", **changes):
    lost = status == "LOST"
    values = dict(
        body_id="rigid-body",
        map_frame_id="site-map",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=None if lost else (0.0, 0.0, 0.0, 1.0),
        attitude_variance_rad2=None if lost else 0.0025,
        status=status,
        source=(
            "LOST"
            if lost
            else "LOCALIZATION_CORRECTED"
            if status == "TRACKING"
            else "IMU_PROPAGATED"
        ),
        last_localization_timestamp_ns=(
            None if lost else 1_010_000_000
        ),
        last_source_sequence_number=None if lost else 11,
        localization_age_ns=None if lost else timestamp - 1_010_000_000,
        clock_mapping_sha256=CLOCK_SHA,
        imu_calibration_sha256=IMU_SHA,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
    )
    values.update(changes)
    return LocalizationIMUAttitudeState(**values)


def covariance(scale=1.0):
    base = np.diag(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.007, 0.008, 0.009]
    ) * scale
    base[0, 3] = base[3, 0] = 0.002 * scale
    base[0, 6] = base[6, 0] = -0.001 * scale
    return tuple(tuple(float(v) for v in row) for row in base)


def translation(timestamp=1_020_000_000, *, status="TRACKING", **changes):
    lost = status == "LOST"
    values = dict(
        body_id="rigid-body",
        imu_id="imu-1",
        map_frame_id="site-map",
        timestamp_ns=timestamp,
        imu_sequence_number=22,
        clock_domain="imu-clock",
        position_world_from_body_m=None if lost else (1.0, 2.0, 3.0),
        velocity_world_from_body_m_s=None if lost else (0.1, 0.2, 0.3),
        residual_accel_bias_body_m_s2=(0.01, -0.02, 0.03),
        covariance_p_v_bias=None if lost else covariance(),
        status=status,
        source=(
            "LOST"
            if lost
            else "POSITION_CORRECTED"
            if status == "TRACKING"
            else "IMU_PROPAGATED"
        ),
        last_position_update_timestamp_ns=1_015_000_000,
        position_age_ns=timestamp - 1_015_000_000,
        last_position_measurement_source="RTK",
        last_position_source_sequence_number=7,
        last_position_source_revision_sha256=POSITION_SOURCE_SHA,
        last_position_calibration_sha256=POSITION_CAL_SHA,
        last_position_clock_mapping_sha256=POSITION_CLOCK_SHA,
        localization_source_revision_sha256=MAP_SHA,
        localization_calibration_sha256=LOC_CAL_SHA,
        imu_calibration_sha256=IMU_SHA,
        gravity_model_sha256=GRAVITY_SHA,
        clock_mapping_sha256=CLOCK_SHA,
    )
    values.update(changes)
    return TranslationalErrorState(**values)


class LocalizationStateSerializationTests(unittest.TestCase):
    def test_tracking_components_serialize_one_metric_state(self):
        payload = localization_state_to_dict(attitude(), translation())
        self.assertEqual(payload["protocol"], "low_cost_mocap_localization")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["kind"], "localization_state")
        self.assertEqual(payload["status"], "TRACKING")
        self.assertEqual(payload["body_id"], "rigid-body")
        self.assertEqual(payload["map_frame_id"], "site-map")
        self.assertEqual(payload["timestamp_ns"], 1_020_000_000)
        self.assertEqual(payload["imu_sequence_number"], 22)
        self.assertEqual(payload["position_units"], "meters")
        self.assertTrue(payload["metric_scale_qualified"])
        self.assertFalse(payload["joint_attitude_translation_covariance_available"])
        self.assertFalse(payload["hardware_provenance_verified"])
        self.assertEqual(payload["quaternion_world_from_body_xyzw"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(payload["position_world_from_body_m"], [1.0, 2.0, 3.0])
        self.assertEqual(len(payload["translational_covariance_p_v_bias"]), 9)
        self.assertEqual(payload["last_position_measurement_source"], "RTK")
        self.assertEqual(payload["last_position_clock_mapping_sha256"], POSITION_CLOCK_SHA)

    def test_any_degraded_component_makes_unified_state_degraded(self):
        payload = localization_state_to_dict(
            attitude(status="DEGRADED"),
            translation(),
        )
        self.assertEqual(payload["status"], "DEGRADED")
        self.assertIsNotNone(payload["quaternion_world_from_body_xyzw"])
        self.assertIsNotNone(payload["position_world_from_body_m"])

        payload = localization_state_to_dict(
            attitude(),
            translation(status="DEGRADED"),
        )
        self.assertEqual(payload["status"], "DEGRADED")

    def test_any_lost_component_hides_complete_dynamic_state(self):
        payload = localization_state_to_dict(
            attitude(),
            translation(status="LOST"),
        )
        self.assertEqual(payload["status"], "LOST")
        for name in (
            "quaternion_world_from_body_xyzw",
            "position_world_from_body_m",
            "velocity_world_from_body_m_s",
            "residual_accel_bias_body_m_s2",
            "attitude_variance_rad2",
            "translational_covariance_p_v_bias",
        ):
            self.assertIsNone(payload[name])
        self.assertEqual(payload["attitude_status"], "TRACKING")
        self.assertEqual(payload["translation_status"], "LOST")

    def test_identity_time_frame_and_revisions_must_match(self):
        cases = (
            (translation(body_id="other-body"), "body mismatch"),
            (translation(imu_id="other-imu"), "IMU mismatch"),
            (translation(map_frame_id="other-map"), "map frame mismatch"),
            (translation(clock_domain="other-clock"), "clock domain mismatch"),
            (translation(timestamp_ns=1_020_000_001, position_age_ns=5_000_001), "timestamps"),
            (translation(imu_calibration_sha256="3"*64), "IMU calibration"),
            (translation(localization_source_revision_sha256="4"*64), "source revision"),
            (translation(localization_calibration_sha256="5"*64), "localization calibration"),
            (translation(clock_mapping_sha256="6"*64), "clock mapping"),
        )
        for state, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                localization_state_to_dict(attitude(), state)


class LocalizationStateValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload = localization_state_to_dict(attitude(), translation())

    def test_unknown_missing_hardware_and_joint_covariance_claims_fail_closed(self):
        extra = dict(self.payload)
        extra["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_localization_state_payload(extra)

        missing = dict(self.payload)
        del missing["gravity_model_sha256"]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_localization_state_payload(missing)

        hardware = dict(self.payload)
        hardware["hardware_provenance_verified"] = True
        with self.assertRaisesRegex(ValueError, "hardware provenance"):
            validate_localization_state_payload(hardware)

        joint = dict(self.payload)
        joint["joint_attitude_translation_covariance_available"] = True
        with self.assertRaisesRegex(ValueError, "joint attitude/translation"):
            validate_localization_state_payload(joint)

    def test_covariance_symmetry_psd_and_quaternion_norm_are_checked(self):
        bad = copy.deepcopy(self.payload)
        bad["translational_covariance_p_v_bias"][0][1] = 0.5
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_localization_state_payload(bad)

        bad = copy.deepcopy(self.payload)
        bad["translational_covariance_p_v_bias"][0][0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            validate_localization_state_payload(bad)

        bad = copy.deepcopy(self.payload)
        bad["quaternion_world_from_body_xyzw"] = [0.0, 0.0, 0.0, 2.0]
        with self.assertRaisesRegex(ValueError, "normalized"):
            validate_localization_state_payload(bad)

    def test_status_and_age_consistency_are_checked(self):
        bad = dict(self.payload)
        bad["status"] = "DEGRADED"
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            validate_localization_state_payload(bad)

        bad = dict(self.payload)
        bad["localization_age_ns"] += 1
        with self.assertRaisesRegex(ValueError, "localization age"):
            validate_localization_state_payload(bad)

        bad = dict(self.payload)
        bad["position_age_ns"] += 1
        with self.assertRaisesRegex(ValueError, "position age"):
            validate_localization_state_payload(bad)

    def test_lost_payload_cannot_retain_stale_dynamic_state(self):
        lost = localization_state_to_dict(attitude(), translation(status="LOST"))
        tampered = dict(lost)
        tampered["position_world_from_body_m"] = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "must hide dynamic state"):
            validate_localization_state_payload(tampered)

    def test_previous_time_and_imu_sequence_must_advance(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_localization_state_payload(
                self.payload,
                previous_timestamp_ns=self.payload["timestamp_ns"],
            )
        with self.assertRaisesRegex(ValueError, "IMU sequence"):
            validate_localization_state_payload(
                self.payload,
                previous_imu_sequence_number=self.payload["imu_sequence_number"],
            )


class LocalizationStateConsumerTests(unittest.TestCase):
    def test_consumer_pins_identity_and_revisions_transactionally(self):
        consumer = LocalizationStateConsumer()
        first = localization_state_to_dict(attitude(), translation())
        accepted = consumer.consume(first)
        self.assertEqual(consumer.state, accepted)

        newer = localization_state_to_dict(
            attitude(timestamp=1_030_000_000),
            translation(
                timestamp=1_030_000_000,
                imu_sequence_number=23,
                position_age_ns=15_000_000,
            ),
        )
        consumer.consume(newer)
        before = copy.deepcopy(consumer.state)

        wrong = dict(newer)
        wrong["timestamp_ns"] = 1_040_000_000
        wrong["imu_sequence_number"] = 24
        wrong["map_frame_id"] = "other-map"
        wrong["localization_age_ns"] += 10_000_000
        wrong["position_age_ns"] += 10_000_000
        with self.assertRaisesRegex(ValueError, "map_frame_id changed"):
            consumer.consume(wrong)
        self.assertEqual(consumer.state, before)

    def test_returned_payload_mutation_does_not_mutate_consumer_state(self):
        consumer = LocalizationStateConsumer()
        accepted = consumer.consume(
            localization_state_to_dict(attitude(), translation())
        )
        accepted["position_world_from_body_m"][0] = 999.0
        accepted["translational_covariance_p_v_bias"][0][0] = 999.0
        self.assertEqual(consumer.state["position_world_from_body_m"][0], 1.0)
        self.assertNotEqual(
            consumer.state["translational_covariance_p_v_bias"][0][0],
            999.0,
        )

    def test_reset_allows_new_session_identity(self):
        consumer = LocalizationStateConsumer()
        consumer.consume(localization_state_to_dict(attitude(), translation()))
        consumer.reset()
        payload = localization_state_to_dict(
            attitude(
                body_id="new-body",
                map_frame_id="new-map",
            ),
            translation(
                body_id="new-body",
                map_frame_id="new-map",
            ),
        )
        accepted = consumer.consume(payload)
        self.assertEqual(accepted["body_id"], "new-body")
        self.assertEqual(accepted["map_frame_id"], "new-map")


if __name__ == "__main__":
    unittest.main()
