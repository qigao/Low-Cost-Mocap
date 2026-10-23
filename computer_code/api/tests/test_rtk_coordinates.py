import math
import unittest

import numpy as np

from outdoor_sensor_contract import RTKSample, RTKSolutionStatus
from rtk_coordinates import (
    ENUOrigin,
    antenna_enu_to_body_origin_enu,
    ecef_to_enu,
    enu_to_ecef,
    geodetic_to_ecef,
    qualified_rtk_position_enu,
)


class RtkCoordinateTests(unittest.TestCase):
    def sample(self, *, status=RTKSolutionStatus.FIXED, **changes):
        values = dict(
            receiver_id="rover-01",
            sequence_number=10,
            solution_timestamp_ns=5_000_000,
            clock_domain="gnss-epoch",
            status=status,
            position_ecef_m=(6_378_139.0, 3.0, 4.0),
            horizontal_accuracy_m=0.02,
            vertical_accuracy_m=0.04,
            correction_age_ms=100.0,
            satellites_used=18,
            datum="WGS84",
            antenna_reference_point="antenna_phase_center",
        )
        values.update(changes)
        return RTKSample(**values)

    def test_wgs84_equator_origin_converts_to_expected_ecef(self):
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 6_378_137.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 0.0, places=6)

    def test_ecef_to_enu_uses_east_north_up_axis_order(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        result = ecef_to_enu((6_378_138.0, 2.0, 3.0), origin)
        np.testing.assert_allclose(result, (2.0, 3.0, 1.0), atol=1e-8, rtol=0.0)

    def test_enu_ecef_round_trip_preserves_local_coordinates(self):
        origin = ENUOrigin(37.4219999, -122.0840575, 17.5)
        local = (123.4, -45.6, 7.8)
        result = ecef_to_enu(enu_to_ecef(local, origin), origin)
        np.testing.assert_allclose(result, local, atol=1e-7, rtol=0.0)

    def test_rtk_is_converted_only_after_explicit_quality_gates(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        result = qualified_rtk_position_enu(
            self.sample(), origin,
            max_correction_age_ms=500.0,
            max_horizontal_accuracy_m=0.05,
            max_vertical_accuracy_m=0.08,
            min_satellites=10,
        )
        np.testing.assert_allclose(result, (3.0, 4.0, 2.0), atol=1e-8, rtol=0.0)

    def test_float_stale_or_datum_mismatch_is_rejected(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        options = dict(
            max_correction_age_ms=500.0,
            max_horizontal_accuracy_m=0.05,
            max_vertical_accuracy_m=0.08,
            min_satellites=10,
        )
        with self.assertRaisesRegex(ValueError, "does not meet declared"):
            qualified_rtk_position_enu(self.sample(status=RTKSolutionStatus.FLOAT), origin, **options)
        with self.assertRaisesRegex(ValueError, "does not meet declared"):
            qualified_rtk_position_enu(self.sample(correction_age_ms=2000.0), origin, **options)
        with self.assertRaisesRegex(ValueError, "datum"):
            qualified_rtk_position_enu(
                self.sample(datum="OTHER-DATUM"), origin, **options
            )

    def test_antenna_lever_arm_is_removed_in_body_frame(self):
        result = antenna_enu_to_body_origin_enu(
            antenna_position_enu_m=(10.0, 20.0, 30.0),
            rotation_enu_from_body=np.eye(3),
            lever_arm_body_m=(1.0, 2.0, 3.0),
        )
        np.testing.assert_allclose(result, (9.0, 18.0, 27.0), atol=1e-9, rtol=0.0)

    def test_antenna_lever_arm_rotates_with_body_orientation(self):
        rotation = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = antenna_enu_to_body_origin_enu(
            antenna_position_enu_m=(10.0, 20.0, 30.0),
            rotation_enu_from_body=rotation,
            lever_arm_body_m=(1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(result, (10.0, 19.0, 30.0), atol=1e-9, rtol=0.0)

    def test_lever_arm_rejects_reflection_nonorthogonal_and_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            antenna_enu_to_body_origin_enu((0, 0, 0), np.diag([-1.0, 1.0, 1.0]), (1, 0, 0))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            antenna_enu_to_body_origin_enu((0, 0, 0), np.diag([2.0, 1.0, 1.0]), (1, 0, 0))
        with self.assertRaisesRegex(ValueError, "finite"):
            antenna_enu_to_body_origin_enu((0, 0, 0), np.eye(3), (math.nan, 0, 0))

    def test_invalid_geodetic_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "latitude"):
            ENUOrigin(91.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "datum"):
            ENUOrigin(0.0, 0.0, 0.0, datum="unknown")


if __name__ == "__main__":
    unittest.main()
