import unittest
import numpy as np

from rigid_pattern import fit_rigid_pattern


class RigidPatternTests(unittest.TestCase):
    def setUp(self):
        self.model = np.array([
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.0, 0.095, 0.0],
            [0.0, 0.0, 0.08],
        ])
        angle = 0.7
        self.rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.translation = np.array([1.2, -0.4, 2.3])
        self.observed = (self.rotation @ self.model.T).T + self.translation

    def test_recovers_proper_rigid_transform(self):
        R, t, metrics = fit_rigid_pattern(self.model, self.observed)
        np.testing.assert_allclose(R, self.rotation, atol=1e-10)
        np.testing.assert_allclose(t, self.translation, atol=1e-10)
        self.assertAlmostEqual(metrics["rms_error"], 0.0, places=10)
        self.assertAlmostEqual(metrics["det_rotation"], 1.0, places=10)

    def test_sparse_observations_keep_model_marker_indices(self):
        observed = self.observed[[0, 2, 3]]
        R, t, metrics = fit_rigid_pattern(
            self.model, observed, marker_indices=[0, 2, 3]
        )
        np.testing.assert_allclose(R, self.rotation, atol=1e-10)
        np.testing.assert_allclose(t, self.translation, atol=1e-10)
        self.assertEqual(metrics["marker_indices"], [0, 2, 3])

    def test_no_scale_fitting_exposes_wrong_pattern_size(self):
        scaled = (self.observed - self.translation) * 1.2 + self.translation
        with self.assertRaisesRegex(ValueError, "RMS"):
            fit_rigid_pattern(self.model, scaled, max_rms_error=0.005)

    def test_rigid_fit_is_invariant_to_small_model_units(self):
        # Coordinates may be expressed in meters, millimeters, or smaller
        # units; the rank check must not depend on an absolute coordinate scale.
        model = self.model * 1e-12
        translation = self.translation * 1e-12
        observed = (self.rotation @ model.T).T + translation
        R, t, metrics = fit_rigid_pattern(model, observed)
        np.testing.assert_allclose(R, self.rotation, atol=1e-10)
        np.testing.assert_allclose(t, translation, atol=1e-22)
        self.assertLess(metrics["rms_error"], 1e-10)

    def test_rejects_invalid_min_markers_values(self):
        for value in (2, 3.5, float("nan"), float("inf"), True):
            with self.subTest(min_markers=value):
                with self.assertRaisesRegex(ValueError, "min_markers"):
                    fit_rigid_pattern(
                        self.model, self.observed, min_markers=value
                    )

    def test_rejects_collinear_marker_subset(self):
        line_model = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        line_observed = line_model + [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "collinear"):
            fit_rigid_pattern(line_model, line_observed)

    def test_rejects_non_numeric_or_boolean_marker_indices(self):
        for indices in (["0", "1", "2"], [0.0, 1.0, 2.5], [False, True, False], [0 + 0j, 1 + 0j, 2 + 0j]):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "marker_indices"):
                    fit_rigid_pattern(
                        self.model, self.observed[:3], marker_indices=indices
                    )

    def test_rejects_boolean_marker_id_hidden_by_numpy_promotion(self):
        for indices in ([0, True, 2], [0, np.bool_(True), 2]):
            with self.subTest(indices=repr(indices)):
                with self.assertRaisesRegex(ValueError, "real integers"):
                    fit_rigid_pattern(
                        self.model, self.observed[:3], marker_indices=indices
                    )

    def test_rejects_nonreal_model_and_observed_coordinates(self):
        for value in ("0.0", True, np.bool_(False), 0.0 + 0.0j):
            for parameter in ("model_points", "observed_points"):
                with self.subTest(value=repr(value), parameter=parameter):
                    model = self.model.astype(object)
                    observed = self.observed.astype(object)
                    if parameter == "model_points":
                        model[0, 0] = value
                    else:
                        observed[0, 0] = value
                    with self.assertRaisesRegex(ValueError, "real numeric coordinates"):
                        fit_rigid_pattern(model, observed)

    def test_rejects_unrepresentable_integer_coordinates_and_indices(self):
        huge = 10**1000
        for parameter in ("model_points", "observed_points"):
            with self.subTest(parameter=parameter):
                model = self.model.astype(object)
                observed = self.observed.astype(object)
                if parameter == "model_points":
                    model[0, 0] = huge
                else:
                    observed[0, 0] = huge
                with self.assertRaisesRegex(ValueError, "finite float64 coordinates"):
                    fit_rigid_pattern(model, observed)
        with self.assertRaisesRegex(ValueError, "outside model_points"):
            fit_rigid_pattern(
                self.model, self.observed[:3],
                marker_indices=[0, 1, huge],
            )

    def test_rejects_invalid_max_rms_error_scalars(self):
        invalid = (
            True, np.bool_(False), "0.1", 0.1 + 0j,
            np.array([0.1]), 10**1000, float("nan"),
            float("inf"), float("-inf"), -0.1,
        )
        for limit in invalid:
            with self.subTest(value=repr(limit)):
                with self.assertRaisesRegex(ValueError, "max_rms_error"):
                    fit_rigid_pattern(
                        self.model, self.observed, max_rms_error=limit
                    )

    def test_integral_float_marker_indices_remain_supported(self):
        R, t, metrics = fit_rigid_pattern(
            self.model, self.observed[:3], marker_indices=[0.0, 1.0, 2.0]
        )
        np.testing.assert_allclose(R, self.rotation, atol=1e-10)
        np.testing.assert_allclose(t, self.translation, atol=1e-10)
        self.assertEqual(metrics["marker_indices"], [0, 1, 2])

    def test_rejects_duplicate_marker_identity(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_rigid_pattern(self.model, self.observed[:3], marker_indices=[0, 0, 2])

    def test_rejects_reflection_as_rotation(self):
        reflected = self.observed.copy()
        reflected[:, 0] *= -1
        R, _, _ = fit_rigid_pattern(self.model, reflected)
        self.assertGreater(np.linalg.det(R), 0.999999)


if __name__ == "__main__":
    unittest.main()
