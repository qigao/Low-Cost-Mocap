import unittest

import numpy as np

from multiview_association import associate_pairwise, sampson_errors


class PairwiseAssociationTests(unittest.TestCase):
    def setUp(self):
        # Rectified horizontal stereo: corresponding points share y.
        self.F = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

    def test_sampson_error_is_zero_for_epipolar_consistent_points(self):
        left = [[100.0, 50.0]]
        right = [[130.0, 50.0]]
        self.assertAlmostEqual(sampson_errors(left, right, self.F)[0, 0], 0.0)

    def test_assignment_is_one_to_one(self):
        left = [[100.0, 50.0], [200.0, 80.0]]
        right = [[210.0, 80.0], [110.0, 50.0]]
        matches = associate_pairwise(left, right, self.F, max_sampson_error_px=1.0)
        self.assertEqual([(m.left_index, m.right_index) for m in matches], [(0, 1), (1, 0)])
        self.assertEqual(len({m.right_index for m in matches}), len(matches))

    def test_bad_candidates_remain_unmatched(self):
        left = [[100.0, 50.0], [200.0, 80.0]]
        right = [[110.0, 50.0], [210.0, 130.0]]
        matches = associate_pairwise(left, right, self.F, max_sampson_error_px=2.0)
        self.assertEqual([(m.left_index, m.right_index) for m in matches], [(0, 0)])

    def test_candidate_just_above_threshold_is_rejected(self):
        # For this F, vertical displacement d gives Sampson error d^2 / 2.
        # Stay measurably above the limit so float64 rounding cannot change
        # which side of the threshold the candidate falls on.
        left = [[100.0, 50.0]]
        right = [[100.0, 50.0 + np.sqrt(2.0) * (1.0 + 1e-6)]]
        matches = associate_pairwise(
            left, right, self.F, max_sampson_error_px=1.0
        )
        self.assertEqual(matches, [])

    def test_candidate_just_below_threshold_is_accepted(self):
        left = [[100.0, 50.0]]
        right = [[100.0, 50.0 + np.sqrt(2.0) * (1.0 - 1e-6)]]
        matches = associate_pairwise(
            left, right, self.F, max_sampson_error_px=1.0
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(matches[0].sampson_error_px, 1.0)

    def test_assignment_maximizes_valid_match_count_before_error(self):
        # The exact match (left 0 -> right 0) conflicts with two valid
        # cross-matches. Prefer two sub-threshold matches over one zero-cost
        # match plus an unmatched detection.
        offset = np.sqrt(1.8)
        left = [[100.0, 50.0], [100.0, 50.0 - offset]]
        right = [[100.0, 50.0], [100.0, 50.0 + offset]]
        matches = associate_pairwise(
            left, right, self.F, max_sampson_error_px=1.0
        )
        self.assertEqual(
            [(m.left_index, m.right_index) for m in matches], [(0, 1), (1, 0)]
        )
        for match in matches:
            self.assertLess(match.sampson_error_px, 1.0)

    def test_empty_inputs_return_no_matches(self):
        self.assertEqual(associate_pairwise([], [[10.0, 20.0]], self.F), [])
        self.assertEqual(associate_pairwise([[10.0, 20.0]], [], self.F), [])

    def test_empty_point_arrays_must_have_valid_shape(self):
        self.assertEqual(sampson_errors(np.empty((0, 2)), [[1.0, 2.0]], self.F).shape, (0, 1))
        for malformed in (np.empty((2, 0)), np.empty((0, 3))):
            with self.subTest(shape=malformed.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    sampson_errors(malformed, [[1.0, 2.0]], self.F)

    def test_rejects_finite_coordinates_outside_safe_sampson_range(self):
        with self.assertRaisesRegex(ValueError, "safe float64 Sampson range"):
            sampson_errors([[1e200, 1e200]], [[-1e200, 1e200]], self.F)

    def test_association_rejects_coordinates_outside_safe_sampson_range(self):
        with self.assertRaisesRegex(ValueError, "safe float64 Sampson range"):
            associate_pairwise(
                [[1e200, 1e200]], [[-1e200, 1e200]], self.F
            )

    def test_rejects_non_real_numeric_point_coordinates(self):
        invalid = (
            [["100", "50"]],
            [[True, False]],
            [[100, True]],
            [[np.float64(100.0), np.bool_(False)]],
            [[100.0 + 0j, 50.0 + 0j]],
        )
        for points in invalid:
            with self.subTest(points=repr(points)):
                with self.assertRaisesRegex(ValueError, "real numeric coordinates"):
                    sampson_errors(points, [[130.0, 50.0]], self.F)
                with self.assertRaisesRegex(ValueError, "real numeric coordinates"):
                    associate_pairwise(points, [[130.0, 50.0]], self.F)

    def test_rejects_non_finite_points(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            associate_pairwise([[float("nan"), 2.0]], [[1.0, 2.0]], self.F)

    def test_sampson_errors_are_invariant_to_fundamental_matrix_scale(self):
        left = [[100.0, 50.0], [200.0, 80.0]]
        right = [[130.0, 50.0], [210.0, 80.0]]
        baseline = sampson_errors(left, right, self.F)
        for scale in (1e-12, 1e12):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(
                    sampson_errors(left, right, self.F * scale), baseline,
                    rtol=1e-12, atol=1e-12,
                )

    def test_rejects_threshold_whose_square_overflows(self):
        with self.assertRaisesRegex(ValueError, "squared must be positive and safely representable"):
            associate_pairwise(
                [[100.0, 50.0]], [[130.0, 50.0]], self.F,
                max_sampson_error_px=1e200,
            )

    def test_rejects_threshold_whose_assignment_cost_scaling_overflows(self):
        with self.assertRaisesRegex(ValueError, "squared must be positive and safely representable"):
            associate_pairwise(
                [[100.0, 50.0]], [[130.0, 50.0]], self.F,
                max_sampson_error_px=1e154,
            )

    def test_rejects_threshold_whose_square_underflows(self):
        with self.assertRaisesRegex(ValueError, "squared must be positive and safely representable"):
            associate_pairwise(
                [[100.0, 50.0]], [[130.0, 50.0]], self.F,
                max_sampson_error_px=1e-200,
            )

    def test_rejects_non_real_or_non_scalar_thresholds(self):
        for threshold in (True, np.bool_(False), "1.0", 1.0 + 0j, np.array([1.0]), 10**1000):
            with self.subTest(threshold=repr(threshold)):
                with self.assertRaisesRegex(ValueError, "finite positive real scalar"):
                    associate_pairwise(
                        [[100.0, 50.0]], [[130.0, 50.0]], self.F,
                        max_sampson_error_px=threshold,
                    )

    def test_rejects_non_real_fundamental_matrix_values(self):
        invalid = (
            self.F.astype(str),
            self.F.astype(bool),
            self.F.astype(complex) + 1j,
        )
        for fundamental in invalid:
            with self.subTest(dtype=fundamental.dtype):
                with self.assertRaisesRegex(ValueError, "fundamental matrix must contain real numeric values"):
                    sampson_errors([[100.0, 50.0]], [[130.0, 50.0]], fundamental)

    def test_rejects_zero_fundamental_matrix(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            sampson_errors([[100.0, 50.0]], [[130.0, 50.0]], np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
