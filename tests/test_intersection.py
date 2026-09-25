"""ideal_intersection 的消元构造、双组合证书、边界与预算测试。"""

from __future__ import annotations

import unittest

from groebner import (
    InvalidGeneratorError,
    Poly,
    ideal_intersection,
    membership,
)


def expand(certificate, generators):
    total = Poly.zero(generators[0].n)
    for coefficient, generator in zip(certificate, generators):
        total = total + coefficient * generator
    return total


class TestIdealIntersection(unittest.TestCase):
    def test_univariate_principal_ideals_manual_case(self):
        x = Poly.variable(1, 0)
        one = Poly.constant(1, 1)
        factor = x - one
        ideal_i = [factor * (x - one)]
        ideal_j = [factor * (x + one)]

        result = ideal_intersection(ideal_i, ideal_j)

        self.assertEqual(result.status, "complete")
        expected = factor * (x - one) * (x + one)
        self.assertEqual(result.basis, (expected,))
        self.assertEqual(len(result.certificates), 1)
        certificate_i, certificate_j = result.certificates[0]
        # 两份证书独立展开核对：
        # h = (x+1)*(x-1)^2 = (x-1)*(x-1)(x+1)。
        self.assertEqual(expand(certificate_i, ideal_i), expected)
        self.assertEqual(expand(certificate_j, ideal_j), expected)
        self.assertEqual(certificate_i, (x + one,))
        self.assertEqual(certificate_j, (x - one,))

    def test_two_variable_shared_factor_manual_case(self):
        x = Poly.variable(2, 0)
        y = Poly.variable(2, 1)
        one = Poly.constant(1, 2)
        shared = x * y
        ideal_i = [shared, x * x]
        ideal_j = [shared, y * y]

        result = ideal_intersection(ideal_i, ideal_j)

        self.assertEqual(result.status, "complete")
        # 手算：<xy,x^2> ∩ <xy,y^2> = <xy>。
        self.assertEqual(result.basis, (shared,))
        certificate_i, certificate_j = result.certificates[0]
        self.assertEqual(expand(certificate_i, ideal_i), shared)
        self.assertEqual(expand(certificate_j, ideal_j), shared)
        self.assertEqual(certificate_i, (one, Poly.zero(2)))
        self.assertEqual(certificate_j, (one, Poly.zero(2)))

    def test_every_returned_generator_reduces_to_zero_on_both_sides(self):
        x = Poly.variable(2, 0)
        y = Poly.variable(2, 1)
        ideal_i = [x * y - x, x * x - y]
        ideal_j = [x * y - x, y * y + x]

        result = ideal_intersection(ideal_i, ideal_j)

        self.assertEqual(result.status, "complete")
        for generator, (certificate_i, certificate_j) in zip(
                result.basis, result.certificates):
            self.assertEqual(membership(generator, ideal_i).status, "member")
            self.assertEqual(membership(generator, ideal_j).status, "member")
            self.assertEqual(expand(certificate_i, ideal_i), generator)
            self.assertEqual(expand(certificate_j, ideal_j), generator)

    def test_zero_ideals(self):
        x = Poly.variable(2, 0)
        y = Poly.variable(2, 1)
        ideal_i = [x, y]
        zero_ideal = [Poly.zero(2)]

        result = ideal_intersection(ideal_i, zero_ideal)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.basis, ())
        self.assertEqual(result.certificates, ())

        result_both_zero = ideal_intersection(
            [Poly.zero(2)], [Poly.zero(2)])
        self.assertEqual(result_both_zero.status, "complete")
        self.assertEqual(result_both_zero.basis, ())

    def test_unit_ideal(self):
        x = Poly.variable(2, 0)
        one = Poly.constant(1, 2)
        result = ideal_intersection([one, x], [one, x * x])
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.basis, (one,))
        certificate_i, certificate_j = result.certificates[0]
        self.assertEqual(expand(certificate_i, [one, x]), one)
        self.assertEqual(expand(certificate_j, [one, x * x]), one)

    def test_duplicate_generators(self):
        x = Poly.variable(1, 0)
        x2 = Poly.variable(2, 0)
        y2 = Poly.variable(2, 1)
        univariate = ideal_intersection([x * x, x * x], [x * x, x])
        self.assertEqual(univariate.status, "complete")
        self.assertEqual(univariate.basis, (x * x,))

        bivariate = ideal_intersection(
            [x2 * y2, x2 * y2, x2 * x2],
            [x2 * y2, y2 * y2, x2 * y2])
        self.assertEqual(bivariate.status, "complete")
        self.assertEqual(bivariate.basis, (x2 * y2,))

    def test_incomplete_never_reports_partial_t_free_basis(self):
        x = Poly.variable(2, 0)
        y = Poly.variable(2, 1)
        ideal_i = [x * y - x, x * x - y]
        ideal_j = [x * y - x, y * y + x]

        result = ideal_intersection(ideal_i, ideal_j, max_pairs=0)

        self.assertEqual(result.elimination.status, "incomplete")
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.basis, ())
        self.assertEqual(result.certificates, ())
        self.assertTrue(result.reason)

    def test_validation_limits(self):
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([], [])
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection(
                [Poly.variable(4, 0)], [Poly.variable(4, 0)])
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection(
                [Poly.variable(2, 0)], [Poly.variable(3, 0)])
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection(Poly.variable(2, 0), [Poly.variable(2, 0)])
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection(
                [Poly.variable(2, 0)], [Poly.variable(2, 0)], max_pairs=-1)


if __name__ == "__main__":
    unittest.main()
