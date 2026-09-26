"""saturate(I, f) 的独立对照测试。

期望值均为可手算的数学事实（分量去除 / 整环性质 / 可逆元），
不依赖被测实现内部路径；证书用独立展开核验。
"""

from __future__ import annotations

import unittest

from groebner import (
    InvalidGeneratorError,
    InvalidPolynomialError,
    Poly,
    check_s_polynomials,
    saturate,
)


def pow_poly(p: Poly, k: int) -> Poly:
    out = Poly.constant(1, p.n)
    for _ in range(k):
        out = out * p
    return out


def check_certificate(result, index):
    """独立核验：f**k * g == sum_j combination[j] * generators[j]。"""
    g = result.basis[index]
    cert = result.certificates[index]
    assert cert is not None
    lhs = pow_poly(result.saturator, cert.k) * g
    rhs = Poly.zero(result.saturator.n)
    for coef, gen in zip(cert.combination, result.generators):
        rhs = rhs + coef * gen
    return lhs == rhs


class TestSaturationMath(unittest.TestCase):
    def setUp(self):
        self.x = Poly.variable(2, 0)
        self.y = Poly.variable(2, 1)

    def test_xy_saturated_by_x_gives_y(self):
        # 手算：V(x*y) 去掉 x=0 分量后只剩 y=0，故 (x*y):x^inf = (y)。
        r = saturate([self.x * self.y], self.x)
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, (self.y,))
        self.assertEqual(r.certificates[0].k, 1)
        self.assertTrue(check_certificate(r, 0))

    def test_repeated_saturation_not_single_quotient(self):
        # 手算：(x^2*y):x = (x*y) 仍含 x=0 分量；(x^2*y):x^inf = (y)，
        # 一次理想商不等于全部饱和。
        r = saturate([self.x * self.x * self.y], self.x)
        self.assertEqual(r.basis, (self.y,))
        self.assertEqual(r.certificates[0].k, 2)
        self.assertTrue(check_certificate(r, 0))

    def test_univariate_power_saturates_to_unit(self):
        # 手算：(x^2):x^inf = (1)，因为 x^2 * 1 属于 (x^2)。
        r = saturate([self.x * self.x], self.x)
        self.assertEqual(r.basis, (Poly.constant(1, 2),))
        self.assertEqual(r.certificates[0].k, 2)
        self.assertTrue(check_certificate(r, 0))

    def test_mixed_components(self):
        # 手算：I = (x*z, y*z) = (z) 交 (x, y)；排除 z=0 后剩 (x, y)。
        z = Poly.variable(3, 2)
        x3 = Poly.variable(3, 0)
        y3 = Poly.variable(3, 1)
        r = saturate([x3 * z, y3 * z], z)
        self.assertTrue(r.is_complete)
        self.assertEqual(set(r.basis), {x3, y3})
        for i in range(len(r.basis)):
            self.assertTrue(check_certificate(r, i))
        # 输出基是 Groebner 基（S 多项式判据）。
        self.assertEqual(check_s_polynomials(list(r.basis)), [])

    def test_no_aux_variable_leak(self):
        r = saturate([self.x * self.y], self.x)
        for b in r.basis:
            self.assertEqual(b.n, 2)
            for exp in b.terms:
                self.assertEqual(len(exp), 2)

    def test_output_is_monic_reduced(self):
        z = Poly.variable(3, 2)
        x3 = Poly.variable(3, 0)
        y3 = Poly.variable(3, 1)
        r = saturate([2 * x3 * z, 4 * y3 * z], z)
        for b in r.basis:
            self.assertEqual(b.lc(), 1)


class TestSaturationSpecialCases(unittest.TestCase):
    def setUp(self):
        self.x = Poly.variable(2, 0)
        self.y = Poly.variable(2, 1)

    def test_zero_saturator_gives_unit_ideal(self):
        # f = 0：f^1 * g = 0 恒在 I 中，故饱和为单位理想。
        r = saturate([self.x * self.y], Poly.zero(2))
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, (Poly.constant(1, 2),))
        self.assertEqual(r.certificates[0].k, 1)

    def test_nonzero_constant_saturator_keeps_ideal(self):
        # f 为非零常数（可逆元）：I:f^inf = I，证书 k = 0。
        r = saturate([self.x * self.y], Poly.constant(3, 2))
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, (self.x * self.y,))
        self.assertEqual(r.certificates[0].k, 0)
        self.assertTrue(check_certificate(r, 0))

    def test_zero_ideal_empty_generators(self):
        # 零理想：整环中 f^k * g = 0 蕴含 g = 0，饱和仍为零理想。
        r = saturate([], self.x)
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, ())
        self.assertEqual(r.certificates, ())

    def test_zero_ideal_all_zero_generators(self):
        r = saturate([Poly.zero(2), Poly.zero(2)], self.x)
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, ())

    def test_unit_ideal_stays_unit(self):
        r = saturate([Poly.constant(1, 2), self.x], self.x)
        self.assertTrue(r.is_complete)
        self.assertEqual(r.basis, (Poly.constant(1, 2),))
        self.assertTrue(check_certificate(r, 0))

    def test_ideal_already_saturated(self):
        # 手算：(x, y) 对 z 已饱和（z 不是零因子）。
        z = Poly.variable(3, 2)
        x3 = Poly.variable(3, 0)
        y3 = Poly.variable(3, 1)
        r = saturate([x3, y3], z)
        self.assertEqual(set(r.basis), {x3, y3})
        for c in r.certificates:
            self.assertEqual(c.k, 0)


class TestSaturationBudgetAndValidation(unittest.TestCase):
    def setUp(self):
        self.x = Poly.variable(2, 0)
        self.y = Poly.variable(2, 1)

    def test_zero_budget_is_incomplete(self):
        r = saturate([self.x * self.y], self.x,
                     max_pairs=0, max_terms=0, max_steps=0)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.basis, ())
        self.assertTrue(r.reason)

    def test_certificate_over_budget_reported(self):
        # 证书换算超预算：基仍正确，证书为 None 且 reason 单独说明。
        r = saturate([self.x * self.x * self.y], self.x, max_terms=1)
        if any(c is None for c in r.certificates):
            self.assertTrue(r.reason)
            self.assertEqual(r.basis, (self.y,))

    def test_reject_non_poly_f(self):
        with self.assertRaises(InvalidPolynomialError):
            saturate([self.x], "x")

    def test_reject_mismatched_n(self):
        with self.assertRaises(InvalidGeneratorError):
            saturate([self.x], Poly.variable(3, 0))

    def test_reject_bad_generators(self):
        with self.assertRaises(InvalidGeneratorError):
            saturate([self.x, 1], self.x)
        with self.assertRaises(InvalidGeneratorError):
            saturate(self.x, self.x)

    def test_reject_bad_budget(self):
        with self.assertRaises(InvalidGeneratorError):
            saturate([self.x], self.x, max_pairs=-1)
        with self.assertRaises(InvalidGeneratorError):
            saturate([self.x], self.x, max_steps=True)


if __name__ == "__main__":
    unittest.main()

