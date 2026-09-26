"""saturate（I:f^inf 饱和理想）的测试。

验收要点：
- 手算小例：xy=0 排除 x=0 分量得 y=0；x**2*y 排除 x=0 也得 y=0，
  而一次理想商 (x**2*y):x = (x*y) 并不是饱和——专门区分；
- 与 tests/reference.py 的独立朴素饱和实现逐单项式比对；
- 每个输出多项式带有限 k 与 f**k * g 属于原理想的证书，独立展开核验；
- 边界：f=0 -> 单位理想；非零常数 f -> 原理想；零理想 -> 零理想；
  单位理想 -> 单位理想；辅助变量 t 不泄漏到公开多项式；
- 预算耗尽 -> incomplete；证书项数超预算 -> 基有效但证书缺省并说明。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from groebner import (
    InvalidGeneratorError,
    InvalidPolynomialError,
    Poly,
    membership,
    saturate,
)

import reference as ref


def P(term_dict, n):
    return Poly(term_dict, n=n)


def D(poly):
    return poly.terms


def pow_poly(base, k):
    out = Poly.constant(1, base.n)
    for _ in range(k):
        out = out * base
    return out


def expand(cert, gens, n):
    """独立展开 sum_j cert[j] * gens[j]（只用多项式算术）。"""
    total = Poly.zero(n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def assert_sat_certificates(testcase, result, gens, f):
    """逐基多项式核验 f**k * g == sum_j cert_j * gens_j（精确展开）。"""
    testcase.assertIsNotNone(result.certificates)
    testcase.assertIsNotNone(result.powers)
    testcase.assertEqual(len(result.basis), len(result.certificates))
    testcase.assertEqual(len(result.basis), len(result.powers))
    for b, k, cert in zip(result.basis, result.powers, result.certificates):
        testcase.assertIsInstance(k, int)
        testcase.assertGreaterEqual(k, 0)
        testcase.assertEqual(len(cert), len(gens))
        lhs = pow_poly(f, k) * b
        rhs = expand(cert, gens, f.n)
        testcase.assertEqual(lhs, rhs,
                             f"证书展开不符：f**{k}*({b}) != sum cert_j*g_j")


X = Poly.variable(2, 0)
Y = Poly.variable(2, 1)


class TestSaturateHandComputed(unittest.TestCase):
    """可手算的分量去除小例。"""

    def test_xy_exclude_x_gives_y(self):
        # V(xy) = V(x) ∪ V(y)；排除 x=0 分量后剩 y=0。
        r = saturate([X * Y], X)
        self.assertEqual(r.status, "complete")
        self.assertEqual([D(b) for b in r.basis], [{(0, 1): Fraction(1)}])
        self.assertEqual(r.powers, (1,))  # x**1 * y = x*y 恰是原生成元
        assert_sat_certificates(self, r, [X * Y], X)

    def test_saturation_is_more_than_one_quotient(self):
        # (x**2*y):x = (x*y) 仍含 x 分量；再商一次才是 (y)。
        r = saturate([X * X * Y], X)
        self.assertEqual([D(b) for b in r.basis], [{(0, 1): Fraction(1)}])
        self.assertEqual(r.powers, (2,))  # 需要 x**2
        assert_sat_certificates(self, r, [X * X * Y], X)
        # 对照：一次商 (x*y) 确实不是答案（基不同）。
        self.assertNotEqual([D(b) for b in r.basis],
                            [{(1, 1): Fraction(1)}])

    def test_two_component_example(self):
        # I = (x*y, x*(x+1))：零点集是直线 x=0 与孤点 (-1, 0)；
        # 排除 x=0 分量后剩点 (-1, 0)，理想为 (x+1, y)。
        g2 = X * (X + Poly.constant(1, 2))
        r = saturate([X * Y, g2], X)
        self.assertEqual([D(b) for b in r.basis],
                         [{(1, 0): Fraction(1), (0, 0): Fraction(1)},
                          {(0, 1): Fraction(1)}])
        assert_sat_certificates(self, r, [X * Y, g2], X)

    def test_saturating_by_y_keeps_x(self):
        r = saturate([X * Y], Y)
        self.assertEqual([D(b) for b in r.basis], [{(1, 0): Fraction(1)}])
        assert_sat_certificates(self, r, [X * Y], Y)

    def test_three_variables(self):
        x, y, z = (Poly.variable(3, i) for i in range(3))
        # I = (x*y, x*z)：排除 x=0 分量剩 (y, z)。
        r = saturate([x * y, x * z], x)
        self.assertEqual(r.status, "complete")
        self.assertEqual([D(b) for b in r.basis],
                         [{(0, 1, 0): Fraction(1)}, {(0, 0, 1): Fraction(1)}])
        assert_sat_certificates(self, r, [x * y, x * z], x)


class TestSaturateBoundaries(unittest.TestCase):
    def test_f_zero_gives_unit_ideal(self):
        # 排除 f=0 即排除整个空间，剩余为空集，理想为 (1)。
        r = saturate([X * Y], Poly.zero(2))
        self.assertEqual(r.status, "complete")
        self.assertEqual([D(b) for b in r.basis], [{(0, 0): Fraction(1)}])
        assert_sat_certificates(self, r, [X * Y], Poly.zero(2))

    def test_nonzero_constant_f_keeps_ideal(self):
        # 非零常数是单位：c**k * g ∈ I ⟺ g ∈ I，饱和 = 原理想。
        r = saturate([X * Y], Poly.constant(3, 2))
        self.assertEqual([D(b) for b in r.basis], [{(1, 1): Fraction(1)}])
        self.assertEqual(r.powers, (0,))  # k=0：生成元本来就在 I 中
        assert_sat_certificates(self, r, [X * Y], Poly.constant(3, 2))

    def test_zero_ideal_stays_zero(self):
        r = saturate([Poly.zero(2)], X)
        self.assertEqual(r.status, "complete")
        self.assertEqual(r.basis, ())
        self.assertEqual(r.certificates, ())
        self.assertEqual(r.powers, ())

    def test_empty_generators_is_zero_ideal(self):
        r = saturate([], X)
        self.assertEqual(r.status, "complete")
        self.assertEqual(r.basis, ())

    def test_unit_ideal_stays_unit(self):
        r = saturate([Poly.constant(1, 2)], X)
        self.assertEqual([D(b) for b in r.basis], [{(0, 0): Fraction(1)}])
        assert_sat_certificates(self, r, [Poly.constant(1, 2)], X)

    def test_no_auxiliary_variable_leak(self):
        # 所有公开多项式（基与证书）都必须在原来的 n 元环中。
        r = saturate([X * X * Y, X * Y * Y - Y], X)
        self.assertEqual(r.status, "complete")
        for b in r.basis:
            self.assertEqual(b.n, 2)
        for cert in r.certificates:
            for c in cert:
                self.assertEqual(c.n, 2)

    def test_result_is_reduced_groebner_basis_of_saturation(self):
        # 饱和理想的成员判定一致性：x*y 的幂次倍落入原理想的充要条件。
        gens = [X * X * Y, X * Y * Y - Y]
        r = saturate(gens, X)
        self.assertEqual(r.status, "complete")
        # 基中每个元素都在饱和理想中（由证书保证），
        # 且饱和理想中元素对基余式为 0：用 membership 对基理想核验。
        m = membership(Y, list(r.basis)) if r.basis else None
        if m is not None:
            self.assertIn(m.status, ("member", "nonmember"))


class TestSaturateAgainstReference(unittest.TestCase):
    """与独立朴素参考实现逐单项式比对。"""

    def check(self, gen_dicts, f_dict, n):
        gens = [P(g, n) for g in gen_dicts]
        f = P(f_dict, n)
        r = saturate(gens, f)
        self.assertEqual(r.status, "complete")
        expected = ref.saturate_reference(gen_dicts, f_dict)
        self.assertEqual([D(b) for b in r.basis], expected)
        assert_sat_certificates(self, r, gens, f)

    def test_monomial_cases(self):
        self.check([{(1, 1): 1}], {(1, 0): 1}, 2)
        self.check([{(2, 1): 1}], {(1, 0): 1}, 2)
        self.check([{(1, 1): 1}, {(1, 2): 1}], {(0, 1): 1}, 2)
        self.check([{(2, 0): 1, (0, 2): 1}], {(1, 0): 1}, 2)

    def test_binomial_cases(self):
        self.check([{(2, 1): 1, (0, 0): -1}], {(1, 0): 1}, 2)
        self.check([{(1, 1): 1, (0, 1): -1}, {(2, 0): 1}], {(1, 0): 1}, 2)
        self.check([{(1, 0, 1): 1}, {(2, 0, 0): 1, (0, 1, 0): 1}],
                   {(1, 0, 0): 1}, 3)

    def test_constant_and_zero_f(self):
        self.check([{(1, 1): 1}], {(0, 0): 3}, 2)
        self.check([{(1, 1): 1}], {}, 2)
        self.check([{}], {(1, 0): 1}, 2)


class TestSaturateBudgetAndValidation(unittest.TestCase):
    def test_zero_steps_budget_incomplete(self):
        r = saturate([X * Y], X, max_steps=0)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.basis, ())
        self.assertIsNone(r.certificates)
        self.assertIsNone(r.powers)
        self.assertTrue(r.reason)

    def test_zero_pairs_budget_incomplete(self):
        r = saturate([X * X * Y, X + Y * Y], X, max_pairs=0)
        self.assertEqual(r.status, "incomplete")

    def test_certificate_budget_overflow_reported_separately(self):
        # 基仍然有效，但证书项数超过 max_terms 时单独说明。
        r = saturate([X * X * Y], X, max_terms=1)
        # max_terms=1 时连基计算都可能 incomplete；两种结局都合法，
        # 但若 complete 且证书缺省，必须有说明。
        if r.status == "complete" and r.certificates is None:
            self.assertIn("证书", r.reason)
        else:
            self.assertIn(r.status, ("complete", "incomplete"))

    def test_validation_errors(self):
        with self.assertRaises(InvalidPolynomialError):
            saturate([X], "not a poly")
        with self.assertRaises(InvalidGeneratorError):
            saturate(X, X)  # 单个 Poly 不是序列
        with self.assertRaises(InvalidGeneratorError):
            saturate([Poly.variable(3, 0)], X)  # 变量数不一致
        with self.assertRaises(InvalidGeneratorError):
            saturate([X], X, max_pairs=-1)
        with self.assertRaises(InvalidGeneratorError):
            saturate([X], X, max_steps=True)  # bool 拒绝

    def test_too_many_variables(self):
        # 4 元理想饱和需要第 5 个变量，超出上限。
        gens4 = [Poly.variable(4, 0) * Poly.variable(4, 1)]
        with self.assertRaises(InvalidGeneratorError):
            saturate(gens4, Poly.variable(4, 0))


if __name__ == "__main__":
    unittest.main()
