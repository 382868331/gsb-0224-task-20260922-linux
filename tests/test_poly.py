"""Poly 数据结构、grlex 次序、四则运算、除法、S 多项式的测试。

期望值来自 tests/reference.py 的独立 dict 实现，不调用被测核心来生成期望。
"""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from groebner import (
    InvalidPolynomialError,
    Poly,
    ZeroDivisorError,
    ZeroPolynomialError,
    divide,
    s_polynomial,
)
from groebner.poly import lcm_monomial

import reference as ref


def P(term_dict, n):
    return Poly(term_dict, n=n)


def D(poly):
    """Poly -> 参考实现使用的 dict。"""
    return poly.terms


# 确定性的小多项式随机生成器（固定种子，供独立参考与被测两边共用输入）
def rand_poly(rng, n, max_deg, max_terms):
    exps = set()
    while len(exps) < rng.randint(1, max_terms):
        exps.add(tuple(rng.randint(0, max_deg) for _ in range(n)))
    d = {}
    for e in exps:
        c = Fraction(rng.randint(-3, 3), rng.randint(1, 3))
        if c != 0:
            d[e] = c
    return d


class TestConstructionAndOrder(unittest.TestCase):
    def test_zero_and_terms(self):
        self.assertTrue(P({}, 2).is_zero())
        self.assertEqual(len(P({}, 2)), 0)
        self.assertEqual(P({(0, 0): 5}, 2), Poly.constant(5, 2))

    def test_combine_like_terms_and_drop_zero(self):
        p = Poly([((1, 0), 2), ((1, 0), -2), ((0, 1), 1)], n=2)
        self.assertEqual(p, P({(0, 1): 1}, 2))
        p2 = Poly([((1, 0), Fraction(1, 2)), ((1, 0), Fraction(1, 2))], n=2)
        self.assertEqual(p2, P({(1, 0): 1}, 2))

    def test_float_finite_becomes_exact_fraction(self):
        p = P({(1, 0): 0.5, (0, 1): -0.25}, 2)
        self.assertEqual(p.monomial_coefficient((1, 0)), Fraction(1, 2))
        self.assertEqual(p.monomial_coefficient((0, 1)), Fraction(-1, 4))
        # -0.0 系数项被剔除
        self.assertTrue(P({(1, 0): -0.0}, 2).is_zero())

    def test_grlex_order(self):
        # x^2 > xy > y^2 > x > y > 1
        p = P({(2, 0): 1, (1, 1): 1, (0, 2): 1,
               (1, 0): 1, (0, 1): 1, (0, 0): 1}, 2)
        self.assertEqual([e for e, _ in p],
                         [(2, 0), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0)])

    def test_immutable_terms_snapshot(self):
        p = P({(1, 0): 1}, 2)
        t = p.terms
        t[(9, 9)] = Fraction(7)
        self.assertNotIn((9, 9), p.terms)

    def test_str(self):
        p = P({(2, 0): 1, (1, 1): -1, (0, 0): 1}, 2)
        self.assertEqual(p.to_string(("x", "y")), "x**2 - x*y + 1")

    def test_lcm_monomial(self):
        self.assertEqual(lcm_monomial((2, 1, 0), (1, 3, 1)), (2, 3, 1))


class TestArithmeticVsReference(unittest.TestCase):
    def test_operations(self):
        rng = random.Random(20260923)
        for _ in range(60):
            n = rng.randint(1, 4)
            a = rand_poly(rng, n, 2, 5)
            b = rand_poly(rng, n, 2, 5)
            pa, pb = P(a, n), P(b, n)
            self.assertEqual(D(pa + pb), ref.add(a, b))
            self.assertEqual(D(pa - pb), ref.sub(a, b))
            self.assertEqual(D(-pa), ref.neg(a))
            self.assertEqual(D(pa * pb), ref.mul(a, b))
            c = Fraction(rng.randint(-4, 4), rng.randint(1, 4))
            self.assertEqual(D(pa.scale(c)), ref.smul(c, a))

    def test_monic(self):
        p = P({(1, 0): Fraction(2, 3), (0, 0): -4}, 2)
        m = p.normalize_monic()
        self.assertEqual(m.lc(), Fraction(1))
        self.assertEqual(m, P({(1, 0): 1, (0, 0): -6}, 2))


class TestDivisionAndS(unittest.TestCase):
    def test_division_identity_and_remainder(self):
        rng = random.Random(4242)
        for _ in range(40):
            n = rng.randint(1, 3)
            f = rand_poly(rng, n, 2, 4)
            ds = [rand_poly(rng, n, 1, 3) for _ in range(rng.randint(1, 3))]
            ds = [d for d in ds if d]
            pf, pds = P(f, n), [P(d, n) for d in ds]
            res = divide(pf, pds)
            # 恒等式 f = sum q_i d_i + r（被测自身算术核验）
            rebuilt = res.remainder
            for q, d in zip(res.quotients, pds):
                rebuilt = rebuilt + q * d
            self.assertEqual(rebuilt, pf)
            # 余式与独立参考逐单项式一致（同样的除式顺序）
            _, rr = ref.divide_full(f, ds)
            self.assertEqual(D(res.remainder), rr)

    def test_division_remainder_depends_on_order(self):
        # 经典例子：xy^2-1 对 [xy+1, y^2-1] 与反过来给不同余式
        f = P({(1, 2): 1, (0, 0): -1}, 2)
        g1 = P({(1, 1): 1, (0, 0): 1}, 2)
        g2 = P({(0, 2): 1, (0, 0): -1}, 2)
        r1 = divide(f, [g1, g2]).remainder
        r2 = divide(f, [g2, g1]).remainder
        self.assertNotEqual(r1, r2)
        for r, ds in ((r1, [g1, g2]), (r2, [g2, g1])):
            res = divide(f, ds)
            self.assertEqual(res.remainder, r)

    def test_s_polynomial_vs_reference(self):
        rng = random.Random(99)
        made = 0
        while made < 40:
            n = rng.randint(1, 4)
            a = rand_poly(rng, n, 2, 4)
            b = rand_poly(rng, n, 2, 4)
            if not a or not b:
                continue
            made += 1
            got = D(s_polynomial(P(a, n), P(b, n)))
            self.assertEqual(got, ref.s_poly(a, b))

    def test_zero_divisor_rejected_with_location(self):
        f = P({(1, 0): 1}, 2)
        with self.assertRaises(ZeroDivisorError) as cm:
            divide(f, [P({(0, 1): 1}, 2), P({}, 2)])
        self.assertIn("1", str(cm.exception))

    def test_empty_divisors_returns_self_as_remainder(self):
        f = P({(1, 0): 2, (0, 0): 1}, 2)
        res = divide(f, [])
        self.assertEqual(res.remainder, f)
        self.assertEqual(res.quotients, ())
        self.assertEqual(res.identity(f, []), f)

    def test_s_poly_zero_rejected(self):
        with self.assertRaises(ZeroDivisorError):
            s_polynomial(P({}, 2), P({(1, 0): 1}, 2))


class TestInvalidInputs(unittest.TestCase):
    def test_reject_nan_infinity(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidPolynomialError) as cm:
                P({(1, 0): bad}, 2)
            self.assertIn("index=0", str(cm.exception))

    def test_reject_bool(self):
        with self.assertRaises(InvalidPolynomialError):
            P({(1, 0): True}, 2)
        with self.assertRaises(InvalidPolynomialError):
            P({(True, 0): 1}, 2)
        with self.assertRaises(InvalidPolynomialError):
            Poly.zero(True)
        with self.assertRaises(InvalidPolynomialError):
            Poly.variable(2, False)

    def test_reject_negative_and_bad_shape(self):
        with self.assertRaises(InvalidPolynomialError):
            P({(-1, 0): 1}, 2)
        with self.assertRaises(InvalidPolynomialError):
            P({(1, 0, 0): 1}, 2)  # 长度不符
        with self.assertRaises(InvalidPolynomialError):
            P({(1.0, 0): 1}, 2)  # 浮点指数
        with self.assertRaises(InvalidPolynomialError):
            Poly.zero(0)
        with self.assertRaises(InvalidPolynomialError):
            Poly.zero(5)

    def test_reject_bad_coefficient_type(self):
        with self.assertRaises(InvalidPolynomialError):
            P({(1, 0): "1"}, 2)

    def test_zero_poly_has_no_leading_term(self):
        with self.assertRaises(ZeroPolynomialError):
            P({}, 2).lm()
        with self.assertRaises(ZeroPolynomialError):
            P({}, 2).lc()

    def test_cross_variable_arithmetic_rejected(self):
        with self.assertRaises(InvalidPolynomialError):
            P({(1,): 1}, 1) + P({(1, 0): 1}, 2)


if __name__ == "__main__":
    unittest.main()
