"""Poly 数据结构、grlex 次序与输入校验测试。"""

import math
import unittest
from fractions import Fraction

from groebner import Poly, PolynomialError, MAX_VARIABLES, grlex_key
from groebner.polynomial import lcm_monomial, divides_monomial


class TestPolyConstruction(unittest.TestCase):
    def test_zero_and_constants(self):
        z = Poly.zero(2)
        self.assertTrue(z.is_zero())
        self.assertEqual(len(z), 0)
        self.assertEqual(z.nvars, 2)
        self.assertTrue(Poly.constant(0, 2).is_zero())
        one = Poly.constant(1, 3)
        self.assertEqual(one.leading_monomial(), (0, 0, 0))
        self.assertEqual(one.leading_coefficient(), Fraction(1))

    def test_var(self):
        x = Poly.var(0, 2)
        self.assertEqual(x.leading_monomial(), (1, 0))
        y = Poly.var(1, 2)
        self.assertEqual(y.leading_monomial(), (0, 1))
        self.assertTrue(x != y)

    def test_dict_and_terms_merge_zero(self):
        # 注意：dict 字面量会自动去重，重复键合并要走 from_terms
        p = Poly.from_terms([((1, 0), 1), ((1, 0), 2), ((0, 1), -3)], nvars=2)
        self.assertEqual(p.coefficient((1, 0)), Fraction(3))
        p2 = Poly.from_terms([((1, 0), 1), ((1, 0), -1)], nvars=2)
        self.assertTrue(p2.is_zero())

    def test_rational_coefficients(self):
        p = Poly({(1, 0): Fraction(1, 2), (0, 1): "3/4"}, nvars=2)
        self.assertEqual(p.coefficient((1, 0)), Fraction(1, 2))
        self.assertEqual(p.coefficient((0, 1)), Fraction(3, 4))
        # 有限 float 被精确转换（不产生 0.300000...0003 一类误差）
        q = Poly({(0,): 0.5}, nvars=1)
        self.assertEqual(q.coefficient((0,)), Fraction(1, 2))


class TestGrlexOrdering(unittest.TestCase):
    def test_total_degree_first(self):
        # 次数 3 恒大于次数 2
        self.assertGreater(grlex_key((3, 0)), grlex_key((0, 2)))
        self.assertGreater(grlex_key((1, 1, 1)), grlex_key((2, 0, 0)))

    def test_lex_tiebreak(self):
        # 同次数：指数向量字典序
        self.assertGreater(grlex_key((2, 0)), grlex_key((1, 1)))
        self.assertGreater(grlex_key((1, 1, 0)), grlex_key((1, 0, 1)))

    def test_leading_term_of_poly(self):
        p = Poly({(0, 2): 1, (2, 0): 1, (1, 1): 1, (3, 0): -5}, nvars=2)
        self.assertEqual(p.leading_monomial(), (3, 0))
        self.assertEqual(p.leading_coefficient(), Fraction(-5))
        # 去掉 3 次项后，2 次项里 (2,0) 最大
        q = Poly({(0, 2): 1, (2, 0): 1, (1, 1): 1}, nvars=2)
        self.assertEqual(q.leading_monomial(), (2, 0))


class TestArithmetic(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)

    def test_add_sub_scalar(self):
        p = self.x + 1
        self.assertEqual(p.coefficient((0, 0)), Fraction(1))
        p2 = 2 - self.x
        self.assertEqual(p2.coefficient((1, 0)), Fraction(-1))
        self.assertEqual(p2.coefficient((0, 0)), Fraction(2))
        self.assertTrue((self.x - self.x).is_zero())

    def test_mul_and_pow(self):
        self.assertEqual((self.x * self.y).leading_monomial(), (1, 1))
        x3 = self.x ** 3
        self.assertEqual(x3.leading_monomial(), (3, 0))
        self.assertEqual(self.x ** 0, Poly.constant(1, 2))
        with self.assertRaises(PolynomialError):
            _ = self.x ** -1
        with self.assertRaises(PolynomialError):
            _ = self.x ** True  # type: ignore[operator]

    def test_scalar_ops(self):
        p = Poly({(1, 0): Fraction(2, 3)}, nvars=2)
        q = p / 2
        self.assertEqual(q.coefficient((1, 0)), Fraction(1, 3))
        self.assertTrue((0 * p).is_zero())
        with self.assertRaises(PolynomialError):
            _ = p / 0

    def test_mixed_nvars_rejected(self):
        z = Poly.var(0, 3)
        with self.assertRaises(PolynomialError):
            _ = self.x + z
        with self.assertRaises(PolynomialError):
            _ = self.x * z

    def test_equality_and_hash(self):
        a = Poly({(1, 0): 1, (0, 1): 2}, nvars=2)
        b = Poly({(0, 1): 2, (1, 0): 1}, nvars=2)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_monomial_helpers(self):
        self.assertEqual(lcm_monomial((2, 0, 1), (1, 3, 1)), (2, 3, 1))
        self.assertTrue(divides_monomial((1, 0), (2, 1)))
        self.assertFalse(divides_monomial((0, 1), (2, 0)))


class TestInputValidation(unittest.TestCase):
    def test_reject_nan_infinity(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(PolynomialError):
                Poly({(0,): bad}, nvars=1)
        for bad in ("nan", "inf", "-inf", "Infinity"):
            with self.assertRaises(PolynomialError):
                Poly({(0,): bad}, nvars=1)
        self.assertTrue(math.isnan(float("nan")))  # 确认坏值确实来自 IEEE 特殊值

    def test_reject_bool(self):
        with self.assertRaises(PolynomialError):
            Poly({(0,): True}, nvars=1)
        with self.assertRaises(PolynomialError):
            Poly({(True, 0): 1}, nvars=2)
        with self.assertRaises(PolynomialError):
            Poly.var(False, 2)  # type: ignore[arg-type]
        with self.assertRaises(PolynomialError):
            _ = Poly.zero(True)  # type: ignore[arg-type]

    def test_reject_bad_nvars(self):
        for bad in (0, MAX_VARIABLES + 1, -1):
            with self.assertRaises(PolynomialError):
                Poly.zero(bad)

    def test_reject_bad_exponents(self):
        with self.assertRaises(PolynomialError):
            Poly({(1,): 1}, nvars=2)          # 长度不符
        with self.assertRaises(PolynomialError):
            Poly({(-1, 0): 1}, nvars=2)        # 负指数
        with self.assertRaises(PolynomialError):
            Poly({("1", 0): 1}, nvars=2)       # 非整数
        with self.assertRaises(PolynomialError):
            Poly({}, nvars=None)  # type: ignore[arg-type]

    def test_reject_bad_coefficient_type(self):
        with self.assertRaises(PolynomialError):
            Poly({(0,): [1]}, nvars=1)
        with self.assertRaises(PolynomialError):
            Poly({(0,): object()}, nvars=1)
        with self.assertRaises(PolynomialError):
            Poly({(0,): "not-a-fraction"}, nvars=1)

    def test_error_messages_are_locatable(self):
        try:
            Poly({(-1, 0): 1}, nvars=2)
        except PolynomialError as exc:
            self.assertIn("多项式项[0]", str(exc))
        try:
            Poly({(0, 0): float("inf")}, nvars=2)
        except PolynomialError as exc:
            self.assertIn("(0, 0)", str(exc))

    def test_immutability_of_snapshots(self):
        p = Poly({(1, 0): 1}, nvars=2)
        terms = p.terms()
        terms.append((("hack",), Fraction(9)))
        self.assertEqual(len(p.terms()), 1)
        self.assertEqual(p.coefficient((1, 0)), Fraction(1))


if __name__ == "__main__":
    unittest.main()
