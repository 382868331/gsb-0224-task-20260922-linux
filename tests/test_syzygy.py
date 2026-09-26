"""bounded_syzygies 的测试。

验收要点对应：
- 每条基关系用 tests/reference.py 的独立 dict 展开核验 sum a_i f_i == 0；
- 维数完备性由独立搭建的矩阵 + 独立求秩证明：
  dimension == num_unknowns - rank；
- 小例 (x, y) 在 d=1 找到 (y, -x)（至多差整体符号），d=0 时基为空；
- 零生成元、重复生成元、常数生成元、输入排列的坐标跟随；
- 旧 Groebner 算法行为不受影响（由 test_groebner.py 原样保留保证）。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from groebner import (
    BoundedSyzygyResult,
    InvalidGeneratorError,
    Poly,
    bounded_syzygies,
    relation_polynomial,
)

import reference as ref


def P(term_dict, n=2):
    return Poly(term_dict, n=n)


def to_dicts(gens):
    return [g.terms for g in gens]


def assert_relation_zero(testcase, coeffs, gens):
    """独立展开核验：sum a_i * f_i 逐项相乘求和必须为 0。"""
    total = ref.expand([c.terms for c in coeffs], to_dicts(gens))
    testcase.assertEqual(total, {}, f"关系展开非零: {total}")


def assert_space_complete(testcase, result, gens, n):
    """独立搭矩阵、独立求秩，证明基空间维数完备。"""
    rows, columns = ref.syzygy_matrix(to_dicts(gens), result.max_degree, n)
    testcase.assertEqual(tuple(columns), result.column_monomials)
    testcase.assertEqual(len(columns), result.num_unknowns)
    testcase.assertEqual(len(rows), result.num_equations)
    rank = ref.rank(rows)
    testcase.assertEqual(rank, result.rank)
    testcase.assertEqual(result.dimension,
                         result.num_unknowns - rank)
    testcase.assertEqual(result.dimension, len(result.basis))


def check_result(testcase, result, gens, n=2):
    """对结果做全部独立核验：逐关系展开为 0 + 维数公式完备性。"""
    testcase.assertIsInstance(result, BoundedSyzygyResult)
    testcase.assertEqual(result.generators, tuple(gens))
    for coeffs in result.basis:
        testcase.assertEqual(len(coeffs), len(gens))
        assert_relation_zero(testcase, coeffs, gens)
        # 次数界：每个系数多项式的次数都不超过 max_degree
        for c in coeffs:
            for exp in c.terms:
                testcase.assertLessEqual(sum(exp), result.max_degree)
    assert_space_complete(testcase, result, gens, n)


X = P({(1, 0): 1})
Y = P({(0, 1): 1})


class TestSmallExample(unittest.TestCase):
    def test_xy_degree1_finds_koszul(self):
        gens = [X, Y]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        self.assertEqual(result.dimension, 1)
        (a0, a1), = result.basis
        # 规范基向量应是 (y, -x) 或其相反数（自由列取 1 决定符号）
        self.assertIn((a0, a1), [(Y, -X), (-Y, X)])

    def test_xy_degree0_no_relation(self):
        # d 太小：常数系数下 x、y 无线性关系，不能捏造关系
        result = bounded_syzygies([X, Y], 0)
        check_result(self, result, [X, Y])
        self.assertEqual(result.dimension, 0)
        self.assertEqual(result.basis, ())

    def test_higher_degree_contains_more(self):
        gens = [P({(2, 0): 1}), P({(1, 1): 1}), P({(0, 2): 1})]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        # Koszul 关系 (y,-x,0) 与 (0,y,-x)
        self.assertEqual(result.dimension, 2)


class TestSpecialGenerators(unittest.TestCase):
    def test_zero_generator_free_columns(self):
        zero = Poly.zero(2)
        gens = [zero, X]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        # a0 任意（3 个次数<=1 的单项式），a1 * x = 0 迫使 a1 = 0
        self.assertEqual(result.dimension, 3)
        for a0, a1 in result.basis:
            self.assertTrue(a1.is_zero())

    def test_duplicate_generators(self):
        gens = [X, X]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        # a0 + a1 = 0，deg <= 1 的二元多项式空间维数 3
        self.assertEqual(result.dimension, 3)
        # 必含 (1, -1) 方向的常数关系：基向量的常数部分张成它
        consts = [(a.monomial_coefficient((0, 0)), b.monomial_coefficient((0, 0)))
                  for a, b in result.basis]
        self.assertTrue(any(u == -v and u != 0 for u, v in consts))

    def test_constant_generator(self):
        one = P({(0, 0): 1})
        gens = [one, X]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        # a0 * 1 + a1 * x = 0 => a0 = -a1 * x，deg(a1) <= 0，维数 1
        self.assertEqual(result.dimension, 1)
        (a0, a1), = result.basis
        self.assertTrue(a1.is_zero() is False)
        self.assertEqual(a0 + a1 * X, Poly.zero(2))

    def test_constant_only(self):
        one = P({(0, 0): 2})
        result = bounded_syzygies([one], 0)
        check_result(self, result, [one])
        self.assertEqual(result.dimension, 0)

    def test_all_zero_generators(self):
        gens = [Poly.zero(2), Poly.zero(2)]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        # 两个系数多项式完全自由：2 * 3 = 6 维
        self.assertEqual(result.dimension, 6)


class TestPermutation(unittest.TestCase):
    def test_coordinates_follow_input_order(self):
        # 排列输入时关系逐坐标跟随，不丢不重
        result = bounded_syzygies([Y, X], 1)
        check_result(self, result, [Y, X])
        self.assertEqual(result.dimension, 1)
        (a0, a1), = result.basis
        # 坐标 0 对应 y、坐标 1 对应 x：关系为 a0 * y + a1 * x = 0
        self.assertIn((a0, a1), [(X, -Y), (-X, Y)])

    def test_duplicate_permuted(self):
        gens = [Y, X, Y]
        result = bounded_syzygies(gens, 1)
        check_result(self, result, gens)
        for a0, a1, a2 in result.basis:
            self.assertEqual(a0 * Y + a1 * X + a2 * Y, Poly.zero(2))


class TestIndependentVerification(unittest.TestCase):
    def test_relation_polynomial_primitive(self):
        # relation_polynomial 是独立核验原语：非零关系必须被它暴露
        gens = [X, Y]
        bad = (Y, X)  # y*x + x*y = 2xy != 0
        self.assertFalse(
            relation_polynomial(bad, gens).is_zero())
        good = (Y, -X)
        self.assertTrue(
            relation_polynomial(good, gens).is_zero())

    def test_result_verify_method(self):
        result = bounded_syzygies([X, Y], 1)
        self.assertTrue(all(result.verify(i)
                            for i in range(result.dimension)))

    def test_column_mapping_contents(self):
        result = bounded_syzygies([X, Y], 1)
        mapping = result.column_monomials
        self.assertEqual(len(mapping), result.num_unknowns)
        gen_indices = {i for i, _ in mapping}
        self.assertEqual(gen_indices, {0, 1})
        for _, alpha in mapping:
            self.assertLessEqual(sum(alpha), 1)


class TestValidation(unittest.TestCase):
    def test_bad_degree(self):
        for bad in (-1, True, 1.5, "1"):
            with self.assertRaises(InvalidGeneratorError):
                bounded_syzygies([X, Y], bad)

    def test_empty_generators_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([], 1)

    def test_mismatched_n_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([X, Poly({(1, 0, 0): 1}, n=3)], 1)

    def test_budget_exceeded(self):
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([X, Y], 1, max_unknowns=2)
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([X, Y], 1, max_equations=2)

    def test_single_generator_no_relation(self):
        result = bounded_syzygies([P({(1, 0): 1, (0, 0): -1})], 2)
        check_result(self, result, [P({(1, 0): 1, (0, 0): -1})])
        self.assertEqual(result.dimension, 0)


if __name__ == "__main__":
    unittest.main()
