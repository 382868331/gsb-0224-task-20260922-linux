"""bounded_syzygies 的测试：独立展开核验 + 独立秩/零空间对照。"""

from __future__ import annotations

import unittest
from fractions import Fraction

from groebner import (
    BoundedSyzygyResult,
    InvalidGeneratorError,
    Poly,
    bounded_syzygies,
    groebner_basis,
    monomials_up_to_degree,
    relation_value,
)

import reference


def relation_to_vectors(result: BoundedSyzygyResult, relation):
    """把一条关系按 result.columns 列映射还原成 Fraction 向量。"""
    vec = []
    for i, exp in result.columns:
        vec.append(relation[i].monomial_coefficient(exp))
    return vec


class TestBoundedSyzygiesBasic(unittest.TestCase):
    def setUp(self):
        self.x = Poly.variable(2, 0)
        self.y = Poly.variable(2, 1)

    def test_xy_finds_y_minus_x(self):
        res = bounded_syzygies([self.x, self.y], 1)
        self.assertEqual(res.dimension, 1)
        (rel,) = res.basis
        self.assertEqual(rel, (self.y, -self.x))
        self.assertTrue(relation_value(rel, [self.x, self.y]).is_zero())

    def test_degree_too_small_no_relation(self):
        # d=0 时 a、b 只能是常数，a*x + b*y = 0 只有零解：不得捏造关系。
        res = bounded_syzygies([self.x, self.y], 0)
        self.assertEqual(res.dimension, 0)
        self.assertEqual(res.basis, ())
        self.assertEqual(res.rank, res.num_unknowns)

    def test_degree_bound_just_enough(self):
        # x**2, y**2 的最简关系 (y**2, -x**2) 次数为 2。
        x2 = self.x * self.x
        y2 = self.y * self.y
        self.assertEqual(bounded_syzygies([x2, y2], 1).dimension, 0)
        res = bounded_syzygies([x2, y2], 2)
        self.assertEqual(res.dimension, 1)
        self.assertEqual(res.basis[0], (y2, -x2))

    def test_zero_generator(self):
        zero = Poly.zero(2)
        res = bounded_syzygies([zero, self.x], 1)
        # 坐标 0 上任意次数<=1 的多项式都是关系：1, x, y 三个单位向量。
        self.assertEqual(res.dimension, 3)
        coords = {rel[0] for rel in res.basis}
        self.assertEqual(coords, {Poly.constant(1, 2), self.x, self.y})
        for rel in res.basis:
            self.assertTrue(rel[1].is_zero())
            self.assertTrue(relation_value(rel, [zero, self.x]).is_zero())

    def test_all_zero_generators(self):
        zero = Poly.zero(2)
        res = bounded_syzygies([zero, zero], 0)
        self.assertEqual(res.dimension, 2)
        one = Poly.constant(1, 2)
        self.assertEqual(set(res.basis), {(one, zero), (zero, one)})

    def test_duplicate_generators(self):
        f = self.x * self.x + self.y
        res = bounded_syzygies([f, f], 0)
        self.assertEqual(res.dimension, 1)
        one = Poly.constant(1, 2)
        self.assertEqual(res.basis[0], (one, -one))
        self.assertTrue(relation_value(res.basis[0], [f, f]).is_zero())

    def test_constant_generator(self):
        # 常数生成元 2 与 x：关系 (x, -2) 即 x*2 + (-2)*x = 0。
        two = Poly.constant(2, 2)
        res = bounded_syzygies([two, self.x], 1)
        self.assertEqual(res.dimension, 1)
        self.assertEqual(res.basis[0], (self.x, Poly.constant(-2, 2)))
        self.assertTrue(relation_value(res.basis[0], [two, self.x]).is_zero())
        # d=0：常数无法互相抵消（2*a + x*b 无常数解），无关系。
        self.assertEqual(bounded_syzygies([two, self.x], 0).dimension, 0)

    def test_permutation_follows_coordinates(self):
        f1 = self.x
        f2 = self.y
        f3 = self.x + self.y
        res123 = bounded_syzygies([f1, f2, f3], 1)
        res321 = bounded_syzygies([f3, f2, f1], 1)
        self.assertEqual(res123.dimension, res321.dimension)
        self.assertGreater(res123.dimension, 0)
        # 排列输入 => 坐标跟随映射：把 res321 的每条关系坐标逆排列回去，
        # 仍是 [f1, f2, f3] 的关系（独立展开为 0），关系不丢失。
        permuted = [(rel[2], rel[1], rel[0]) for rel in res321.basis]
        for rel in permuted:
            self.assertTrue(relation_value(rel, [f1, f2, f3]).is_zero())
        # 且两个解空间相同：把两组基向量按同一列映射堆起来，
        # 联合秩仍等于维数（用参考实现的独立消元复算）。
        vecs123 = [relation_to_vectors(res123, rel) for rel in res123.basis]
        vecs321 = [relation_to_vectors(res123, rel) for rel in permuted]
        _, pivots = reference._rref_independent(vecs123 + vecs321)
        self.assertEqual(len(pivots), res123.dimension)

    def test_result_metadata(self):
        res = bounded_syzygies([self.x, self.y], 1)
        self.assertEqual(res.degree_bound, 1)
        self.assertEqual(res.generators, (self.x, self.y))
        self.assertEqual(res.monomials, ((1, 0), (0, 1), (0, 0)))
        self.assertEqual(res.columns,
                         tuple((i, e) for i in range(2) for e in res.monomials))
        self.assertEqual(res.num_unknowns, 6)
        self.assertEqual(res.dimension, res.num_unknowns - res.rank)


class TestIndependentVerification(unittest.TestCase):
    """每条关系独立展开为 0；维数完备性由独立复算的矩阵秩证明。"""

    CASES = [
        ([Poly.variable(2, 0), Poly.variable(2, 1)], 2),
        ([Poly({(2, 0): 1, (0, 1): 1}, n=2),
          Poly({(1, 1): 1, (0, 0): -1}, n=2)], 3),
        ([Poly.zero(2), Poly.variable(2, 0), Poly.variable(2, 0)], 1),
        ([Poly.constant(3, 2), Poly.variable(2, 1)], 2),
        ([Poly.variable(3, 0), Poly.variable(3, 1), Poly.variable(3, 2)], 1),
    ]

    def test_relations_expand_to_zero_and_rank_matches(self):
        for gens, d in self.CASES:
            with self.subTest(d=d, gens=[g.to_string() for g in gens]):
                res = bounded_syzygies(gens, d)
                # 1) 逐多项式相乘求和：每条关系独立展开必须为 0。
                for rel in res.basis:
                    self.assertTrue(
                        relation_value(rel, gens).is_zero(),
                        f"关系未取零值: {rel}")
                    for a in rel:
                        if not a.is_zero():
                            self.assertLessEqual(sum(a.lm()), d)
                # 2) 秩证明：dimension == num_unknowns - rank，
                #    且 rank 与完全独立复算的秩一致。
                self.assertEqual(res.dimension, res.num_unknowns - res.rank)
                gen_dicts = [g.terms for g in gens]
                ref_rank, ref_num, ref_basis = (
                    reference.bounded_syzygy_expectation(
                        gen_dicts, gens[0].n, d))
                self.assertEqual(res.rank, ref_rank)
                self.assertEqual(res.num_unknowns, ref_num)
                self.assertEqual(res.dimension, ref_num - ref_rank)
                # 3) 规范基与独立零空间基逐向量一致。
                got = [relation_to_vectors(res, rel) for rel in res.basis]
                self.assertEqual(got, ref_basis)

    def test_reference_relation_also_expands_to_zero(self):
        # 用参考实现的 dict 算术独立展开被测库给出的关系（双重独立）。
        gens = [Poly({(2, 0): 1, (0, 2): -1}, n=2),
                Poly({(1, 0): 1, (0, 1): -1}, n=2)]
        res = bounded_syzygies(gens, 3)
        self.assertGreater(res.dimension, 0)
        gen_dicts = [g.terms for g in gens]
        for rel in res.basis:
            total = reference.expand([a.terms for a in rel], gen_dicts)
            self.assertEqual(total, {})


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.x = Poly.variable(2, 0)

    def test_bad_degree_bound(self):
        for bad in (-1, True, 1.5, "2"):
            with self.assertRaises(InvalidGeneratorError):
                bounded_syzygies([self.x], bad)

    def test_empty_generators_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([], 1)

    def test_unknowns_budget(self):
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([self.x, Poly.variable(2, 1)], 5,
                             max_unknowns=4)
        with self.assertRaises(InvalidGeneratorError):
            bounded_syzygies([self.x], 1, max_unknowns=0)

    def test_monomials_up_to_degree(self):
        self.assertEqual(monomials_up_to_degree(2, 1),
                         ((1, 0), (0, 1), (0, 0)))
        self.assertEqual(len(monomials_up_to_degree(3, 2)), 10)

    def test_old_groebner_still_works(self):
        res = groebner_basis([self.x, Poly.variable(2, 1)])
        self.assertEqual(res.status, "complete")
        self.assertEqual(len(res.basis), 2)


if __name__ == "__main__":
    unittest.main()
