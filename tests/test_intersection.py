"""理想交集 ideal_intersection 的测试。

验收要点对应：
- 单变量主理想、两个共享因子的二元理想：与手算结果比对（双向约化为零）；
- 两份组合证书（分别相对 I 与 J 的生成元）独立展开核验；
- 返回基中的每一项对两边的 Groebner 基分别约化为零（用独立参考实现复算）；
- 零理想、单位理想、重复生成元、预算 incomplete、变量数上限。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from groebner import (
    InvalidGeneratorError,
    Poly,
    groebner_basis,
    ideal_intersection,
)

import reference as ref


def P(term_dict, n):
    return Poly(term_dict, n=n)


def D(poly):
    return poly.terms


def expand_certificate(cert, gens):
    """用被测的 Poly 算术独立展开 sum_j cert[j]*g_j（只用加法/乘法）。"""
    total = Poly.zero(gens[0].n) if gens else Poly.zero(cert[0].n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def ref_reduces_to_zero(f_dict, gens_dicts):
    """独立核验：f 对 gens 理想的（独立算出的）约化基取余为 0。"""
    return ref.ideal_remainder(f_dict, gens_dicts) == {}


def assert_intersection_certificates(testcase, res):
    """每个交集基多项式的两份证书分别展开等于它本身。"""
    for b, ci, cj in zip(res.basis, res.certificates_i, res.certificates_j):
        testcase.assertEqual(len(ci), len(res.generators_i))
        testcase.assertEqual(len(cj), len(res.generators_j))
        testcase.assertEqual(expand_certificate(ci, res.generators_i), b,
                             "相对 I 的证书展开不等于基多项式")
        testcase.assertEqual(expand_certificate(cj, res.generators_j), b,
                             "相对 J 的证书展开不等于基多项式")


def assert_basis_membership_both_sides(testcase, res, i_dicts, j_dicts):
    """返回基中的每一项对两边的理想分别（独立参考）约化为零。"""
    for b in res.basis:
        testcase.assertTrue(ref_reduces_to_zero(D(b), i_dicts),
                            f"{b} 不在 I 中（独立参考复核）")
        testcase.assertTrue(ref_reduces_to_zero(D(b), j_dicts),
                            f"{b} 不在 J 中（独立参考复核）")


# ------------------------------------------------------------------ 手算例子

# 单变量主理想：I = (x^2 - x) = (x(x-1))，J = (x^2 - 3x + 2) = ((x-1)(x-2))
# 主理想交 = (lcm) = (x(x-1)(x-2)) = (x^3 - 3x^2 + 2x)
UNI_I = [{(2,): 1, (1,): -1}]
UNI_J = [{(2,): 1, (1,): -3, (0,): 2}]
UNI_LCM = {(3,): 1, (2,): -3, (1,): 2}

# 共享因子的二元理想：I = (x(x+y), y(x+y)) = (x+y)*(x,y)，J = ((x+y)(x-1))
# (x,y) 与 (x-1) 互素（无公共零点结构），故 (x,y)∩(x-1) = (x(x-1), y(x-1))，
# 于是 I∩J = (x+y)(x-1)*(x,y) = (x(x-1)(x+y), y(x-1)(x+y))。
BI_I = [{(2, 0): 1, (1, 1): 1}, {(1, 1): 1, (0, 2): 1}]
BI_J = [{(2, 0): 1, (1, 0): -1, (1, 1): 1, (0, 1): -1}]
BI_EXP1 = {(3, 0): 1, (2, 0): -1, (2, 1): 1, (1, 1): -1}   # x(x-1)(x+y)
BI_EXP2 = {(2, 1): 1, (1, 1): -1, (1, 2): 1, (0, 2): -1}   # y(x-1)(x+y)


class TestIntersectionKnown(unittest.TestCase):
    def test_univariate_principal(self):
        res = ideal_intersection([P(g, 1) for g in UNI_I],
                                 [P(g, 1) for g in UNI_J])
        self.assertEqual(res.status, "complete")
        # 手算交 = (x^3 - 3x^2 + 2x)：主理想 ⇒ 基恰为一个首一多项式
        self.assertEqual(len(res.basis), 1)
        self.assertEqual(D(res.basis[0]), UNI_LCM)
        assert_intersection_certificates(self, res)
        assert_basis_membership_both_sides(self, res, UNI_I, UNI_J)

    def test_bivariate_shared_factor(self):
        res = ideal_intersection([P(g, 2) for g in BI_I],
                                 [P(g, 2) for g in BI_J])
        self.assertEqual(res.status, "complete")
        self.assertTrue(res.basis)
        # 手算的两个生成交叉方向核验：期望生成元对返回基约化为零（独立参考），
        # 返回基每项对两边理想约化为零。
        basis_dicts = [D(b) for b in res.basis]
        self.assertEqual(ref.remainder(BI_EXP1, basis_dicts), {})
        self.assertEqual(ref.remainder(BI_EXP2, basis_dicts), {})
        assert_basis_membership_both_sides(self, res, BI_I, BI_J)
        assert_intersection_certificates(self, res)

    def test_certificates_in_original_ring(self):
        # 证书必须不含消元变量：变量数等于原变量数
        res = ideal_intersection([P(g, 2) for g in BI_I],
                                 [P(g, 2) for g in BI_J])
        for ci, cj in zip(res.certificates_i, res.certificates_j):
            for c in ci + cj:
                self.assertEqual(c.n, 2)


class TestIntersectionEdgeCases(unittest.TestCase):
    def test_zero_ideal_side(self):
        res = ideal_intersection([Poly.zero(2)], [P(g, 2) for g in BI_I])
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.basis, ())
        self.assertEqual(res.certificates_i, ())
        self.assertEqual(res.certificates_j, ())

    def test_empty_generator_list_is_zero_ideal(self):
        res = ideal_intersection([], [P(g, 2) for g in BI_I])
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.basis, ())

    def test_both_empty_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([], [])

    def test_unit_ideal(self):
        # (1) ∩ J = J：返回基生成 J，且每个元素属于 J
        j_gens = [{(2, 0): 1, (1, 1): 1}, {(1, 1): 1, (0, 2): 1}]
        res = ideal_intersection([Poly.constant(1, 2)], [P(g, 2) for g in j_gens])
        self.assertEqual(res.status, "complete")
        basis_dicts = [D(b) for b in res.basis]
        for g in j_gens:
            self.assertEqual(ref.remainder(g, basis_dicts), {})
        assert_basis_membership_both_sides(self, res, [{(0, 0): 1}], j_gens)
        assert_intersection_certificates(self, res)

    def test_duplicate_generators(self):
        g = {(2,): 1, (1,): -1}
        h = {(2,): 1, (1,): -3, (0,): 2}
        res1 = ideal_intersection([P(g, 1)], [P(h, 1)])
        res2 = ideal_intersection([P(g, 1), P(g, 1), Poly.zero(1)],
                                  [P(h, 1), P(h, 1)])
        self.assertEqual([D(b) for b in res2.basis], [D(b) for b in res1.basis])
        # 零生成元位置的证书恒为零多项式
        for ci in res2.certificates_i:
            self.assertTrue(ci[2].is_zero())
        assert_intersection_certificates(self, res2)

    def test_variable_count_limit(self):
        g4 = Poly({(1, 0, 0, 0): 1}, n=4)
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([g4], [g4])
        g3 = Poly({(1, 0, 0): 1}, n=3)
        res = ideal_intersection([g3], [g3])  # n=3 合法（加 t 后恰为 4）
        self.assertEqual(res.status, "complete")

    def test_inconsistent_variable_counts(self):
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([Poly({(1,): 1}, n=1)], [Poly({(1, 0): 1}, n=2)])

    def test_non_poly_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([Poly({(1,): 1}, n=1)], [42])

    def test_budget_incomplete(self):
        res = ideal_intersection([P(g, 2) for g in BI_I],
                                 [P(g, 2) for g in BI_J],
                                 max_pairs=0)
        self.assertEqual(res.status, "incomplete")
        self.assertFalse(res.reduced)
        self.assertFalse(res.is_complete)
        # incomplete 时不得自称完整交集；已发现元素（若有）证书仍合法
        assert_intersection_certificates(self, res)

    def test_budget_validation(self):
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([P(g, 1) for g in UNI_I],
                               [P(g, 1) for g in UNI_J], max_pairs=-1)
        with self.assertRaises(InvalidGeneratorError):
            ideal_intersection([P(g, 1) for g in UNI_I],
                               [P(g, 1) for g in UNI_J], max_steps=True)


class TestIntersectionOldInterfaceUnchanged(unittest.TestCase):
    """旧 grlex 接口不受块次序影响：同一输入结果与参考实现一致。"""

    def test_grlex_entry_unchanged(self):
        gens = [P(g, 2) for g in BI_I]
        res = groebner_basis(gens)
        self.assertEqual(res.status, "complete")
        self.assertEqual([D(b) for b in res.basis],
                         ref.reduced_groebner(BI_I))


if __name__ == "__main__":
    unittest.main()

