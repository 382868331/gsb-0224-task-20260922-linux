"""Groebner 基、约化首一基、证书、成员判定、预算语义的测试。

验收要点对应：
- 与 tests/reference.py 的独立朴素实现逐单项式比对；
- 独立证书展开检查（用参考的展开或被测算术，不依赖被测算法结论）；
- 全部最终 S 多项式余式为 0 的核验；
- 零生成元、重复生成元、不同输入顺序、非成员、预算 incomplete。
"""

from __future__ import annotations

import random
import unittest
from fractions import Fraction

from groebner import (
    InvalidGeneratorError,
    Poly,
    check_s_polynomials,
    groebner_basis,
    groebner_basis_of_zero,
    membership,
    membership_in_zero_ideal,
)

import reference as ref


def P(term_dict, n):
    return Poly(term_dict, n=n)


def D(poly):
    return poly.terms


def expand_certificate(cert, gens):
    """用**被测的 Poly 算术**独立展开 sum_j cert[j]*g_j。

    展开只用到多项式乘法/加法，与“证书是怎么算出来的”无关，
    因此是对证书内容的独立核验，而不是重复算法结论。
    """
    total = Poly.zero(gens[0].n) if gens else Poly.zero(cert[0].n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def assert_certificate(testcase, result):
    """逐基多项式核验 b_i = sum_j cert_i[j] * g_j，且零生成元位置证书为 0。"""
    gens = result.generators
    for b, cert in zip(result.basis, result.certificates):
        testcase.assertEqual(len(cert), len(gens))
        testcase.assertEqual(expand_certificate(cert, gens), b)
    for j, g in enumerate(gens):
        if g.is_zero():
            for cert in result.certificates:
                testcase.assertTrue(cert[j].is_zero(),
                                    f"零生成元位置 {j} 的证书必须为零多项式")


# ----------------------------------------------------------------- 标准小例子

# 经典 Cox-Little-O'Shea 例：g1 = x^2 y - 1, g2 = x y^2 - x
G1 = {(2, 1): 1, (0, 0): -1}
G2 = {(1, 2): 1, (1, 0): -1}

# 圆与抛物线：x^2 + y^2 - 1, x - y^2
C1 = {(2, 0): 1, (0, 2): 1, (0, 0): -1}
C2 = {(1, 0): 1, (0, 2): -1}


class TestGroebnerKnownExamples(unittest.TestCase):
    def test_classic_pair_matches_reference(self):
        gens = [P(G1, 2), P(G2, 2)]
        res = groebner_basis(gens)
        self.assertEqual(res.status, "complete")
        self.assertTrue(res.reduced)
        expected = ref.reduced_groebner([G1, G2])
        self.assertEqual([D(b) for b in res.basis], expected)
        # 每个基多项式都首一
        for b in res.basis:
            self.assertEqual(b.lc(), Fraction(1))
        # 约化性：每个基多项式的**每一个**单项式都不被其他元素的首项整除
        for i, b in enumerate(res.basis):
            for e in b.terms:
                for j, d in enumerate(res.basis):
                    if i == j:
                        continue
                    self.assertFalse(
                        all(x >= y for x, y in zip(e, d.lm())),
                        f"{e} 被第 {j} 个基元素首项整除，基未约化")
        assert_certificate(self, res)
        self.assertEqual(check_s_polynomials(list(res.basis)), [])

    def test_circle_parabola_elimination(self):
        gens = [P(C1, 2), P(C2, 2)]
        res = groebner_basis(gens)
        self.assertEqual(res.status, "complete")
        expected = ref.reduced_groebner([C1, C2])
        self.assertEqual([D(b) for b in res.basis], expected)
        # grlex 首一约化基：{x**2 + x - 1, y**2 - x}
        bases = [D(b) for b in res.basis]
        self.assertIn({(2, 0): 1, (1, 0): 1, (0, 0): -1}, bases)
        self.assertIn({(0, 2): 1, (1, 0): -1}, bases)
        # 消元关系：代入 x = y**2 得只含 y 的关系 y**4 + y**2 - 1 = 0，
        # 它是理想成员，且证书给出它在原生成元下的具体组合
        elim = P({(0, 4): 1, (0, 2): 1, (0, 0): -1}, 2)
        m = membership(elim, gens)
        self.assertEqual(m.status, "member")
        self.assertEqual(expand_certificate(m.certificate, list(gens)), elim)
        assert_certificate(self, res)
        self.assertTrue(all(b.lc() == 1 for b in res.basis))

    def test_reduced_basis_is_unique_under_order_and_duplicates(self):
        base = [P(G1, 2), P(G2, 2)]
        ref_basis = groebner_basis(base).basis

        # 不同输入顺序
        r1 = groebner_basis([base[1], base[0]])
        self.assertEqual(r1.basis, ref_basis)
        # 重复生成元、以及非首一的倍数重复
        r2 = groebner_basis([
            base[0], base[1], base[0],
            P({(2, 1): 2, (0, 0): -2}, 2),   # 2*g1
            P({(1, 2): -3, (1, 0): 3}, 2),  # -3*g2
        ])
        self.assertEqual(r2.status, "complete")
        self.assertEqual(r2.basis, ref_basis)
        assert_certificate(self, r1)
        assert_certificate(self, r2)

    def test_zero_generators_ignored_but_positions_kept(self):
        gens = [P({}, 2), P(G1, 2), P({}, 2), P(G2, 2)]
        res = groebner_basis(gens)
        self.assertEqual(res.status, "complete")
        self.assertEqual(
            [D(b) for b in res.basis],
            ref.reduced_groebner([{}, G1, {}, G2]))
        assert_certificate(self, res)
        # 证书长度等于原始输入个数（4）
        self.assertEqual(len(res.certificates[0]), 4)

    def test_all_zero_generators_is_zero_ideal(self):
        res = groebner_basis([P({}, 2), P({}, 2)])
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.basis, ())
        self.assertEqual(res.certificates, ())
        res0 = groebner_basis_of_zero(3)
        self.assertEqual(res0.status, "complete")
        self.assertEqual(res0.basis, ())

    def test_s_polynomial_criterion_for_final_basis(self):
        # 对多个例子：最终基的每对 S 多项式余式必须为 0
        examples = [
            [G1, G2],
            [C1, C2],
            [{(1, 0): 1, (0, 1): -1}, {(0, 2): 1, (0, 0): -1}],
            [{(3, 0): 1, (0, 0): -1}, {(0, 2): 1, (1, 0): 1}],
        ]
        for ex in examples:
            with self.subTest(ex=ex):
                res = groebner_basis([P(d, 2) for d in ex])
                self.assertEqual(res.status, "complete")
                self.assertEqual(check_s_polynomials(list(res.basis)), [])
                # 同一判据再用 tests/reference.py 的独立 S 多项式与约化复算一遍
                bd = [D(b) for b in res.basis]
                for i in range(len(bd)):
                    for j in range(i + 1, len(bd)):
                        self.assertEqual(
                            ref.remainder(ref.s_poly(bd[i], bd[j]), bd), {})

    def test_three_variable_chain(self):
        # x^2 - y, y^2 - z, z - 1：三变量 grlex 例
        gens_d = [
            {(2, 0, 0): 1, (0, 1, 0): -1},
            {(0, 2, 0): 1, (0, 0, 1): -1},
            {(0, 0, 1): 1, (0, 0, 0): -1},
        ]
        res = groebner_basis([P(g, 3) for g in gens_d])
        self.assertEqual(res.status, "complete")
        self.assertEqual([D(b) for b in res.basis],
                         ref.reduced_groebner(gens_d))
        assert_certificate(self, res)
        bd = [D(b) for b in res.basis]
        for i in range(len(bd)):
            for j in range(i + 1, len(bd)):
                self.assertEqual(
                    ref.remainder(ref.s_poly(bd[i], bd[j]), bd), {})
        # x^4 - 1 是消元得到的成员
        f = P({(4, 0, 0): 1, (0, 0, 0): -1}, 3)
        m = membership(f, [P(g, 3) for g in gens_d])
        self.assertEqual(m.status, "member")
        self.assertEqual(expand_certificate(m.certificate,
                                             [P(g, 3) for g in gens_d]), f)

    def test_four_variables_smoke(self):
        # 4 变量：线性三角系统，约化基就是其首一线性形式
        gens_d = [
            {(1, 0, 0, 0): 1, (0, 1, 0, 0): -1},
            {(0, 1, 0, 0): 1, (0, 0, 1, 0): -1},
            {(0, 0, 1, 0): 1, (0, 0, 0, 1): -1},
            {(0, 0, 0, 1): 1, (0, 0, 0, 0): -1},
        ]
        res = groebner_basis([P(g, 4) for g in gens_d])
        self.assertEqual(res.status, "complete")
        self.assertEqual([D(b) for b in res.basis],
                         ref.reduced_groebner(gens_d))
        assert_certificate(self, res)


class TestRandomizedVsReference(unittest.TestCase):
    def test_random_systems(self):
        rng = random.Random(7777)
        for trial in range(25):
            n = rng.randint(1, 3)
            k = rng.randint(1, 3)
            gens = []
            for _ in range(k):
                exps = set()
                for _ in range(rng.randint(1, 4)):
                    exps.add(tuple(rng.randint(0, 2) for _ in range(n)))
                g = {e: Fraction(rng.randint(-2, 2), rng.randint(1, 2))
                     for e in exps}
                g = {e: c for e, c in g.items() if c != 0}
                if not g:
                    g = {tuple([0] * n): Fraction(1)}
                gens.append(g)
            with self.subTest(trial=trial, n=n):
                res = groebner_basis([P(g, n) for g in gens])
                self.assertEqual(res.status, "complete")
                self.assertEqual(
                    [D(b) for b in res.basis],
                    ref.reduced_groebner(gens))
                assert_certificate(self, res)
                self.assertEqual(check_s_polynomials(list(res.basis)), [])


class TestMembership(unittest.TestCase):
    def test_member_has_zero_remainder_and_valid_certificate(self):
        gens = [P(C1, 2), P(C2, 2)]
        # f = x*g1 + y*g2 + g1，必为理想成员
        f = (P({(1, 0): 1}, 2) * gens[0]
             + P({(0, 1): 1}, 2) * gens[1]
             + gens[0])
        m = membership(f, gens)
        self.assertEqual(m.status, "member")
        self.assertTrue(m.remainder.is_zero())
        self.assertEqual(expand_certificate(m.certificate, list(gens)), f)
        # 证书长度含全部生成元
        self.assertEqual(len(m.certificate), 2)

    def test_nonmember(self):
        gens = [P(C1, 2), P(C2, 2)]
        f = P({(1, 0): 1, (0, 0): 1}, 2)  # x + 1
        m = membership(f, gens)
        self.assertEqual(m.status, "nonmember")
        self.assertFalse(m.remainder.is_zero())
        self.assertIsNone(m.certificate)
        # 独立参考同样给出非零余式
        self.assertNotEqual(ref.ideal_remainder(D(f), [C1, C2]), {})

    def test_zero_polynomial_membership(self):
        gens = [P(C1, 2), P(C2, 2)]
        m = membership(P({}, 2), gens)
        self.assertEqual(m.status, "member")
        self.assertTrue(all(c.is_zero() for c in m.certificate))

    def test_zero_ideal_membership(self):
        yes = membership(P({}, 2), [P({}, 2)])
        self.assertEqual(yes.status, "member")
        no = membership(P({(1, 0): 1}, 2), [P({}, 2)])
        self.assertEqual(no.status, "nonmember")
        m0 = membership_in_zero_ideal(P({}, 3))
        self.assertEqual(m0.status, "member")
        m1 = membership_in_zero_ideal(P({(0, 0, 1): 1}, 3))
        self.assertEqual(m1.status, "nonmember")

    def test_membership_random_certificates(self):
        rng = random.Random(31337)
        for trial in range(15):
            n = rng.randint(1, 3)
            k = rng.randint(1, 3)
            gens_d = []
            for _ in range(k):
                exps = {tuple(rng.randint(0, 2) for _ in range(n))
                        for _ in range(2)}
                g = {e: Fraction(rng.randint(-2, 2), 1) for e in exps}
                g = {e: c for e, c in g.items() if c}
                gens_d.append(g or {tuple([0] * n): Fraction(1)})
            gens = [P(g, n) for g in gens_d]
            res = groebner_basis(gens)
            # 用基元素的线性组合构造“必为成员”的 f
            f = Poly.zero(n)
            for b in res.basis:
                exps = {tuple(rng.randint(0, 1) for _ in range(n))
                        for _ in range(2)}
                q = P({e: Fraction(rng.randint(-2, 2), 1) for e in exps}, n)
                f = f + q * b
            m = membership(f, gens)
            self.assertEqual(m.status, "member")
            self.assertEqual(expand_certificate(m.certificate, list(gens)), f)
            # 独立参考余式也为 0
            self.assertEqual(ref.ideal_remainder(D(f), gens_d), {})


class TestBudget(unittest.TestCase):
    def test_max_pairs_incomplete_is_not_claimed_final(self):
        gens = [P(G1, 2), P(G2, 2)]
        res = groebner_basis(gens, max_pairs=0)
        # max_pairs=0：第一对就触顶（初始入库不计入对）
        self.assertEqual(res.status, "incomplete")
        self.assertFalse(res.reduced)
        self.assertTrue(res.reason)
        # incomplete 时证书仍必须是合法的
        assert_certificate(self, res)

    def test_incomplete_distinct_from_answer(self):
        # 极小预算：结果不能自称 complete；调大预算后变为 complete
        small = groebner_basis([P(G1, 2), P(G2, 2)], max_pairs=1)
        full = groebner_basis([P(G1, 2), P(G2, 2)])
        self.assertEqual(small.status, "incomplete")
        self.assertEqual(full.status, "complete")
        self.assertIn("预算", small.reason)

    def test_membership_inconclusive_when_budget_exhausted(self):
        # 预算耗尽且余式非零：只能 inconclusive，不能误判 nonmember
        m = membership(P({(3, 3): 1, (0, 0): 1}, 2),
                       [P(G1, 2), P(G2, 2)], max_pairs=1)
        self.assertEqual(m.groebner.status, "incomplete")
        # 本例在极小预算下余式非零 → inconclusive；若恰为零则 member，
        # 二者都不允许错误地宣称 nonmember
        self.assertIn(m.status, ("inconclusive", "member"))
        if m.status == "member":
            self.assertTrue(m.remainder.is_zero())

    def test_bad_budget_rejected(self):
        gens = [P(G1, 2), P(G2, 2)]
        for bad in (-1, 1.5, True, False):
            with self.assertRaises(InvalidGeneratorError):
                groebner_basis(gens, max_pairs=bad)
            with self.assertRaises(InvalidGeneratorError):
                groebner_basis(gens, max_terms=bad)
            with self.assertRaises(InvalidGeneratorError):
                groebner_basis(gens, max_steps=bad)

    def test_zero_budget_is_valid_but_incomplete(self):
        gens = [P(G1, 2), P(G2, 2)]
        self.assertEqual(groebner_basis(gens, max_pairs=0).status, "incomplete")
        self.assertEqual(groebner_basis(gens, max_terms=0).status, "incomplete")
        self.assertEqual(groebner_basis(gens, max_steps=0).status, "incomplete")

    def test_max_terms_incomplete_carries_reason(self):
        # 极小项数预算：任何非平凡约化都会触顶
        res = groebner_basis([P(G1, 2), P(G2, 2)], max_terms=1)
        self.assertEqual(res.status, "incomplete")
        self.assertIn("项数预算", res.reason)
        # 已发布的基元素证书仍然合法（失败不留半成品语义）
        assert_certificate(self, res)

    def test_incomplete_basis_still_spans_same_ideal(self):
        # incomplete 基的每个元素仍是原生成元的组合；
        # 且原生成元也能由 incomplete 基线性表示（同理想）。
        gens = [P(G1, 2), P(G2, 2)]
        res = groebner_basis(gens, max_pairs=1)
        if res.status == "complete":
            self.skipTest("该系统一对即完成，无法构造 incomplete")
        for g in gens:
            from groebner import divide as gdivide
            r = gdivide(g, list(res.basis)).remainder
            self.assertTrue(r.is_zero(),
                            "incomplete 基必须仍能约化掉所有原生成元")

    def test_terms_boundary_accepted(self):
        # 恰好 100 项合法
        terms = {(i % 10, i // 10): 1 for i in range(100)}
        res = groebner_basis([P(terms, 2)])
        self.assertEqual(res.status, "complete")

    def test_max_steps_budget_returns_incomplete(self):
        # 步数预算同样能熔断；证书仍合法
        res = groebner_basis([P(G1, 2), P(G2, 2)], max_steps=0)
        self.assertEqual(res.status, "incomplete")
        self.assertIn("步数", res.reason)
        assert_certificate(self, res)

    def test_hard_system_bounded_runtime_inconclusive(self):
        # 稠密对抗输入（确定性生成）：默认预算必须在有限时间内给出 incomplete
        # （非挂死），且 incomplete 基仍张成同一理想、证书合法。
        import time
        rng = random.Random(123)

        def dense(n, terms, deg):
            exps = {}
            while len(exps) < terms:
                exps[tuple(rng.randint(0, deg) for _ in range(n))] = \
                    Fraction(rng.randint(-3, 3))
            return P({e: c for e, c in exps.items() if c}, n=n)

        gens = [dense(3, 6, 3) for _ in range(4)]
        t0 = time.time()
        res = groebner_basis(gens)
        self.assertLess(time.time() - t0, 10.0)
        # 该系统在题面朴素能力下不可在默认预算内完成：
        if res.status == "incomplete":
            assert_certificate(self, res)
            from groebner import divide as gdivide
            for g in gens:
                self.assertTrue(gdivide(g, list(res.basis)).remainder.is_zero())
            # membership 同样必须有界且不得误判 nonmember
            f = P({(3, 3, 3): 1, (0, 0, 0): 1}, 3)
            t0 = time.time()
            mm = membership(f, gens)
            self.assertLess(time.time() - t0, 10.0)
            self.assertIn(mm.status, ("inconclusive", "member"))
        else:
            # 若未来优化使其可完成，则必须是真正的 Groebner 基。
            self.assertEqual(check_s_polynomials(list(res.basis)), [])


class TestGeneratorValidation(unittest.TestCase):
    def test_non_poly_rejected_with_location(self):
        with self.assertRaises(InvalidGeneratorError) as cm:
            groebner_basis([P(G1, 2), 42])
        self.assertIn("1", str(cm.exception))

    def test_mixed_variable_count_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            groebner_basis([P({(1,): 1}, 1), P({(1, 0): 1}, 2)])

    def test_empty_generators_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            groebner_basis([])

    def test_total_terms_limit(self):
        # 总输入项数 > 100 拒绝；错误信息给出位置
        terms = {(i % 4, i // 4): 1 for i in range(60)}
        g1 = P(terms, 2)
        g2 = P({(i, 0): 1 for i in range(45)}, 2)
        with self.assertRaises(InvalidGeneratorError) as cm:
            groebner_basis([g1, g2])
        self.assertIn("100", str(cm.exception))

    def test_single_poly_not_accepted_as_sequence(self):
        with self.assertRaises(InvalidGeneratorError):
            groebner_basis(P(G1, 2))


if __name__ == "__main__":
    unittest.main()
