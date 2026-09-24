"""lex 次序与消元入口 eliminate 的测试。

覆盖：
- lex 次序键与 Poly.leading；
- lex 除法 / S 多项式 / Groebner 基与 tests/reference.py 的独立实现逐项比对；
- 消元的手算例子（代入关系、隐式化、无解系统、空消元理想）；
- 非前缀消去变量、同时消去多个变量、证书独立展开；
- 随机小系统（至多 3 变量、次数 ≤ 1、确定性种子）与独立参考比对；
- 边界：消去全部变量、一变量、顺序/重复无关、零生成元占位；
- 失败：非法 order / eliminate 参数；预算耗尽 → incomplete 且不发布半成品。

期望值要么来自手算，要么来自 tests/reference.py 的独立朴素实现，
不调用被测核心生成期望。
"""

from __future__ import annotations

import random
import unittest
from fractions import Fraction

from groebner import (
    InvalidGeneratorError,
    Poly,
    check_s_polynomials,
    divide,
    eliminate,
    groebner_basis,
    groebner_basis_of_zero,
    membership,
    s_polynomial,
)
from groebner.poly import grlex_key, lex_key

import reference as ref


def P(term_dict, n):
    return Poly(term_dict, n=n)


def D(poly):
    return poly.terms


def expand_certificate(cert, gens):
    """用被测 Poly 算术独立展开 sum_j cert[j]*g_j（与证书产生路径无关）。"""
    total = Poly.zero(gens[0].n) if gens else Poly.zero(cert[0].n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def assert_certificate_expands(testcase, basis, certs, gens):
    for b, cert in zip(basis, certs):
        testcase.assertEqual(len(cert), len(gens))
        testcase.assertEqual(expand_certificate(cert, list(gens)), b)


# 手算系统 --------------------------------------------------------------

# x^2 + y^2 - 1, x - y^2
C1 = {(2, 0): 1, (0, 2): 1, (0, 0): -1}
C2 = {(1, 0): 1, (0, 2): -1}

# x^2 - y, y^2 - z, z - 1
CHAIN = [
    {(2, 0, 0): 1, (0, 1, 0): -1},
    {(0, 2, 0): 1, (0, 0, 1): -1},
    {(0, 0, 1): 1, (0, 0, 0): -1},
]


class TestLexOrderPrimitives(unittest.TestCase):
    def test_lex_key_ordering(self):
        # lex(x>y)：x^2 > xy > x > y^2 > y > 1（与 grlex 明显不同）
        exps = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
        ordered = sorted(exps, key=lex_key)
        self.assertEqual(ordered,
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)])

    def test_poly_leading_respects_order(self):
        # p = x + y^2：grlex 首项是 y^2（次数 2），lex 首项是 x
        p = P({(1, 0): 1, (0, 2): 1}, 2)
        self.assertEqual(p.leading(lex_key), ((1, 0), Fraction(1)))
        self.assertEqual(p.leading(grlex_key), ((0, 2), Fraction(1)))
        # lm()/lc() 固定 grlex（旧行为不变）
        self.assertEqual(p.lm(), (0, 2))
        self.assertEqual(p.lc(), Fraction(1))
        self.assertEqual(p.lt(), ((0, 2), Fraction(1)))

    def test_iteration_is_still_grlex(self):
        p = P({(1, 0): 1, (0, 2): 1}, 2)
        self.assertEqual([e for e, _ in p], [(0, 2), (1, 0)])


class TestLexDivideAndS(unittest.TestCase):
    def test_lex_division_matches_reference(self):
        rng = random.Random(20260924)
        for _ in range(40):
            n = rng.randint(1, 3)
            f_d = {}
            for _ in range(rng.randint(1, 4)):
                e = tuple(rng.randint(0, 1) for _ in range(n))
                f_d[e] = Fraction(rng.randint(-2, 2), rng.randint(1, 2))
            f_d = {e: c for e, c in f_d.items() if c} or {
                tuple([0] * n): Fraction(1)}
            ds_d = []
            for _ in range(rng.randint(1, 3)):
                d = {}
                for _ in range(rng.randint(1, 3)):
                    e = tuple(rng.randint(0, 1) for _ in range(n))
                    d[e] = Fraction(rng.randint(-2, 2), rng.randint(1, 2))
                d = {e: c for e, c in d.items() if c}
                if d:
                    ds_d.append(d)
            f, ds = P(f_d, n), [P(d, n) for d in ds_d]
            res = divide(f, ds, order="lex")
            # 恒等式（被测算术，仅作补充）
            rebuilt = res.remainder
            for q, d in zip(res.quotients, ds):
                rebuilt = rebuilt + q * d
            self.assertEqual(rebuilt, f)
            # 余式与独立参考（lex）逐单项式一致
            _, rr = ref.divide_full(f_d, ds_d, ref.key_lex)
            self.assertEqual(D(res.remainder), rr)

    def test_lex_s_polynomial_matches_reference(self):
        rng = random.Random(9901)
        made = 0
        while made < 30:
            n = rng.randint(2, 3)
            a, b = {}, {}
            for _ in range(rng.randint(2, 3)):
                a[tuple(rng.randint(0, 1) for _ in range(n))] = \
                    Fraction(rng.randint(-2, 2), rng.randint(1, 2))
                b[tuple(rng.randint(0, 1) for _ in range(n))] = \
                    Fraction(rng.randint(-2, 2), rng.randint(1, 2))
            a = {e: c for e, c in a.items() if c}
            b = {e: c for e, c in b.items() if c}
            if not a or not b:
                continue
            made += 1
            got = D(s_polynomial(P(a, n), P(b, n), order="lex"))
            self.assertEqual(got, ref.s_poly(a, b, ref.key_lex))

    def test_lex_s_polynomial_can_differ_from_grlex(self):
        # f = x + y^2, g = x + 1：两种次序首项不同，S 多项式也不同
        f = P({(1, 0): 1, (0, 2): 1}, 2)
        g = P({(1, 0): 1, (0, 0): 1}, 2)
        self.assertNotEqual(s_polynomial(f, g, order="lex"),
                            s_polynomial(f, g, order="grlex"))

    def test_bad_order_rejected(self):
        f = P({(1, 0): 1}, 2)
        g = P({(0, 1): 1}, 2)
        for bad in ("grlexx", "", "LEX", 1, None, ["lex"]):
            with self.assertRaises(InvalidGeneratorError):
                groebner_basis([f, g], order=bad)
            with self.assertRaises(InvalidGeneratorError):
                divide(f, [g], order=bad)
            with self.assertRaises(InvalidGeneratorError):
                s_polynomial(f, g, order=bad)
            with self.assertRaises(InvalidGeneratorError):
                check_s_polynomials([f, g], order=bad)
            with self.assertRaises(InvalidGeneratorError):
                membership(f, [g], order=bad)

    def test_lex_limited_to_three_variables(self):
        # lex / 消元本轮限定至多 3 变量；4 变量只允许 grlex
        from groebner import groebner_basis_of_zero
        gens4 = [P({(1, 0, 0, 0): 1, (0, 1, 0, 0): -1}, 4),
                 P({(0, 0, 0, 1): 1, (0, 0, 0, 0): -1}, 4)]
        with self.assertRaises(InvalidGeneratorError):
            groebner_basis(gens4, order="lex")
        with self.assertRaises(InvalidGeneratorError):
            divide(gens4[0], [gens4[1]], order="lex")
        with self.assertRaises(InvalidGeneratorError):
            s_polynomial(gens4[0], gens4[1], order="lex")
        with self.assertRaises(InvalidGeneratorError):
            check_s_polynomials(gens4, order="lex")
        with self.assertRaises(InvalidGeneratorError):
            groebner_basis_of_zero(4, order="lex")
        with self.assertRaises(InvalidGeneratorError):
            eliminate(gens4, [0])
        # 同一 4 变量系统 grlex 合法（旧能力不受影响）
        self.assertEqual(groebner_basis(gens4).status, "complete")
        self.assertEqual(groebner_basis_of_zero(4).status, "complete")


class TestLexGroebnerBasis(unittest.TestCase):
    def test_circle_parabola_lex_hand_computed(self):
        gens = [P(C1, 2), P(C2, 2)]
        res = groebner_basis(gens, order="lex")
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.order, "lex")
        self.assertTrue(res.reduced)
        # 手算（lex x>y）唯一首一约化基：{x - y^2, y^4 + y^2 - 1}
        bases = [D(b) for b in res.basis]
        self.assertEqual(bases, [
            {(1, 0): 1, (0, 2): -1},
            {(0, 4): 1, (0, 2): 1, (0, 0): -1},
        ])
        self.assertEqual(check_s_polynomials(list(res.basis), order="lex"), [])
        # 独立参考（lex）逐对复算 S 多项式余式为 0
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                self.assertEqual(ref.remainder(
                    ref.s_poly(bases[i], bases[j], ref.key_lex),
                    bases, ref.key_lex), {})
        assert_certificate_expands(self, res.basis, res.certificates, gens)

    def test_lex_basis_matches_reference_known_examples(self):
        examples = [
            (2, [C1, C2]),
            (3, CHAIN),
            (2, [{(1, 1): 1, (0, 0): -1}, {(1, 0): 1, (0, 1): -1}]),
        ]
        for n, gens_d in examples:
            with self.subTest(gens=gens_d):
                gens = [P(d, n) for d in gens_d]
                res = groebner_basis(gens, order="lex")
                self.assertEqual(res.status, "complete")
                self.assertEqual([D(b) for b in res.basis],
                                 ref.reduced_groebner(gens_d, ref.key_lex))
                self.assertEqual(
                    check_s_polynomials(list(res.basis), order="lex"), [])

    def test_lex_basis_independent_of_order_and_duplicates(self):
        gens = [P(C1, 2), P(C2, 2)]
        base = groebner_basis(gens, order="lex").basis
        r1 = groebner_basis([gens[1], gens[0]], order="lex")
        self.assertEqual(r1.basis, base)
        r2 = groebner_basis([
            gens[0], gens[1], gens[0],
            P({(2, 0): 2, (0, 2): 2, (0, 0): -2}, 2),
        ], order="lex")
        self.assertEqual(r2.status, "complete")
        self.assertEqual(r2.basis, base)

    def test_lex_differs_from_grlex_on_same_input(self):
        # 同一系统两种次序的约化基确实不同（旧 grlex 行为未被改变）
        gl = groebner_basis([P(C1, 2), P(C2, 2)], order="grlex")
        lx = groebner_basis([P(C1, 2), P(C2, 2)], order="lex")
        self.assertNotEqual(gl.basis, lx.basis)
        self.assertEqual(gl.order, "grlex")
        self.assertEqual(lx.order, "lex")

    def test_one_variable_lex_equals_grlex(self):
        gens = [P({(3,): 1, (1,): -2}, 1), P({(2,): 1, (0,): -1}, 1)]
        r_lex = groebner_basis(gens, order="lex")
        r_grlex = groebner_basis(gens)
        self.assertEqual(r_lex.basis, r_grlex.basis)

    def test_zero_generators_positions_kept_in_lex(self):
        gens = [P({}, 2), P(C1, 2), P({}, 2), P(C2, 2)]
        res = groebner_basis(gens, order="lex")
        self.assertEqual(res.status, "complete")
        self.assertEqual(
            [D(b) for b in res.basis],
            ref.reduced_groebner([{}, C1, {}, C2], ref.key_lex))
        assert_certificate_expands(self, res.basis, res.certificates, gens)
        self.assertEqual(len(res.certificates[0]), 4)

    def test_zero_ideal_lex(self):
        res = groebner_basis_of_zero(3, order="lex")
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.basis, ())
        self.assertEqual(res.order, "lex")

    def test_random_lex_systems_vs_reference(self):
        rng = random.Random(60601)
        for trial in range(20):
            n = rng.randint(2, 3)
            k = rng.randint(1, 3)
            gens_d = []
            for _ in range(k):
                exps = set()
                for _ in range(rng.randint(1, 3)):
                    exps.add(tuple(rng.randint(0, 1) for _ in range(n)))
                g = {e: Fraction(rng.randint(-2, 2), rng.randint(1, 2))
                     for e in exps}
                g = {e: c for e, c in g.items() if c != 0}
                gens_d.append(g or {tuple([0] * n): Fraction(1)})
            with self.subTest(trial=trial, n=n):
                gens = [P(d, n) for d in gens_d]
                res = groebner_basis(gens, order="lex")
                self.assertEqual(res.status, "complete")
                self.assertEqual(
                    [D(b) for b in res.basis],
                    ref.reduced_groebner(gens_d, ref.key_lex))
                assert_certificate_expands(
                    self, res.basis, res.certificates, gens)
                self.assertEqual(
                    check_s_polynomials(list(res.basis), order="lex"), [])


class TestLexMembership(unittest.TestCase):
    def test_member_and_nonmember_under_lex(self):
        gens = [P(C1, 2), P(C2, 2)]
        # y^4 + y^2 - 1 是成员（消元关系）
        elim = P({(0, 4): 1, (0, 2): 1, (0, 0): -1}, 2)
        m = membership(elim, gens, order="lex")
        self.assertEqual(m.status, "member")
        self.assertTrue(m.remainder.is_zero())
        self.assertEqual(expand_certificate(m.certificate, list(gens)), elim)
        self.assertEqual(m.groebner.order, "lex")
        # x + 1 非成员：独立参考 lex 余式也非零
        f = P({(1, 0): 1, (0, 0): 1}, 2)
        mn = membership(f, gens, order="lex")
        self.assertEqual(mn.status, "nonmember")
        self.assertNotEqual(ref.ideal_remainder(D(f), [C1, C2], ref.key_lex),
                            {})

    def test_inconclusive_under_zero_budget_lex(self):
        gens = [P(C1, 2), P(C2, 2)]
        m = membership(P({(3, 3): 1, (0, 0): 1}, 2), gens,
                       order="lex", max_pairs=0)
        self.assertEqual(m.groebner.status, "incomplete")
        self.assertIn(m.status, ("inconclusive", "member"))


# ---------------------------------------------------------------- 消元手算例


class TestEliminationHandExamples(unittest.TestCase):
    def test_circle_parabola_eliminate_x(self):
        gens = [P(C1, 2), P(C2, 2)]
        e = eliminate(gens, [0])
        self.assertEqual(e.status, "complete")
        self.assertEqual(e.eliminate, (0,))
        self.assertEqual([D(b) for b in e.elimination_basis],
                         [{(0, 4): 1, (0, 2): 1, (0, 0): -1}])
        for b in e.elimination_basis:
            for exp in b.terms:
                self.assertEqual(exp[0], 0)
        assert_certificate_expands(
            self, e.elimination_basis, e.certificates, gens)
        # 消元基本身构成保留变量环上的 lex Groebner 基（逐对 S 余式为 0；
        # 被消去前缀位置恒为 0，n 变量 lex 在这些单项式上的限制即其 lex 序）
        self.assertEqual(
            check_s_polynomials(list(e.elimination_basis), order="lex"), [])
        self.assertEqual(e.groebner.order, "lex")

    def test_linear_substitution(self):
        # x = 2y, y = 3
        g1 = P({(1, 0): 1, (0, 1): -2}, 2)
        g2 = P({(0, 1): 1, (0, 0): -3}, 2)
        e_y = eliminate([g1, g2], [1])       # 消去 y -> x - 6
        self.assertEqual([D(b) for b in e_y.elimination_basis],
                         [{(1, 0): 1, (0, 0): -6}])
        e_x = eliminate([g1, g2], [0])       # 消去 x -> y - 3
        self.assertEqual([D(b) for b in e_x.elimination_basis],
                         [{(0, 1): 1, (0, 0): -3}])
        assert_certificate_expands(
            self, e_y.elimination_basis, e_y.certificates, [g1, g2])
        assert_certificate_expands(
            self, e_x.elimination_basis, e_x.certificates, [g1, g2])

    def test_implicitization_nonprefix_variable(self):
        # 参数曲线 x = t, y = t^2：g1 = x - t, g2 = y - t^2；
        # 消去 t（下标 2，**非前缀**变量）-> y - x^2
        g1 = P({(1, 0, 0): 1, (0, 0, 1): -1}, 3)
        g2 = P({(0, 1, 0): 1, (0, 0, 2): -1}, 3)
        e = eliminate([g1, g2], [2])
        self.assertEqual(e.eliminate, (2,))
        # 消元基是 Q[x,y] 上 lex 的首一约化基：x^2 >_lex y，故为 x**2 - y
        self.assertEqual([D(b) for b in e.elimination_basis],
                         [{(2, 0, 0): 1, (0, 1, 0): -1}])
        assert_certificate_expands(
            self, e.elimination_basis, e.certificates, [g1, g2])
        # 消去 x,y（保留 t）：参数可取任意 t，消元理想为零 -> 空基
        e_t = eliminate([g1, g2], [0, 1])
        self.assertEqual(e_t.status, "complete")
        self.assertEqual(e_t.elimination_basis, ())

    def test_xy_minus_1_xz_minus_1(self):
        # xy - 1, xz - 1，消去 x -> y - z
        g1 = P({(1, 1, 0): 1, (0, 0, 0): -1}, 3)
        g2 = P({(1, 0, 1): 1, (0, 0, 0): -1}, 3)
        e = eliminate([g1, g2], [0])
        self.assertEqual([D(b) for b in e.elimination_basis],
                         [{(0, 1, 0): 1, (0, 0, 1): -1}])
        assert_certificate_expands(
            self, e.elimination_basis, e.certificates, [g1, g2])

    def test_three_variable_chain_elimination(self):
        gens = [P(d, 3) for d in CHAIN]
        # 消去 x：y^2 - 1, z - 1
        e = eliminate(gens, [0])
        self.assertEqual([D(b) for b in e.elimination_basis],
                         ref.elimination_basis(CHAIN, [0], 3))
        self.assertEqual([D(b) for b in e.elimination_basis], [
            {(0, 2, 0): 1, (0, 0, 0): -1},
            {(0, 0, 1): 1, (0, 0, 0): -1},
        ])
        assert_certificate_expands(
            self, e.elimination_basis, e.certificates, gens)
        # 消去 z（非前缀）：x^2 - y, y^2 - 1
        ez = eliminate(gens, [2])
        self.assertEqual([D(b) for b in ez.elimination_basis],
                         ref.elimination_basis(CHAIN, [2], 3))
        self.assertEqual([D(b) for b in ez.elimination_basis], [
            {(2, 0, 0): 1, (0, 1, 0): -1},
            {(0, 2, 0): 1, (0, 0, 0): -1},
        ])
        assert_certificate_expands(
            self, ez.elimination_basis, ez.certificates, gens)

    def test_simultaneously_eliminate_two_nonprefix(self):
        # x = t, y = t^2，消去 x 与 t，保留 y：理想 ∩ Q[y] = 0
        g1 = P({(1, 0, 0): 1, (0, 0, 1): -1}, 3)
        g2 = P({(0, 1, 0): 1, (0, 0, 2): -1}, 3)
        e = eliminate([g1, g2], (0, 2))
        self.assertEqual(e.eliminate, (0, 2))
        self.assertEqual(e.elimination_basis, ())

    def test_empty_elimination_ideal(self):
        # 主理想 (x) 消去 x -> 空（I ∩ Q[y] = 0）
        x = P({(1, 0): 1}, 2)
        self.assertEqual(eliminate([x], [0]).elimination_basis, ())
        # 消去 y -> (x)
        e = eliminate([x], [1])
        self.assertEqual([D(b) for b in e.elimination_basis], [{(1, 0): 1}])

    def test_eliminate_all_variables(self):
        # 真理想 (x, y) ∩ Q = 0
        gens = [P({(1, 0): 1}, 2), P({(0, 1): 1}, 2)]
        self.assertEqual(eliminate(gens, [0, 1]).elimination_basis, ())
        # 不相容系统 (x, x-1)：1 属于理想 -> 消元基含常数 1
        bad = [P({(1,): 1}, 1), P({(1,): 1, (0,): -1}, 1)]
        e = eliminate(bad, [0])
        self.assertEqual([D(b) for b in e.elimination_basis], [{(0,): 1}])
        assert_certificate_expands(
            self, e.elimination_basis, e.certificates, bad)

    def test_zero_generators_kept_positions(self):
        gens = [P({}, 2), P(C1, 2), P({}, 2), P(C2, 2)]
        e = eliminate(gens, [0])
        self.assertEqual(e.status, "complete")
        self.assertEqual([D(b) for b in e.elimination_basis],
                         [{(0, 4): 1, (0, 2): 1, (0, 0): -1}])
        for cert in e.certificates:
            self.assertEqual(len(cert), 4)
            self.assertTrue(cert[0].is_zero())
            self.assertTrue(cert[2].is_zero())
            self.assertEqual(expand_certificate(cert, list(gens)),
                             e.elimination_basis[0])

    def test_random_eliminations_vs_reference(self):
        rng = random.Random(252525)
        for trial in range(18):
            n = rng.randint(2, 3)
            k = rng.randint(1, 3)
            gens_d = []
            for _ in range(k):
                exps = set()
                for _ in range(rng.randint(1, 3)):
                    exps.add(tuple(rng.randint(0, 1) for _ in range(n)))
                g = {e: Fraction(rng.randint(-2, 2), rng.randint(1, 2))
                     for e in exps}
                gens_d.append({e: c for e, c in g.items() if c}
                              or {tuple([0] * n): Fraction(1)})
            elim_count = rng.randint(1, n)
            elim_idx = tuple(sorted(
                rng.sample(range(n), elim_count)))
            with self.subTest(trial=trial, n=n, elim=elim_idx):
                gens = [P(d, n) for d in gens_d]
                e = eliminate(gens, list(elim_idx))
                self.assertEqual(e.status, "complete")
                expected = ref.elimination_basis(gens_d, elim_idx, n)
                self.assertEqual(
                    [D(b) for b in e.elimination_basis], expected)
                # 输出项确实不含被消去变量
                elim_set = set(elim_idx)
                for b in e.elimination_basis:
                    for exp in b.terms:
                        for p in elim_set:
                            self.assertEqual(exp[p], 0)
                # 证书可独立展开
                assert_certificate_expands(
                    self, e.elimination_basis, e.certificates, gens)
                # 消元基在保留变量环上自身是 lex Groebner 基
                self.assertEqual(
                    check_s_polynomials(list(e.elimination_basis),
                                        order="lex"), [])
                # 完整 lex 基的 S 多项式判据只在消去变量是前缀 {0..k-1}
                # 时可直接在原坐标核验；非前缀时内部次序是置换后的 lex。
                if tuple(sorted(elim_idx)) == tuple(range(elim_count)):
                    self.assertEqual(
                        check_s_polynomials(list(e.groebner.basis),
                                            order="lex"), [])


class TestEliminationValidation(unittest.TestCase):
    def setUp(self):
        self.gens = [P(C1, 2), P(C2, 2)]

    def test_bare_int_rejected(self):
        with self.assertRaises(InvalidGeneratorError) as cm:
            eliminate(self.gens, 0)
        self.assertIn("[0]", str(cm.exception))

    def test_empty_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, [])

    def test_out_of_range_rejected(self):
        with self.assertRaises(InvalidGeneratorError) as cm:
            eliminate(self.gens, [2])
        self.assertIn("越界", str(cm.exception))
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, [-1])

    def test_duplicate_rejected(self):
        with self.assertRaises(InvalidGeneratorError) as cm:
            eliminate(
                [P({(1, 0, 0): 1}, 3), P({(0, 0, 1): 1}, 3)], [0, 0])
        self.assertIn("重复", str(cm.exception))

    def test_bool_and_bad_types_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, [True])
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, [0.0])
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, "x")
        with self.assertRaises(InvalidGeneratorError):
            eliminate(self.gens, None)

    def test_bad_generators_still_rejected(self):
        with self.assertRaises(InvalidGeneratorError):
            eliminate([P({(1,): 1}, 1), P({(1, 0): 1}, 2)], [0])
        with self.assertRaises(InvalidGeneratorError):
            eliminate([42], [0])
        # 对照：合法调用正常完成
        self.assertEqual(eliminate(self.gens, [0]).status, "complete")

    def test_bad_budget_rejected(self):
        for bad in (-1, 1.5, True):
            with self.assertRaises(InvalidGeneratorError):
                eliminate(self.gens, [0], max_pairs=bad)
            with self.assertRaises(InvalidGeneratorError):
                eliminate(self.gens, [0], max_steps=bad)


class TestEliminationBudgetFailure(unittest.TestCase):
    def test_zero_pair_budget_is_incomplete_with_no_partial_basis(self):
        gens = [P(C1, 2), P(C2, 2)]
        e = eliminate(gens, [0], max_pairs=0)
        self.assertEqual(e.status, "incomplete")
        self.assertFalse(e.groebner.reduced)
        self.assertTrue(e.reason)
        # 不发布未核验的半成品消元基
        self.assertEqual(e.elimination_basis, ())
        self.assertEqual(e.certificates, ())
        # 但中间 lex 基的证书仍合法（不留语义不一致的状态）
        assert_certificate_expands(
            self, e.groebner.basis, e.groebner.certificates, gens)
        self.assertEqual(e.groebner.order, "lex")

    def test_zero_step_budget_incomplete(self):
        gens = [P(C1, 2), P(C2, 2)]
        e = eliminate(gens, [0], max_steps=0)
        self.assertEqual(e.status, "incomplete")
        self.assertEqual(e.elimination_basis, ())

    def test_incomplete_then_full_run_agrees_with_reference(self):
        gens = [P(C1, 2), P(C2, 2)]
        small = eliminate(gens, [0], max_pairs=0)
        full = eliminate(gens, [0])
        self.assertEqual(small.status, "incomplete")
        self.assertEqual(full.status, "complete")
        self.assertEqual([D(b) for b in full.elimination_basis],
                         ref.elimination_basis([C1, C2], [0], 2))


if __name__ == "__main__":
    unittest.main()
