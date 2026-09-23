"""除法、S 多项式、Buchberger、约化基与组合证书测试。

期望值来自 tests/reference.py 的独立朴素实现（不调用被测核心）；
对被测核心做的只有“独立展开证书 == 基多项式”“最终 S 余式全为 0”等核验。
"""

import unittest
from fractions import Fraction
from itertools import combinations

from groebner import (
    Poly,
    PolynomialError,
    GroebnerBudgets,
    divide,
    remainder,
    s_polynomial,
    groebner_basis,
    reduced_basis,
    ideal_membership,
    membership_with_basis,
    MAX_INPUT_TERMS,
)
from groebner.polynomial import lcm_monomial

from tests import reference as R


def to_ref(p: Poly) -> R.TermMap:
    return {e: c for e, c in p.items()}


def from_ref(d: R.TermMap, n: int) -> Poly:
    return Poly(dict(d), nvars=n)


def independent_expand(combo: dict[int, Poly], gens: list[Poly]) -> R.TermMap:
    """用独立参考实现展开 Σ combo[i]*g_i。"""
    rc = {i: to_ref(q) for i, q in combo.items()}
    return R.ref_expand(rc, [to_ref(g) for g in gens])


def canon(term_maps):
    """dict 列表的规范排序（dict 本身不可比较），按 (项, 系数) 元组序列。"""
    return sorted(
        term_maps,
        key=lambda d: tuple(sorted((e, str(c)) for e, c in d.items())),
    )


class TestDivision(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)

    def test_division_identity(self):
        f = self.x ** 3 * self.y ** 2 + 2 * self.x * self.y + self.y + 1
        d1 = self.x * self.y + 1
        d2 = self.y + 1
        res = divide(f, [d1, d2])
        # 被测接口自证恒等式（往返检查，仅作补充）
        self.assertEqual(res.identity(), f)
        # 与独立参考实现的商、余式逐一对照
        qs, r = R.ref_divrem(to_ref(f), [to_ref(d1), to_ref(d2)])
        self.assertEqual(to_ref(res.remainder), r)
        for q, rq in zip(res.quotients, qs):
            self.assertEqual(to_ref(q), rq)

    def test_remainder_no_term_divisible(self):
        f = self.x ** 2 * self.y + self.x + self.y
        d = self.x * self.y
        r = remainder(f, [d])
        for e in r.items():
            self.assertFalse(all(a <= b for a, b in zip((1, 1), e[0])))
        qs, rr = R.ref_divrem(to_ref(f), [to_ref(d)])
        self.assertEqual(to_ref(r), rr)

    def test_zero_dividend(self):
        res = divide(Poly.zero(2), [self.x + 1, self.y])
        self.assertTrue(res.remainder.is_zero())
        self.assertTrue(all(q.is_zero() for q in res.quotients))

    def test_zero_divisor_rejected(self):
        with self.assertRaises(PolynomialError):
            divide(self.x, [Poly.zero(2)])

    def test_order_changes_quotients_not_gb_remainder(self):
        # 非 GB 除数：换顺序，教科书余式可能不同（这是正常数学事实）
        f = self.x * self.y ** 2 + 1
        d1 = self.x * self.y + 1
        d2 = self.y ** 2 - 1
        r_a = remainder(f, [d1, d2])
        r_b = remainder(f, [d2, d1])
        # GB 下余式与顺序无关：用经典基再试
        g1 = self.x ** 2 - self.y
        g2 = self.x * self.y - 1
        gb = reduced_basis(groebner_basis([g1, g2])).basis
        f2 = self.x ** 3 * self.y + 2 * self.x + self.y ** 2
        self.assertEqual(to_ref(remainder(f2, gb)), to_ref(remainder(f2, list(reversed(gb)))))
        self.assertIsInstance(r_a, Poly)
        self.assertIsInstance(r_b, Poly)


class TestSPolynomial(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)

    def test_cox_example(self):
        # Cox–Little–O'Shea: f=x^3-2xy, g=x^2y-2y^2+x => S = -x^2
        f = self.x ** 3 - 2 * self.x * self.y
        g = self.x ** 2 * self.y - 2 * self.y ** 2 + self.x
        self.assertEqual(to_ref(s_polynomial(f, g)), {(2, 0): Fraction(-1)})
        # 独立参考实现交叉验证
        self.assertEqual(to_ref(s_polynomial(f, g)), R.ref_s_poly(to_ref(f), to_ref(g)))

    def test_symmetry_sign(self):
        f = self.x ** 2 + self.y
        g = self.x * self.y - 1
        self.assertEqual(s_polynomial(f, g), -s_polynomial(g, f))

    def test_zero_rejected(self):
        with self.assertRaises(PolynomialError):
            s_polynomial(Poly.zero(2), self.x)


class TestBuchbergerAndReduction(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)

    def _assert_gb_criterion(self, basis: list[Poly]) -> None:
        """验收要求：全部最终 S 多项式除以该基余式为 0。"""
        self.assertTrue(basis)
        for a, b in combinations(basis, 2):
            r = remainder(s_polynomial(a, b), basis)
            self.assertTrue(r.is_zero(), f"S({a},{b}) 余式非零: {r}")

    def _assert_reduced_monic(self, basis: list[Poly]) -> None:
        for k, p in enumerate(basis):
            self.assertEqual(p.leading_coefficient(), Fraction(1))
            others = [basis[j].leading_monomial() for j in range(len(basis)) if j != k]
            for exp in (e for e, _ in p.items()):
                for lm in others:
                    self.assertFalse(
                        all(a <= b for a, b in zip(lm, exp)),
                        f"约化基成员 {p} 的项 {exp} 可被 {lm} 整除",
                    )

    def test_cox_basis_matches_reference(self):
        f = self.x ** 3 - 2 * self.x * self.y
        g = self.x ** 2 * self.y - 2 * self.y ** 2 + self.x
        gens = [f, g]
        rb = reduced_basis(groebner_basis(gens))
        ref = R.ref_reduced_basis([to_ref(f), to_ref(g)])
        self.assertEqual(canon([to_ref(p) for p in rb.basis]), canon(ref))
        self._assert_gb_criterion(rb.basis)
        self._assert_reduced_monic(rb.basis)

    def test_elimination_relations(self):
        # <x+y, xy-1> 的消元基含 y^2+1 与 x+y
        g1 = self.x + self.y
        g2 = self.x * self.y - 1
        rb = reduced_basis(groebner_basis([g1, g2]))
        as_set = {frozenset(p.items()) for p in rb.basis}
        self.assertIn(frozenset([((0, 2), Fraction(1)), ((0, 0), Fraction(1))]), as_set)
        self.assertIn(frozenset([((1, 0), Fraction(1)), ((0, 1), Fraction(1))]), as_set)
        self._assert_gb_criterion(rb.basis)
        # 与独立参考一致
        self.assertEqual(
            canon([to_ref(p) for p in rb.basis]),
            canon(R.ref_reduced_basis([to_ref(g1), to_ref(g2)])),
        )

    def test_unit_ideal(self):
        rb = reduced_basis(groebner_basis([self.x, self.x - 1]))
        self.assertEqual(len(rb.basis), 1)
        self.assertEqual(to_ref(rb.basis[0]), {(0, 0): Fraction(1)})

    def test_three_variables_reference(self):
        x = Poly.var(0, 3)
        y = Poly.var(1, 3)
        z = Poly.var(2, 3)
        gens = [x ** 2 - y, y ** 2 - z, x * z - 1]
        rb = reduced_basis(groebner_basis(gens))
        self._assert_gb_criterion(rb.basis)
        self.assertEqual(
            canon([to_ref(p) for p in rb.basis]),
            canon(R.ref_reduced_basis([to_ref(g) for g in gens])),
        )

    def test_zero_generators(self):
        res = groebner_basis([Poly.zero(2), Poly.zero(2)])
        self.assertTrue(res.complete)
        self.assertEqual(res.basis, [])
        rb = reduced_basis(res)
        self.assertTrue(rb.reduced and rb.monic)
        self.assertEqual(rb.basis, [])

    def test_zero_among_nonzero(self):
        g = self.x ** 2 - self.y
        res = groebner_basis([Poly.zero(2), g, Poly.zero(2)])
        rb = reduced_basis(res)
        # 证书下标保留：第 1 个输入（下标 1）是非零生成元
        for combo in rb.combinations:
            self.assertIn(1, combo)
            self.assertTrue(all(k in (1,) for k in combo))
        self._assert_gb_criterion(rb.basis)

    def test_duplicate_generators(self):
        g1 = self.x ** 2 - self.y
        g2 = self.x * self.y - 1
        rb_a = reduced_basis(groebner_basis([g1, g1, g2, g2]))
        rb_b = reduced_basis(groebner_basis([g1, g2]))
        self.assertEqual(
            canon([to_ref(p) for p in rb_a.basis]),
            canon([to_ref(p) for p in rb_b.basis]),
        )
        self._assert_gb_criterion(rb_a.basis)

    def test_input_order_independence(self):
        g1 = self.x ** 2 - self.y
        g2 = self.x * self.y - 1
        g3 = self.y ** 2 + self.x
        orders = [
            [g1, g2, g3],
            [g3, g1, g2],
            [g2, g3, g1],
            [g1, g3, g2, g1],
        ]
        ref_basis = None
        for gs in orders:
            rb = reduced_basis(groebner_basis(gs))
            cur = canon([to_ref(p) for p in rb.basis])
            if ref_basis is None:
                ref_basis = cur
            else:
                self.assertEqual(cur, ref_basis or cur)
            self._assert_gb_criterion(rb.basis)

    def test_coefficient_scaling_invariance(self):
        g1 = self.x ** 2 - self.y
        g2 = self.x * self.y - 1
        rb_a = reduced_basis(groebner_basis([g1, g2]))
        rb_b = reduced_basis(
            groebner_basis([Fraction(2, 3) * g1, -5 * g2])
        )
        self.assertEqual(
            canon([to_ref(p) for p in rb_a.basis]),
            canon([to_ref(p) for p in rb_b.basis]),
        )

    def test_input_term_limit(self):
        terms = [((i, 0), 1) for i in range(MAX_INPUT_TERMS + 1)]
        # 指数 (100, 0) 合法但项数超限
        big = Poly.from_terms(terms, nvars=2)
        with self.assertRaises(PolynomialError):
            groebner_basis([big])

    def test_input_term_limit_boundary_exact(self):
        # 恰好 100 项必须接受；101 项必须拒绝（边界两侧）
        ok = Poly.from_terms(
            [((i, 0), 1) for i in range(MAX_INPUT_TERMS)], nvars=2
        )
        res = groebner_basis([ok])
        self.assertTrue(res.complete)
        too_many = Poly.from_terms(
            [((i, 0), 1) for i in range(MAX_INPUT_TERMS + 1)], nvars=2
        )
        with self.assertRaises(PolynomialError):
            groebner_basis([too_many])

    def test_four_variable_certificates_expand(self):
        x = Poly.var(0, 4)
        y = Poly.var(1, 4)
        z = Poly.var(2, 4)
        w = Poly.var(3, 4)
        gens = [x * y - z, z * w - 1, x - y]
        rb = reduced_basis(groebner_basis(gens))
        for p, combo in zip(rb.basis, rb.combinations):
            self.assertEqual(independent_expand(combo, gens), to_ref(p))

    def test_max_variables_boundary(self):
        x = Poly.var(0, 4)
        w = Poly.var(3, 4)
        rb = reduced_basis(groebner_basis([x * w - 1, x + w]))
        self._assert_gb_criterion(rb.basis)


class TestCertificates(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)
        self.g1 = self.x ** 2 - self.y
        self.g2 = self.x * self.y - 1
        self.g3 = self.y ** 2 + self.x

    def test_basis_certificates_expand_independently(self):
        """独立证书展开检查：每个基多项式 == Σ combo[i]*g_i（用 reference 展开）。"""
        rb = reduced_basis(groebner_basis([self.g1, self.g2, self.g3]))
        gens = [self.g1, self.g2, self.g3]
        for k, (p, combo) in enumerate(zip(rb.basis, rb.combinations)):
            expanded = independent_expand(combo, gens)
            self.assertEqual(expanded, to_ref(p), f"基成员 {k} 证书展开不符")

    def test_basis_spans_same_ideal_membership_roundtrip(self):
        # 证书行展开是基成员；基成员也都在原理想中（独立参考成员性）
        rb = reduced_basis(groebner_basis([self.g1, self.g2]))
        ref_gb = R.ref_reduced_basis([to_ref(self.g1), to_ref(self.g2)])
        for p in rb.basis:
            r = R.ref_rem(to_ref(p), ref_gb)
            self.assertEqual(r, {})

    def test_member_certificate(self):
        # f = x*g1 + y*g2 是当然成员；证书须真实展开成 f
        f = self.x * self.g1 + self.y * self.g2
        m = ideal_membership(f, [self.g1, self.g2])
        self.assertEqual(m.status, "member")
        self.assertTrue(m.member is True)
        expanded = independent_expand(m.coefficients, [self.g1, self.g2])
        self.assertEqual(expanded, to_ref(f))

    def test_member_certificate_non_trivial(self):
        # 取一个非显然成员：基成员的多项式倍，用证书验证
        rb = reduced_basis(groebner_basis([self.g1, self.g2]))
        p = (self.x + self.y) * rb.basis[0]
        m = ideal_membership(p, [self.g1, self.g2])
        self.assertEqual(m.status, "member")
        expanded = independent_expand(m.coefficients, [self.g1, self.g2])
        self.assertEqual(expanded, to_ref(p))
        self.assertTrue(m.remainder.is_zero())

    def test_nonmember(self):
        f = self.x + self.y + 1
        m = ideal_membership(f, [self.g1, self.g2])
        self.assertEqual(m.status, "nonmember")
        self.assertTrue(m.member is False)
        self.assertFalse(m.remainder.is_zero())
        self.assertIsNone(m.certificate)
        # 独立参考确认非成员：除以独立 GB 余式非零
        ref_gb = R.ref_reduced_basis([to_ref(self.g1), to_ref(self.g2)])
        self.assertNotEqual(R.ref_rem(to_ref(f), ref_gb), {})

    def test_nonmember_zero_ideal(self):
        m = ideal_membership(self.x, [Poly.zero(2)])
        self.assertEqual(m.status, "nonmember")
        m0 = ideal_membership(Poly.zero(2), [Poly.zero(2)])
        self.assertEqual(m0.status, "member")


class TestBudgets(unittest.TestCase):
    def setUp(self):
        self.x = Poly.var(0, 2)
        self.y = Poly.var(1, 2)
        self.g1 = self.x ** 2 - self.y
        self.g2 = self.x * self.y - 1

    def test_incomplete_is_marked_and_not_final(self):
        res = groebner_basis([self.g1, self.g2], GroebnerBudgets(max_pairs=1))
        # 小预算确实可能在处理完前耗尽
        if not res.complete:
            self.assertFalse(res.reduced)
            self.assertIsNotNone(res.incomplete_reason)
            with self.assertRaises(PolynomialError):
                reduced_basis(res)  # 不允许把 incomplete 基伪装成最终基

    def test_incomplete_membership_unknown_vs_nonmember(self):
        # 预算耗尽 + 余式非零 => unknown，与“证明非成员”严格分开
        m = ideal_membership(
            self.x + self.y + 1,
            [self.g1, self.g2],
            GroebnerBudgets(max_pairs=1),
        )
        if not m.complete:
            self.assertEqual(m.status, "unknown")
            self.assertIsNone(m.member)
            self.assertIsNone(m.certificate)
        # 即便基不完整，余式为 0 仍是可靠成员，证书必须成立
        mm = ideal_membership(
            self.g1, [self.g1, self.g2], GroebnerBudgets(max_pairs=1)
        )
        self.assertEqual(mm.status, "member")
        expanded = independent_expand(mm.coefficients, [self.g1, self.g2])
        self.assertEqual(expanded, to_ref(self.g1))

    def test_max_terms_budget(self):
        # 初始输入 4 项恰好放得下，但加入 S 余式时超 max_terms
        res = groebner_basis([self.g1, self.g2], GroebnerBudgets(max_pairs=10, max_terms=4))
        self.assertFalse(res.complete)
        self.assertIn("max_terms", res.incomplete_reason or "")

    def test_max_terms_budget_immediate(self):
        # 输入本身就超过 max_terms：立即 incomplete，不产生任何基成员
        res = groebner_basis([self.g1, self.g2], GroebnerBudgets(max_pairs=10, max_terms=3))
        self.assertFalse(res.complete)
        self.assertEqual(res.basis, [])
        self.assertIn("max_terms", res.incomplete_reason or "")
        self.assertEqual(res.pairs_processed, 0)

    def test_bad_budgets_rejected(self):
        with self.assertRaises(PolynomialError):
            GroebnerBudgets(max_pairs=0)
        with self.assertRaises(PolynomialError):
            GroebnerBudgets(max_terms=True)  # type: ignore[arg-type]
        with self.assertRaises(PolynomialError):
            groebner_basis([self.g1], budgets=(1, 2))  # type: ignore[arg-type]

    def test_large_enough_budget_completes(self):
        res = groebner_basis(
            [self.g1, self.g2], GroebnerBudgets(max_pairs=1000, max_terms=1000)
        )
        self.assertTrue(res.complete)
        rb = reduced_basis(res)
        for a, b in combinations(rb.basis, 2):
            self.assertTrue(remainder(s_polynomial(a, b), rb.basis).is_zero())


class TestStrengthenedProperties(unittest.TestCase):
    """额外性质核验：约化性质、恒等式独立展开、4 变量、不完整基恒等式。"""

    def test_three_var_reduced_monic_and_identity(self):
        x = Poly.var(0, 3)
        y = Poly.var(1, 3)
        z = Poly.var(2, 3)
        gens = [x ** 2 - y, y ** 2 - z, x * z - 1]
        rb = reduced_basis(groebner_basis(gens))
        for k, p in enumerate(rb.basis):
            self.assertEqual(p.leading_coefficient(), Fraction(1))
            others = [rb.basis[j].leading_monomial() for j in range(len(rb.basis)) if j != k]
            for exp in (e for e, _ in p.items()):
                for lm in others:
                    self.assertFalse(all(a <= b for a, b in zip(lm, exp)))

    def test_nonmember_division_identity_expansion(self):
        # 非成员情形也必须满足 f - remainder = Σ coeff[i]*g_i（独立展开）
        x = Poly.var(0, 2)
        y = Poly.var(1, 2)
        gens = [x ** 2 - y, x * y - 1]
        f = x ** 3 + 2 * y + 1
        m = ideal_membership(f, gens)
        self.assertEqual(m.status, "nonmember")
        expanded = independent_expand(m.coefficients, gens)
        expected = R.ref_sub(to_ref(f), to_ref(m.remainder))
        self.assertEqual(expanded, expected)

    def test_four_variables_gb(self):
        x = Poly.var(0, 4)
        y = Poly.var(1, 4)
        z = Poly.var(2, 4)
        w = Poly.var(3, 4)
        gens = [x * y - z, z * w - 1, x - y]
        rb = reduced_basis(groebner_basis(gens))
        for a, b in combinations(rb.basis, 2):
            self.assertTrue(remainder(s_polynomial(a, b), rb.basis).is_zero())
        self.assertEqual(
            canon([to_ref(p) for p in rb.basis]),
            canon(R.ref_reduced_basis([to_ref(g) for g in gens])),
        )

    def test_incomplete_basis_certificates_still_expand(self):
        # 即使预算耗尽，已产出的基成员证书恒等式必须继续成立（原子性/无部分破坏）
        x = Poly.var(0, 2)
        y = Poly.var(1, 2)
        gens = [x ** 2 - y, x * y - 1]
        res = groebner_basis(gens, GroebnerBudgets(max_pairs=1))
        if res.complete:
            self.skipTest("该例在 max_pairs=1 下已完成，无法观察 incomplete")
        for p, combo in zip(res.basis, res.combinations):
            self.assertEqual(independent_expand(combo, gens), to_ref(p))
        # membership 的 unknown 结果仍满足 f - r = Σ c g
        m = membership_with_basis(x + y + 1, res)
        self.assertEqual(m.status, "unknown")
        expanded = independent_expand(m.coefficients, gens)
        self.assertEqual(expanded, R.ref_sub(to_ref(m.dividend), to_ref(m.remainder)))

    def test_incomplete_max_terms_certificates_expand(self):
        x = Poly.var(0, 2)
        y = Poly.var(1, 2)
        gens = [x ** 2 - y, x * y - 1]
        res = groebner_basis(gens, GroebnerBudgets(max_pairs=10, max_terms=4))
        self.assertFalse(res.complete)
        for p, combo in zip(res.basis, res.combinations):
            self.assertEqual(independent_expand(combo, gens), to_ref(p))

    def test_rational_generators(self):
        x = Poly.var(0, 1)
        g1 = Poly({(2,): Fraction(1, 2), (0,): Fraction(-3, 4)}, nvars=1)
        # 一元：约化基必为首一一次式
        rb = reduced_basis(groebner_basis([g1]))
        self.assertEqual(len(rb.basis), 1)
        p = rb.basis[0]
        self.assertEqual(p.leading_monomial(), (2,))
        self.assertEqual(p.leading_coefficient(), Fraction(1))
        # 独立核验证书
        self.assertEqual(independent_expand(rb.combinations[0], [g1]), to_ref(p))


class TestErrorSemantics(unittest.TestCase):
    def test_empty_and_bad_generators(self):
        with self.assertRaises(PolynomialError):
            groebner_basis([])
        with self.assertRaises(PolynomialError):
            groebner_basis("not-a-list")  # type: ignore[arg-type]
        x2 = Poly.var(0, 2)
        x3 = Poly.var(0, 3)
        with self.assertRaises(PolynomialError):
            groebner_basis([x2, x3])
        with self.assertRaises(PolynomialError):
            groebner_basis([42])  # type: ignore[list-item]

    def test_membership_nvars_mismatch(self):
        x2 = Poly.var(0, 2)
        x3 = Poly.var(0, 3)
        with self.assertRaises(PolynomialError):
            ideal_membership(x3, [x2 + 1])


if __name__ == "__main__":
    unittest.main()
