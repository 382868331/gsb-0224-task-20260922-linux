"""确定性随机属性测试：与独立参考实现全面对照。

使用固定种子（题面要求示例数据由仓库内确定性代码生成），覆盖 1~3 变量、
随机生成的小理想；期望值全部来自 tests/reference.py。
"""

import itertools
import random
import unittest
from fractions import Fraction
from itertools import combinations

from groebner import (
    Poly,
    groebner_basis,
    reduced_basis,
    ideal_membership,
    divide,
    remainder,
    s_polynomial,
)
from tests import reference as R


def to_ref(p: Poly) -> R.TermMap:
    return {e: c for e, c in p.items()}


def canon(term_maps):
    return sorted(
        term_maps,
        key=lambda d: tuple(sorted((e, str(c)) for e, c in d.items())),
    )


def _monomial_pool(nvars: int, max_deg: int) -> list[tuple[int, ...]]:
    """所有总次数 <= max_deg 的指数元组。"""
    return [
        exps
        for exps in itertools.product(range(max_deg + 1), repeat=nvars)
        if sum(exps) <= max_deg
    ]


def random_poly(rng: random.Random, nvars: int, max_deg: int, max_terms: int) -> Poly:
    """生成稀疏随机多项式：单项式总次数 <= max_deg，系数在 [-3,3] 内的非零有理数。

    先枚举单项式池再不重复采样，避免“可选项耗尽导致重试死循环”。
    """
    pool = _monomial_pool(nvars, max_deg)
    rng.shuffle(pool)
    count = min(rng.randint(1, max_terms), len(pool))
    pairs = []
    for e in pool[:count]:
        c = rng.randint(-3, 3)
        while c == 0:  # 保证非零系数
            c = rng.randint(-3, 3)
        pairs.append((e, Fraction(c)))
    return Poly.from_terms(pairs, nvars=nvars)


class TestRandomizedAgainstReference(unittest.TestCase):
    @unittest.skipIf(__import__("sys").version_info[:2] < (3, 8), "需要 Python 3.8+")
    def test_random_ideals(self):
        rng = random.Random(20260923)
        trials = 30
        for trial in range(trials):
            nvars = rng.choice([1, 2, 2, 3])
            k = rng.randint(1, 3)
            gens = [random_poly(rng, nvars, max_deg=2, max_terms=3) for _ in range(k)]
            # 允许零生成元、重复
            if trial % 5 == 0:
                gens.append(Poly.zero(nvars))
            if trial % 7 == 0:
                gens.append(gens[0])
            ref_gens = [to_ref(g) for g in gens]

            res = groebner_basis(gens)
            self.assertTrue(res.complete, f"trial {trial}: 默认预算下应完成")
            rb = reduced_basis(res)

            # 1) 与独立参考的首一约化基逐成员相等
            ref_gb = R.ref_reduced_basis(ref_gens)
            self.assertEqual(
                canon([to_ref(p) for p in rb.basis]),
                canon(ref_gb),
                f"trial {trial} 约化基不一致",
            )

            # 2) 全部最终 S 多项式余式为 0
            for a, b in combinations(rb.basis, 2):
                self.assertTrue(
                    remainder(s_polynomial(a, b), rb.basis).is_zero(),
                    f"trial {trial}: S 余式非零",
                )

            # 3) 证书独立展开
            for p, combo in zip(rb.basis, rb.combinations):
                expanded = R.ref_expand(
                    {i: to_ref(q) for i, q in combo.items()}, ref_gens
                )
                self.assertEqual(expanded, to_ref(p), f"trial {trial}: 证书展开不符")

            # 4) 随机被除式的除法：恒等式 + 与参考余式一致
            f = random_poly(rng, nvars, max_deg=3, max_terms=4)
            div = divide(f, rb.basis)
            self.assertEqual(div.identity(), f, f"trial {trial}: 除法恒等式不成立")
            self.assertEqual(
                to_ref(div.remainder),
                R.ref_rem(to_ref(f), ref_gb),
                f"trial {trial}: GB 余式与参考不符",
            )

            # 5) 成员判定与参考一致；成员证书可展开
            m = ideal_membership(f, gens)
            ref_r = R.ref_rem(to_ref(f), ref_gb)
            if ref_r == {}:
                self.assertEqual(m.status, "member", f"trial {trial}")
                self.assertTrue(m.remainder.is_zero())
                expanded = R.ref_expand(
                    {i: to_ref(q) for i, q in m.coefficients.items()}, ref_gens
                )
                self.assertEqual(expanded, to_ref(f), f"trial {trial}: 成员证书不符")
            else:
                self.assertEqual(m.status, "nonmember", f"trial {trial}")
                self.assertEqual(to_ref(m.remainder), ref_r)


if __name__ == "__main__":
    unittest.main()
