"""次数受限的 syzygy：求 sum_i a_i * f_i = 0 且 deg(a_i) <= d 的全部系数向量。

给定生成元 f_1..f_m 与次数上限 d，本模块把“系数多项式次数不超过 d 的
关系”展开成有理数域上的**有限维线性系统**并精确求解：

- 未知量：每个 a_i 在每个总次数 <= d 的单项式上的系数；
- 方程：sum_i a_i * f_i 在每个可能出现（总次数 <= d + max deg(f_i)）
  的单项式上的系数等于 0；
- 消元：精确 Fraction 高斯消元到行最简形（RREF），零空间的规范基
  （每个自由列取 1、其余自由列取 0）即所求系数向量空间的规范基。

**明确范围声明**：这里只求次数 <= d 的有限维空间，**不**声称生成了
完整的 syzygy 模（后者需要 Schreyer 定理等模块级算法，属于非目标）。

零生成元、重复生成元、常数生成元都自然纳入同一线性系统：
零生成元的系数列全部自由；常数生成元把关系约束为“其余乘积之和必须
是常数倍”；坐标始终跟随**原始输入顺序**，输入排列时关系逐坐标跟随，
不丢不重。

核验是独立的：relation_polynomial 逐多项式相乘求和重算
sum_i a_i * f_i；结果对象携带 rank 与 num_unknowns，
完备性由维数公式 dimension == num_unknowns - rank 独立复算保证。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .errors import InvalidGeneratorError
from .groebner import _validate_generators
from .poly import Poly, grlex_key

#: 线性系统规模的防御性上限（未知数个数 / 方程个数）。
#: 维度公式保证任何输入都在有限时间内完成或在此熔断；可用关键字调大。
DEFAULT_MAX_UNKNOWNS = 20000
DEFAULT_MAX_EQUATIONS = 20000


def _check_degree(value: object, name: str = "max_degree") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGeneratorError(f"{name} 必须是非负 int，得到 {value!r}")
    if value < 0:
        raise InvalidGeneratorError(f"{name} 必须是非负 int，得到 {value}")
    return value


def _check_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGeneratorError(f"{name} 必须是正 int，得到 {value!r}")
    if value <= 0:
        raise InvalidGeneratorError(f"{name} 必须是正 int，得到 {value}")
    return value


def monomials_up_to(n: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """n 元、总次数 <= degree 的全部单项式指数，按 grlex 降序。"""
    out: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, acc: list[int]) -> None:
        if pos == n - 1:
            for e in range(remaining + 1):
                acc.append(e)
                out.append(tuple(acc))
                acc.pop()
            return
        for e in range(remaining + 1):
            acc.append(e)
            rec(pos + 1, remaining - e, acc)
            acc.pop()

    rec(0, degree, [])
    out.sort(key=grlex_key, reverse=True)
    return tuple(out)


@dataclass(frozen=True)
class BoundedSyzygyResult:
    """次数受限 syzygy 空间的计算结果。

    - basis：系数向量空间的规范基；每个元素是长度等于输入生成元个数
      的 Poly 元组 (a_1, ..., a_m)，满足 sum_i a_i * f_i = 0
      且 deg(a_i) <= max_degree；坐标顺序与**原始输入**一一对应
      （含零生成元、重复生成元的位置）；
    - column_monomials：线性系统第 k 列对应的 (生成元下标, 单项式指数)，
      即系数列映射；
    - dimension：解空间维数（len(basis)）；
    - rank / num_unknowns / num_equations：线性系统的秩与规模，
      满足 dimension == num_unknowns - rank，供独立复算完备性。
    """

    generators: tuple[Poly, ...]
    max_degree: int
    basis: tuple[tuple[Poly, ...], ...]
    column_monomials: tuple[tuple[int, tuple[int, ...]], ...]
    dimension: int
    rank: int
    num_unknowns: int
    num_equations: int

    def relation(self, index: int) -> tuple[Poly, ...]:
        """取第 index 条基关系（系数向量）。"""
        return self.basis[index]

    def verify(self, index: int) -> bool:
        """独立逐多项式相乘求和，核验第 index 条关系确实为 0。"""
        return relation_polynomial(self.basis[index], self.generators).is_zero()


def relation_polynomial(coefficients: Sequence[Poly],
                        generators: Sequence[Poly]) -> Poly:
    """独立重算 sum_i coefficients[i] * generators[i]（逐多项式相乘求和）。

    只使用多项式加法/乘法，与线性系统的求解路径无关，
    是对每条关系的独立核验原语。
    """
    if len(coefficients) != len(generators):
        raise InvalidGeneratorError(
            f"系数向量长度 {len(coefficients)} 与生成元个数 "
            f"{len(generators)} 不一致")
    if not generators:
        raise InvalidGeneratorError("生成元序列为空，无法确定变量数")
    n = generators[0].n
    total = Poly.zero(n)
    for idx, (a, f) in enumerate(zip(coefficients, generators)):
        if not isinstance(a, Poly) or a.n != n:
            raise InvalidGeneratorError(
                f"第 {idx} 个系数不是 {n} 元 Poly")
        total = total + a * f
    return total


def _rref(matrix: list[list[Fraction]], num_cols: int
          ) -> tuple[list[list[Fraction]], list[int]]:
    """精确行最简形消元，返回 (rref 行列表, 主元列下标)。"""
    rows = [row for row in matrix if any(row)]
    pivots: list[int] = []
    r = 0
    for c in range(num_cols):
        pivot = -1
        for k in range(r, len(rows)):
            if rows[k][c]:
                pivot = k
                break
        if pivot < 0:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for k in range(len(rows)):
            if k != r and rows[k][c]:
                factor = rows[k][c]
                rows[k] = [v - factor * w for v, w in zip(rows[k], rows[r])]
        pivots.append(c)
        r += 1
        if r == len(rows):
            break
    return rows[:r], pivots


def bounded_syzygies(generators: Iterable[Poly],
                     max_degree: int,
                     *,
                     max_unknowns: int = DEFAULT_MAX_UNKNOWNS,
                     max_equations: int = DEFAULT_MAX_EQUATIONS
                     ) -> BoundedSyzygyResult:
    """求 sum_i a_i * f_i = 0 且 deg(a_i) <= max_degree 的系数向量空间基。

    返回 BoundedSyzygyResult；basis 为空元组表示该次数界内**不存在**
    非平凡关系（这是精确证明，不是“没找到”）。

    规模超过 max_unknowns / max_equations 时抛 InvalidGeneratorError
    （可调大预算重试）；输入校验语义与 groebner_basis 一致（空序列、
    变量数不一致、项数超限、max_degree 为负或 bool 等均拒绝）。
    """
    d = _check_degree(max_degree)
    max_unknowns = _check_limit(max_unknowns, "max_unknowns")
    max_equations = _check_limit(max_equations, "max_equations")
    gens = _validate_generators(generators)
    n = gens[0].n
    m = len(gens)

    monoms = monomials_up_to(n, d)
    columns: list[tuple[int, tuple[int, ...]]] = [
        (i, alpha) for i in range(m) for alpha in monoms
    ]
    num_unknowns = len(columns)
    if num_unknowns > max_unknowns:
        raise InvalidGeneratorError(
            f"未知数个数 {num_unknowns} 超过预算 max_unknowns={max_unknowns}"
            f"（{m} 个生成元 × 次数<= {d} 的 {len(monoms)} 个单项式）；"
            f"请降低 max_degree 或调大预算")

    max_deg_f = 0
    for g in gens:
        for exp in g.terms:
            max_deg_f = max(max_deg_f, sum(exp))
    eq_monoms = monomials_up_to(n, d + max_deg_f)
    num_equations = len(eq_monoms)
    if num_equations > max_equations:
        raise InvalidGeneratorError(
            f"方程个数 {num_equations} 超过预算 max_equations={max_equations}；"
            f"请降低 max_degree 或调大预算")

    # 展开线性系统：第 (beta) 行、第 (i, alpha) 列 = f_i 在 beta - alpha 处的系数。
    gen_terms = [g.terms for g in gens]
    matrix: list[list[Fraction]] = []
    for beta in eq_monoms:
        row = [Fraction(0)] * num_unknowns
        for col, (i, alpha) in enumerate(columns):
            diff = tuple(b - a for a, b in zip(alpha, beta))
            if all(x >= 0 for x in diff):
                coef = gen_terms[i].get(diff)
                if coef:
                    row[col] = coef
        matrix.append(row)

    rref_rows, pivots = _rref(matrix, num_unknowns)
    rank = len(pivots)
    pivot_set = set(pivots)
    free_cols = [c for c in range(num_unknowns) if c not in pivot_set]

    basis: list[tuple[Poly, ...]] = []
    for free in free_cols:
        vec = [Fraction(0)] * num_unknowns
        vec[free] = Fraction(1)
        for row, pivot_col in zip(rref_rows, pivots):
            vec[pivot_col] = -row[free]
        coeffs: list[Poly] = []
        for i in range(m):
            terms = {}
            for k, alpha in enumerate(monoms):
                value = vec[i * len(monoms) + k]
                if value:
                    terms[alpha] = value
            coeffs.append(Poly._from_dict(terms, n))
        basis.append(tuple(coeffs))

    return BoundedSyzygyResult(
        generators=gens,
        max_degree=d,
        basis=tuple(basis),
        column_monomials=tuple(columns),
        dimension=len(basis),
        rank=rank,
        num_unknowns=num_unknowns,
        num_equations=num_equations,
    )
