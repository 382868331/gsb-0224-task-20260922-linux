"""次数有界 syzygy：求所有 sum_i a_i * f_i = 0 且 deg a_i <= d 的系数向量空间基。

数学口径（明确限定，不冒称完整 syzygy 模）：

- 只解**次数不超过 d** 的系数多项式 a_i，答案是一个**有限维有理向量空间**，
  本模块给出它的一组规范基；完整 syzygy 模（任意次数）不在本接口承诺范围内；
- 方法：把每个未知系数 a_i 按固定 grlex 次序的次数 <= d 单项式展开，
  sum_i a_i * f_i = 0 等价于关于这些未知系数的有理线性方程组，
  对其做**精确有理数** Gauss-Jordan 消元（RREF），取自由变量给出的标准零空间基；
- 零空间基向量个数 = 未知量个数 - 系数矩阵的秩，这一秩等式即
  “基空间维数完备”的证明；结果对象公开 rank 与 num_unknowns，
  供调用方/测试独立复算核验。

特殊输入（均自然纳入上述线性系统，不做特判）：

- 零生成元 f_i = 0：其列全为零，坐标 i 上任意次数 <= d 的多项式
  都是关系，基中对应出现该坐标的全部单项式单位向量；
- 重复生成元：如 f, f 给出关系 (1, -1)；
- 常数生成元：如 f_i = c != 0 时，a_i 须被其余项抵消，正常求解；
- 输入排列：坐标严格跟随输入顺序，不打乱、不去重，排列输入只排列输出坐标，
  关系本身不会丢失。

全程精确有理数（fractions.Fraction），无浮点、无容差。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .errors import InvalidGeneratorError
from .groebner import Certificate, _validate_generators
from .poly import Poly, grlex_key

#: 未知量总数（生成元个数 x 次数 <= d 的单项式个数）的防御性上限，
#: 保证任何合法输入的消元工作量有有限上界。可用关键字参数调大。
DEFAULT_MAX_UNKNOWNS = 4096


@dataclass(frozen=True)
class BoundedSyzygyResult:
    """次数有界 syzygy 的计算结果。

    - degree_bound：系数多项式的次数上限 d；
    - generators：原样保留的输入生成元（坐标顺序即输入顺序）；
    - monomials：每个坐标共用的“次数 <= d 单项式”列，按 grlex 降序；
    - columns：系数列映射，columns[k] = (生成元下标 i, 单项式指数)，
      即第 k 个未知量是 a_i 中该单项式的系数；
    - basis：规范零空间基，每个元素是长度等于生成元个数的
      Certificate（tuple[Poly, ...]），满足
      sum_i basis[r][i] * generators[i] == 0 且 deg basis[r][i] <= d；
    - rank / num_unknowns：系数矩阵的秩与未知量个数，
      恒有 dimension == num_unknowns - rank（维数完备的秩证明）。
    """

    degree_bound: int
    generators: tuple[Poly, ...]
    monomials: tuple[tuple[int, ...], ...]
    columns: tuple[tuple[int, tuple[int, ...]], ...]
    basis: tuple[Certificate, ...]
    rank: int
    num_unknowns: int

    @property
    def dimension(self) -> int:
        """解空间维数（基中关系条数），恒等于 num_unknowns - rank。"""
        return len(self.basis)


def _check_degree_bound(d: object) -> int:
    if isinstance(d, bool) or not isinstance(d, int):
        raise InvalidGeneratorError(
            f"次数上限 d 必须是非负 int，得到 {d!r}（bool 不被接受）")
    if d < 0:
        raise InvalidGeneratorError(f"次数上限 d 必须非负，得到 {d}")
    return d


def _check_unknowns_budget(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidGeneratorError(
            f"max_unknowns 必须是正 int，得到 {value!r}")
    return value


def monomials_up_to_degree(n: int, d: int) -> tuple[tuple[int, ...], ...]:
    """n 个变量、总次数 <= d 的全部单项式指数，按 grlex 降序（规范列序）。"""
    out: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, acc: list[int]) -> None:
        if pos == n - 1:
            for e in range(remaining + 1):
                out.append(tuple(acc + [e]))
            return
        for e in range(remaining + 1):
            rec(pos + 1, remaining - e, acc + [e])

    rec(0, d, [])
    out.sort(key=grlex_key, reverse=True)
    return tuple(out)


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """精确有理数 Gauss-Jordan 消元，返回 (行最简形, 主元列下标列表)。"""
    rows = [list(r) for r in matrix]
    ncols = len(rows[0]) if rows else 0
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        pivot = next((k for k in range(r, len(rows)) if rows[k][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for k in range(len(rows)):
            if k != r and rows[k][c] != 0:
                factor = rows[k][c]
                rows[k] = [v - factor * w for v, w in zip(rows[k], rows[r])]
        pivots.append(c)
        r += 1
        if r == len(rows):
            break
    return rows, pivots


def bounded_syzygies(generators: Iterable[Poly],
                     d: int,
                     *,
                     max_unknowns: int = DEFAULT_MAX_UNKNOWNS
                     ) -> BoundedSyzygyResult:
    """求所有 sum_i a_i * f_i = 0、deg a_i <= d 的系数向量空间基。

    只承诺次数 <= d 的有限维解空间；不声称生成完整 syzygy 模。
    返回 BoundedSyzygyResult，其 basis 是规范基
    （RREF 自由变量标准基，列序固定为 grlex 降序，结果唯一确定）。
    """
    d = _check_degree_bound(d)
    max_unknowns = _check_unknowns_budget(max_unknowns)
    gens = _validate_generators(generators)
    n = gens[0].n
    m = len(gens)

    mons = monomials_up_to_degree(n, d)
    num_unknowns = m * len(mons)
    if num_unknowns > max_unknowns:
        raise InvalidGeneratorError(
            f"未知量总数 {num_unknowns}（{m} 个生成元 x 次数<= {d} 的 "
            f"{len(mons)} 个单项式）超过上限 max_unknowns={max_unknowns}；"
            f"请降低次数上限 d 或显式调大 max_unknowns")

    # 系数列映射：第 k 个未知量 = a_i 中单项式 mons[j] 的系数。
    columns: tuple[tuple[int, tuple[int, ...]], ...] = tuple(
        (i, exp) for i in range(m) for exp in mons)

    # 按统一 grlex 单项式序展开 sum_i a_i f_i，逐单项式系数 = 0 得线性方程组。
    equations: dict[tuple[int, ...], list[Fraction]] = {}
    for col, (i, exp) in enumerate(columns):
        f = gens[i]
        for e_f, c in f.terms.items():
            beta = tuple(a + b for a, b in zip(exp, e_f))
            row = equations.get(beta)
            if row is None:
                row = [Fraction(0)] * num_unknowns
                equations[beta] = row
            row[col] += c
    matrix = [equations[beta]
              for beta in sorted(equations, key=grlex_key, reverse=True)]

    rref, pivots = _rref(matrix)
    rank = len(pivots)
    pivot_set = set(pivots)

    # 标准零空间基：每个自由列取 1，主元列取 -RREF 对应项；
    # 再统一“首一化”（按列序第一个非零系数化为 1），使基唯一确定，
    # 例如 [x, y] 在 d=1 时得到规范关系 (y, -x)。
    basis: list[Certificate] = []
    for free in range(num_unknowns):
        if free in pivot_set:
            continue
        vec = [Fraction(0)] * num_unknowns
        vec[free] = Fraction(1)
        for row_idx, pcol in enumerate(pivots):
            vec[pcol] = -rref[row_idx][free]
        lead = next(v for v in vec if v != 0)
        if lead != 1:
            vec = [v / lead for v in vec]
        relation: list[Poly] = []
        for i in range(m):
            terms = {mons[j]: vec[i * len(mons) + j]
                     for j in range(len(mons))
                     if vec[i * len(mons) + j] != 0}
            relation.append(Poly._from_dict(terms, n))
        basis.append(tuple(relation))

    return BoundedSyzygyResult(
        degree_bound=d,
        generators=gens,
        monomials=mons,
        columns=columns,
        basis=tuple(basis),
        rank=rank,
        num_unknowns=num_unknowns,
    )


def relation_value(relation: Sequence[Poly],
                   generators: Sequence[Poly]) -> Poly:
    """独立展开一条关系：sum_i relation[i] * generators[i]。

    只使用多项式加法/乘法重算，与线性系统的产生路径无关，
    用于核验关系确实取值零。
    """
    if len(relation) != len(generators):
        raise InvalidGeneratorError(
            f"关系长度 {len(relation)} 与生成元个数 {len(generators)} 不一致")
    total = Poly.zero(generators[0].n)
    for a, g in zip(relation, generators):
        total = total + a * g
    return total
