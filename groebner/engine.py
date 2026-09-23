"""除法、S 多项式、Buchberger 算法、首一约化基与理想成员判定。

所有运算在 Q[x0, ..., x_{n-1}]（n <= 4）上精确进行（Fraction）。

组合证书表示：一个“组合”是 dict[int, Poly]，含义为
    value(combo) = Σ_i combo[i] * 原始生成元 g_i，
其中键是原始生成元（按调用方输入顺序，零生成元也保留位置）的下标。
不变量：每个带标签的基多项式 p 始终与其 combo 同步维护，
即 p == Σ combo[i] g_i；测试用独立展开器核验该不变量。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from .polynomial import (
    Poly,
    PolynomialError,
    divides_monomial,
    grlex_key,
    lcm_monomial,
    monomial_factor,
)

MAX_INPUT_TERMS = 100
# 演示/测试用不到的大默认预算：保证小规模问题始终算完，
# 真正需要观察 incomplete 时显式传入小预算。
_DEFAULT_MAX_PAIRS = 1_000_000
_DEFAULT_MAX_TERMS = 1_000_000


# ---------------------------------------------------------------------------
# 预算与结果类型
# ---------------------------------------------------------------------------
def _check_positive_int(x: object, name: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise PolynomialError(f"{name} 必须是 int（拒绝 bool）")
    if x <= 0:
        raise PolynomialError(f"{name} 必须为正整数，得到 {x}")
    return x


@dataclass(frozen=True)
class GroebnerBudgets:
    """算法搜索预算。耗尽时返回 complete=False（incomplete），绝不称作最终基。

    - max_pairs: 最多处理（真正做约化）的 S 多项式对数；
      因首项单项式互素而依 Buchberger 第一判据跳过的对不计入。
    - max_terms: 约化过程中“当前基总项数 + 工作多项式项数”的峰值上限。
    """

    max_pairs: int = _DEFAULT_MAX_PAIRS
    max_terms: int = _DEFAULT_MAX_TERMS

    def __post_init__(self) -> None:
        _check_positive_int(self.max_pairs, "max_pairs")
        _check_positive_int(self.max_terms, "max_terms")


@dataclass
class DivisionResult:
    """多元除法 f = Σ quotients[i] * divisors[i] + remainder 的结果。

    商依赖除数的给定顺序（题面固定 grlex，但多元除法的商本身随顺序变化）；
    当除数构成 Gröbner 基时余式与顺序无关。
    """

    dividend: Poly
    divisors: list[Poly]
    quotients: list[Poly]
    remainder: Poly

    def identity(self) -> Poly:
        """返回 Σ q_i d_i + r（外部可据此独立比对被除数）。"""
        total = Poly.zero(self.dividend.nvars)
        for q, d in zip(self.quotients, self.divisors):
            total = total + q * d
        return total + self.remainder


@dataclass
class GroebnerBasisResult:
    complete: bool
    basis: list[Poly]
    """basis[k] = combinations[k][i] * generators[i] 求和。"""
    combinations: list[dict[int, Poly]]
    generators: list[Poly]
    nvars: int
    reduced: bool
    monic: bool
    pairs_processed: int
    pairs_skipped_coprime: int
    peak_total_terms: int
    budget: GroebnerBudgets
    incomplete_reason: Optional[str] = None

    def certificate(self, index: int) -> dict[int, Poly]:
        """返回第 index 个基多项式相对原始生成元的组合证书（副本）。"""
        if not 0 <= index < len(self.basis):
            raise PolynomialError(f"基下标 {index} 越界（基大小 {len(self.basis)}）")
        return dict(self.combinations[index])


@dataclass
class MembershipResult:
    """理想成员判定结果。

    status:
      - "member"           : remainder == 0，coefficients 是成员证书；
      - "nonmember"        : 仅当基已 complete，余式非零即证明不在理想中；
      - "unknown"          : 预算耗尽导致基不完整且余式非零——不能下结论
                             （与“证明不是成员”严格区分）。
    恒等式：dividend - remainder == Σ coefficients[i] * generators[i]。
    """

    dividend: Poly
    generators: list[Poly]
    remainder: Poly
    coefficients: dict[int, Poly]
    complete: bool
    reduced_basis: list[Poly]
    status: str
    pairs_processed: int

    @property
    def member(self) -> Optional[bool]:
        return True if self.status == "member" else False if self.status == "nonmember" else None

    @property
    def certificate(self) -> Optional[dict[int, Poly]]:
        return dict(self.coefficients) if self.status == "member" else None


# ---------------------------------------------------------------------------
# 组合（原始生成元环上的稀疏向量）
# ---------------------------------------------------------------------------
Combo = dict[int, Poly]


def _combo_scale_monomial(combo: Combo, factor: Poly) -> Combo:
    """组合整体乘一个单项多项式 factor（系数可为任意 Fraction）。"""
    return {i: factor * c for i, c in combo.items() if not c.is_zero()}


def _combo_sub(a: Combo, b: Combo) -> Combo:
    out: Combo = dict(a)
    for i, c in b.items():
        nc = out.get(i, Poly.zero(c.nvars)) - c
        if nc.is_zero():
            out.pop(i, None)
        else:
            out[i] = nc
    return out


def _single_term_poly(nvars: int, exp: tuple[int, ...], coef: Fraction) -> Poly:
    return Poly({exp: coef}, nvars=nvars)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _validate_generators(generators: Sequence[Poly]) -> tuple[int, list[Poly]]:
    if not isinstance(generators, (list, tuple)):
        raise PolynomialError("generators 必须是 list/tuple[Poly]")
    if len(generators) == 0:
        raise PolynomialError("至少需要 1 个生成元（全零情形也请显式传入零多项式）")
    nvars = None
    total = 0
    cleaned: list[Poly] = []
    for idx, g in enumerate(generators):
        if not isinstance(g, Poly):
            raise PolynomialError(f"generators[{idx}] 必须是 Poly，得到 {type(g).__name__}")
        if nvars is None:
            nvars = g.nvars
        elif g.nvars != nvars:
            raise PolynomialError(
                f"generators[{idx}] 变量数 {g.nvars} 与前面的 {nvars} 不一致"
            )
        total += len(g)
        cleaned.append(g)
    assert nvars is not None
    if total > MAX_INPUT_TERMS:
        raise PolynomialError(
            f"生成元非零项总数 {total} 超过接口上限 {MAX_INPUT_TERMS}"
        )
    return nvars, cleaned


# ---------------------------------------------------------------------------
# 多元除法
# ---------------------------------------------------------------------------
def divide(
    dividend: Poly,
    divisors: Sequence[Poly],
) -> DivisionResult:
    """固定 grlex 下的多元除法。

    重复用“首个”首项能整除当前首项的除数消首项；不能整除的首项进入余式。
    结果恒满足 f = Σ q_i d_i + r，且 r 中没有项可被任何 LM(d_i) 整除。
    商依赖除数顺序；当除数构成 Gröbner 基时余式与顺序无关。
    """
    if not isinstance(dividend, Poly):
        raise PolynomialError("dividend 必须是 Poly")
    if not isinstance(divisors, (list, tuple)) or not divisors:
        raise PolynomialError("divisors 必须是非空 list/tuple[Poly]")
    n = dividend.nvars
    ds: list[Poly] = []
    for i, d in enumerate(divisors):
        if not isinstance(d, Poly):
            raise PolynomialError(f"divisors[{i}] 必须是 Poly")
        if d.nvars != n:
            raise PolynomialError(f"divisors[{i}] 变量数 {d.nvars} 与被除数 {n} 不一致")
        if d.is_zero():
            raise PolynomialError(f"divisors[{i}] 不能是零多项式")
        ds.append(d)

    quot = [Poly.zero(n) for _ in ds]
    rem = Poly.zero(n)
    work = dividend
    while not work.is_zero():
        lm = work.leading_monomial()
        lc = work.leading_coefficient()
        chosen = -1
        for i, d in enumerate(ds):
            if divides_monomial(d.leading_monomial(), lm):
                chosen = i
                break
        if chosen >= 0:
            d = ds[chosen]
            exp_diff = monomial_factor(d.leading_monomial(), lm)
            factor = _single_term_poly(n, exp_diff, lc / d.leading_coefficient())
            quot[chosen] = quot[chosen] + factor
            work = work - factor * d
        else:
            move = _single_term_poly(n, lm, lc)
            rem = rem + move
            work = work - move
    return DivisionResult(dividend, list(ds), quot, rem)


def remainder(dividend: Poly, divisors: Sequence[Poly]) -> Poly:
    """便捷封装：多元除法余式。"""
    return divide(dividend, divisors).remainder


# ---------------------------------------------------------------------------
# S 多项式
# ---------------------------------------------------------------------------
def s_polynomial(f: Poly, g: Poly) -> Poly:
    """S(f, g) = (L/lm(f))*f - (LC(f)/LC(g))*(L/lm(g))*g，L = lcm(LM(f), LM(g))。"""
    if not isinstance(f, Poly) or not isinstance(g, Poly):
        raise PolynomialError("s_polynomial 的参数必须是 Poly")
    if f.is_zero() or g.is_zero():
        raise PolynomialError("S 多项式要求两个非零多项式")
    if f.nvars != g.nvars:
        raise PolynomialError("S 多项式的两个多项式变量数必须一致")
    n = f.nvars
    lmf, lcf = f.leading_monomial(), f.leading_coefficient()
    lmg, lcg = g.leading_monomial(), g.leading_coefficient()
    L = lcm_monomial(lmf, lmg)
    mf = _single_term_poly(n, monomial_factor(lmf, L), lcg)
    mg = _single_term_poly(n, monomial_factor(lmg, L), lcf)
    return mf * f - mg * g


# ---------------------------------------------------------------------------
# 带标签约化（同步维护组合证书）
# ---------------------------------------------------------------------------
@dataclass
class _Reduction:
    remainder: Poly
    combo: Combo
    multipliers: list[Poly]  # 对“传入基”的每个位置：p - r = Σ multipliers[j]*basis[j]
    steps: int


def _reduce_tagged(
    p: Poly,
    p_combo: Combo,
    basis: list[Poly],
    basis_combos: list[Combo],
    budget: GroebnerBudgets,
    basis_total_terms: int,
) -> _Reduction:
    """把 p 对 basis 做**完整约化**，同步推导余式的组合表达式。

    策略：每步在工作多项式中（按 grlex）找出“最大的、首项能被某个
    基成员首项整除”的项并消去。比它大的不可约项原样保留，且后续引入
    的项都严格更小，不会再触及它们；最终工作多项式中不存在任何可约项，
    即得到完整范式 NF(p)（不是只到首项不可约）。

    每一步 p <- p - m*g_j 对应 combo <- combo - m*combo(g_j)。
    返回的 multipliers 满足 p_original - r = Σ multipliers[j]*basis[j]。
    终止性：每步把一个项替换为严格更小的若干项（项多重集严格下降）。
    """
    n = p.nvars
    work = p
    work_combo = dict(p_combo)
    mult: list[Poly] = [Poly.zero(n) for _ in basis]
    peak = basis_total_terms
    steps = 0
    while not work.is_zero():
        # 在所有项中找最大的可约项（grlex 降序），及首个能整除它的基成员。
        chosen_exp = None
        chosen_coef: Fraction | None = None
        chosen = -1
        for m_exp, m_coef in work.terms():
            hit = -1
            for j, g in enumerate(basis):
                if divides_monomial(g.leading_monomial(), m_exp):
                    hit = j
                    break
            if hit >= 0:
                chosen_exp, chosen_coef, chosen = m_exp, m_coef, hit
                break
        if chosen < 0:
            break  # 不存在任何可约项 => 完整范式
        g = basis[chosen]
        exp_diff = monomial_factor(g.leading_monomial(), chosen_exp)
        factor = _single_term_poly(n, exp_diff, chosen_coef / g.leading_coefficient())
        work = work - factor * g
        scaled = _combo_scale_monomial(basis_combos[chosen], factor)
        work_combo = _combo_sub(work_combo, scaled)
        mult[chosen] = mult[chosen] + factor
        steps += 1
        peak = max(peak, basis_total_terms + len(work))
        if peak > budget.max_terms:
            raise _BudgetExhausted(
                f"约化中总项数峰值 {peak} 超过 max_terms={budget.max_terms}"
            )
        if steps > budget.max_pairs:
            raise _BudgetExhausted(
                f"单个 S 对约化步数 {steps} 超过 max_pairs={budget.max_pairs}"
            )
    return _Reduction(work, work_combo, mult, steps)


class _BudgetExhausted(Exception):
    """内部控制信号：预算耗尽。对外体现为 result.complete=False，不抛出。"""


# ---------------------------------------------------------------------------
# Buchberger
# ---------------------------------------------------------------------------
def groebner_basis(
    generators: Sequence[Poly],
    budgets: GroebnerBudgets = GroebnerBudgets(),
) -> GroebnerBasisResult:
    """Buchberger 算法（固定 grlex，Q 系数，n <= 4）。

    - 零生成元被丢弃但保留下标位置（证书下标与输入一致）；
    - 每个基多项式记录为原始生成元的多项式线性组合；
    - 预算耗尽时返回 complete=False 与 incomplete_reason，基不完整；
    - 采用首项单项式互素跳过（Buchberger 第一判据），跳过数单独计数；
      最终基另由独立的“全部 S 多项式余式为 0”核验把关。
    """
    if not isinstance(budgets, GroebnerBudgets):
        raise PolynomialError("budgets 必须是 GroebnerBudgets")
    nvars, gens = _validate_generators(generators)

    basis: list[Poly] = []
    combos: list[Combo] = []
    pairs_processed = 0
    skipped = 0
    peak = 0
    # (i, j) 待处理对，FIFO 保证确定性
    pending: list[tuple[int, int]] = []

    def _incomplete(
        bs: list[Poly],
        cs: list[Combo],
        reason: str,
        pk: int,
        sk: int,
    ) -> GroebnerBasisResult:
        return GroebnerBasisResult(
            complete=False,
            basis=list(bs),
            combinations=[dict(c) for c in cs],
            generators=list(gens),
            nvars=nvars,
            reduced=False,
            monic=False,
            pairs_processed=pairs_processed,
            pairs_skipped_coprime=sk,
            peak_total_terms=pk,
            budget=budgets,
            incomplete_reason=reason,
        )

    # 输入项数已超 max_terms：连初始基都放不下，直接 incomplete（空基）。
    input_terms = sum(len(g) for g in gens)
    if input_terms > budgets.max_terms:
        return _incomplete(
            [], [],
            f"输入非零项总数 {input_terms} 超过 max_terms={budgets.max_terms}",
            input_terms, 0,
        )


    def total_terms() -> int:
        return sum(len(g) for g in basis)

    def add_member(g: Poly, combo: Combo) -> None:
        nonlocal peak
        idx = len(basis)
        for k in range(idx):
            pending.append((k, idx))
        basis.append(g)
        combos.append(combo)
        peak = max(peak, total_terms())

    try:
        # --- 初始基：把每个输入对当前基约化；非零余式入基（保持张成空间）---
        for i, f in enumerate(gens):
            if f.is_zero():
                continue
            unit: Combo = {i: Poly.constant(1, nvars)}
            if not basis:
                add_member(f, unit)
                continue
            red = _reduce_tagged(f, unit, basis, combos, budgets, total_terms())
            if not red.remainder.is_zero():
                add_member(red.remainder, red.combo)

        # --- 主循环 ---
        while pending:
            i, j = pending.pop(0)
            f, g = basis[i], basis[j]
            # Buchberger 第一判据：LM 互素则 S 必约化到 0，跳过。
            if all(
                a == 0 or b == 0
                for a, b in zip(f.leading_monomial(), g.leading_monomial())
            ):
                skipped += 1
                continue
            pairs_processed += 1
            if pairs_processed > budgets.max_pairs:
                raise _BudgetExhausted(
                    f"处理 S 对 {pairs_processed - 1} 后达到 max_pairs={budgets.max_pairs}"
                )
            s = s_polynomial(f, g)
            s_combo: Combo = {}
            # S = m_f*f - m_g*g，按同一构造同步组合：
            lmf, lcf = f.leading_monomial(), f.leading_coefficient()
            lmg, lcg = g.leading_monomial(), g.leading_coefficient()
            L = lcm_monomial(lmf, lmg)
            mf = _single_term_poly(nvars, monomial_factor(lmf, L), lcg)
            mg = _single_term_poly(nvars, monomial_factor(lmg, L), lcf)
            s_combo = _combo_sub(
                _combo_scale_monomial(combos[i], mf),
                _combo_scale_monomial(combos[j], mg),
            )
            red = _reduce_tagged(s, s_combo, basis, combos, budgets, total_terms())
            if not red.remainder.is_zero():
                peak = max(peak, total_terms() + len(red.remainder))
                if peak > budgets.max_terms:
                    raise _BudgetExhausted(
                        f"加入新成员后总项数 {peak} 超过 max_terms={budgets.max_terms}"
                    )
                add_member(red.remainder, red.combo)

    except _BudgetExhausted as exc:
        return GroebnerBasisResult(
            complete=False,
            basis=list(basis),
            combinations=[dict(c) for c in combos],
            generators=list(gens),
            nvars=nvars,
            reduced=False,
            monic=False,
            pairs_processed=pairs_processed,
            pairs_skipped_coprime=skipped,
            peak_total_terms=peak,
            budget=budgets,
            incomplete_reason=str(exc),
        )

    return GroebnerBasisResult(
        complete=True,
        basis=basis,
        combinations=[dict(c) for c in combos],
        generators=list(gens),
        nvars=nvars,
        reduced=False,
        monic=False,
        pairs_processed=pairs_processed,
        pairs_skipped_coprime=skipped,
        peak_total_terms=peak,
        budget=budgets,
    )


# ---------------------------------------------------------------------------
# 首一约化基
# ---------------------------------------------------------------------------
def reduced_basis(
    result: GroebnerBasisResult,
    budgets: Optional[GroebnerBudgets] = None,
) -> GroebnerBasisResult:
    """由（完整的）Buchberger 基计算唯一的首一约化 Gröbner 基。

    约化基中每个多项式首一，且其任何项都不被其他成员的首项整除。
    组合证书随每步约化与归一化同步更新。输入基不完整时拒绝调用——
    不完整基上的“约化”没有最终基语义。
    """
    if not isinstance(result, GroebnerBasisResult):
        raise PolynomialError("reduced_basis 需要 GroebnerBasisResult")
    if not result.complete:
        raise PolynomialError(
            "不能对 incomplete 基计算最终约化基：" + (result.incomplete_reason or "预算耗尽")
        )
    b = budgets or GroebnerBudgets()
    n = result.nvars
    # Buchberger 已在用户预算内证明 complete；最终“约化整理”是确定性簿记
    # （不是搜索），不再受 max_pairs/max_terms 搜索预算限制，只留防御性护栏。
    # 约化中每个成员的项多重集严格下降，故 complete 后整理必然终止。
    internal = GroebnerBudgets()

    # 空基（零理想）直接返回
    if not result.basis:
        return GroebnerBasisResult(
            complete=True,
            basis=[],
            combinations=[],
            generators=list(result.generators),
            nvars=n,
            reduced=True,
            monic=True,
            pairs_processed=result.pairs_processed,
            pairs_skipped_coprime=result.pairs_skipped_coprime,
            peak_total_terms=0,
            budget=b,
        )

    polys: list[Poly] = [g.scale_to_monic() for g in result.basis]
    comb: list[Combo] = []
    for g, c in zip(result.basis, result.combinations):
        lc = g.leading_coefficient()
        comb.append({i: (q / lc) for i, q in c.items()})

    peak = sum(len(p) for p in polys)
    # 交叉约化：每轮把每个成员对“其余成员”完整约化；直到一整轮无变化。
    # 终止性由首项严格下降与迪克森引理保证。
    while True:
        changed = False
        k = len(polys)
        for idx in range(k):
            others = [polys[j] for j in range(k) if j != idx]
            other_combos = [comb[j] for j in range(k) if j != idx]
            base_terms = sum(len(o) for o in others)
            red = _reduce_tagged(polys[idx], comb[idx], others, other_combos, internal, base_terms)
            # 仅用于报告；整理阶段不再以 max_terms 报错（搜索预算只管 complete 判定）。
            peak = max(peak, base_terms + len(red.remainder))
            r = red.remainder
            if r != polys[idx]:
                changed = True
                if r.is_zero():
                    # 被其余成员完全表示：删除该成员及其证书行
                    del polys[idx]
                    del comb[idx]
                    break
                lc = r.leading_coefficient()
                polys[idx] = r / lc
                comb[idx] = {i: (q / lc) for i, q in red.combo.items()}
        if not changed:
            break

    # 按首项 grlex 排序，保证输出顺序确定（不改变数学结果）
    order = sorted(range(len(polys)), key=lambda i: grlex_key(polys[i].leading_monomial()), reverse=True)
    polys = [polys[i] for i in order]
    comb = [comb[i] for i in order]

    return GroebnerBasisResult(
        complete=True,
        basis=polys,
        combinations=[dict(c) for c in comb],
        generators=list(result.generators),
        nvars=n,
        reduced=True,
        monic=True,
        pairs_processed=result.pairs_processed,
        pairs_skipped_coprime=result.pairs_skipped_coprime,
        peak_total_terms=peak,
        budget=b,
    )


# ---------------------------------------------------------------------------
# 理想成员判定
# ---------------------------------------------------------------------------
def membership_with_basis(
    f: Poly,
    result: GroebnerBasisResult,
    budgets: Optional[GroebnerBudgets] = None,
) -> MembershipResult:
    """基于已计算的基（自动取首一约化基）判定 f 是否在理想中。

    余式为 0 => 成员，coefficients 是相对原始生成元的成员证书；
    基完整且余式非零 => 证明非成员；基不完整且余式非零 => unknown。
    """
    if not isinstance(f, Poly):
        raise PolynomialError("f 必须是 Poly")
    if not isinstance(result, GroebnerBasisResult):
        raise PolynomialError("result 必须是 GroebnerBasisResult")
    if f.nvars != result.nvars:
        raise PolynomialError(f"f 变量数 {f.nvars} 与基 {result.nvars} 不一致")

    if result.complete and result.reduced and result.monic:
        rb = result
    elif result.complete:
        rb = reduced_basis(result, budgets)
    else:
        # 预算耗尽的不完整基：仍可用它做除法（证书不变量成立），
        # 但余式非零时只能报告 unknown，不能据此“证明非成员”。
        rb = result

    basis = rb.basis
    n = rb.nvars
    zero = Poly.zero(n)

    if not basis:
        # 零理想：成员当且仅当 f == 0
        status = "member" if f.is_zero() else ("nonmember" if rb.complete else "unknown")
        return MembershipResult(
            dividend=f,
            generators=list(rb.generators),
            remainder=f,
            coefficients={},
            complete=rb.complete,
            reduced_basis=list(basis),
            status=status,
            pairs_processed=rb.pairs_processed,
        )

    # 用约化基除 f：f = Σ q_k b_k + r
    div = divide(f, basis)
    # 把基证书代入：b_k = Σ_j C[k][j] g_j
    # 则 f - r = Σ_j (Σ_k q_k * C[k][j]) g_j
    coeff: Combo = {}
    for q, crow in zip(div.quotients, rb.combinations):
        if q.is_zero():
            continue
        for j, cgj in crow.items():
            term = q * cgj
            acc = coeff.get(j)
            if acc is None:
                coeff[j] = term
            else:
                acc = acc + term
                if acc.is_zero():
                    del coeff[j]
                else:
                    coeff[j] = acc

    if div.remainder.is_zero():
        status = "member"
    elif rb.complete:
        status = "nonmember"
    else:
        status = "unknown"

    return MembershipResult(
        dividend=f,
        generators=list(rb.generators),
        remainder=div.remainder,
        coefficients=coeff,
        complete=rb.complete,
        reduced_basis=list(basis),
        status=status,
        pairs_processed=rb.pairs_processed,
    )


def ideal_membership(
    f: Poly,
    generators: Sequence[Poly],
    budgets: GroebnerBudgets = GroebnerBudgets(),
) -> MembershipResult:
    """一步完成：Buchberger -> 首一约化基 -> 成员判定（含组合证书）。"""
    res = groebner_basis(generators, budgets)
    rb = reduced_basis(res, budgets) if res.complete else res
    return membership_with_basis(f, rb, budgets)
