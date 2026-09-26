"""Buchberger 算法、约化首一 Groebner 基、理想成员判定与组合证书。

次序固定为 graded lex（见 :mod:`groebner.poly`），系数为精确有理数，
因此本模块**没有任何浮点运算与容差**：相等判定是精确的。

预算语义（“搜索预算耗尽”与“证明无解”严格分开）：

- ``max_pairs``：至多处理多少对关键对（含按判据直接跳过的对，
  跳过也计数，保证工作量可由预算界定）；
- ``max_terms``：约化过程中任何中间多项式的非零项数上限；
- ``max_steps``：跨所有约化累计的“初等约化步数”上限。项数限制无法
  约束“项数不多但步数极多”的病态约化，步数预算保证运行时间总有上界。

任一预算耗尽时 :class:`GroebnerResult` 的 ``status`` 为 ``"incomplete"``，
此时 ``basis`` 只是当前生成空间的一组首一生成元（带合法证书），
**不声明它是 Groebner 基**，``reduced`` 为 ``False``。

关键对管理采用正常选择策略（grlex 下 lcm 最小者优先），并使用
Buchberger 第一判据（首项互素）与第二判据（lcm 判据）安全跳过
必约为 0 的 S 多项式；两判据的正确性最终由完整基的 S 多项式核验独立保证。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence

from .errors import (
    InvalidGeneratorError,
    InvalidPolynomialError,
    ZeroDivisorError,
)
from .poly import (
    MAX_TOTAL_TERMS,
    MAX_VARIABLES,
    Poly,
    grlex_key,
    lcm_monomial,
)

#: 组合证书：长度等于输入生成元个数（含零生成元的位置）。
#: ``certificate[j]`` 是乘在第 j 个输入生成元前的多项式系数。
Certificate = tuple[Poly, ...]

DEFAULT_MAX_PAIRS = 2000
# 默认预算按“典型合法输入实际消耗（步数约几十、峰值项数个位数）”留出
# 两个数量级余量设定；再难的病态输入会在其中任一预算处及时熔断为 incomplete。
# 数值均可用关键字参数覆盖以处理更大的合法系统。
DEFAULT_MAX_TERMS = 300
DEFAULT_MAX_STEPS = 5000
# 公开的除法/核验原语不提供 incomplete 返回，按数学保证必终止；
# 仍保留一个极高的防御性项数上限，避免病态输入下无界占用内存。
UNBOUNDED_TERMS = 10**12


class _BudgetExhausted(Exception):
    """内部信号：约化中途触及预算。不对外抛出。

    ``kind`` 为 ``"terms"``（项数超限）或 ``"steps"``（初等约化步数超限）。
    """

    def __init__(self, what: str, terms: int, limit: int, kind: str = "terms"):
        super().__init__(what, terms, limit, kind)
        self.what = what
        self.terms = terms
        self.limit = limit
        self.kind = kind


# --------------------------------------------------------------------- 结果


@dataclass(frozen=True)
class DivisionResult:
    """多元多项式除法 ``f = sum_i quotients[i] * divisors[i] + remainder``。"""

    quotients: tuple[Poly, ...]
    remainder: Poly

    def identity(self, f: Poly, divisors: Sequence[Poly]) -> Poly:
        """按除法恒等式重算左边，供核验。"""
        total = Poly.zero(f.n)
        for q, g in zip(self.quotients, divisors):
            total = total + q * g
        return total + self.remainder


@dataclass(frozen=True)
class GroebnerResult:
    """Groebner 基计算结果。

    - ``status == "complete"``：``basis`` 是唯一的**首一约化** Groebner 基，
      ``reduced is True``，每个基多项式都带相对输入生成元的组合证书；
    - ``status == "incomplete"``：预算耗尽，``basis`` 是当前首一生成元集合
      （仍然张成同样的理想、证书仍然合法），但**不是**已核验的 Groebner 基，
      ``reduced is False``，``reason`` 说明触顶原因。
    """

    status: str  # "complete" | "incomplete"
    basis: tuple[Poly, ...]
    certificates: tuple[Certificate, ...]
    generators: tuple[Poly, ...]
    reduced: bool
    pairs_processed: int
    pairs_skipped_coprime: int
    reason: str
    pairs_skipped_lcm: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    def certificate_of(self, index: int) -> Certificate:
        return self.certificates[index]


@dataclass(frozen=True)
class MembershipResult:
    """理想成员判定结果。

    - ``"member"``：余式为 0，``certificate`` 给出
      ``f = sum_j certificate[j] * generators[j]`` 的组合证明（非空理想情形）；
    - ``"nonmember"``：Groebner 基完整且余式非零，余数作为非成员证明；
    - ``"inconclusive"``：预算在判定前耗尽，余式非零但不能据此下结论。
    """

    status: str  # "member" | "nonmember" | "inconclusive"
    remainder: Poly
    certificate: Optional[Certificate]
    groebner: GroebnerResult


# ----------------------------------------------------------------- 输入校验


def _check_budget(value: object, name: str) -> int:
    """预算参数校验：非负 int；bool 明确拒绝；0 是合法的零预算。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGeneratorError(f"{name} 必须是非负 int，得到 {value!r}")
    if value < 0:
        raise InvalidGeneratorError(f"{name} 必须是非负 int，得到 {value}")
    return value


def _validate_generators(generators: object) -> tuple[Poly, ...]:
    if isinstance(generators, Poly):
        raise InvalidGeneratorError(
            "generators 必须是 Poly 的序列（收到单个 Poly；请用 [p] 包裹）")
    if isinstance(generators, (str, bytes)) or not isinstance(generators, Iterable):
        raise InvalidGeneratorError(
            f"generators 必须是 Poly 的可迭代序列，得到 {type(generators).__name__}")
    gens: list[Poly] = []
    n: Optional[int] = None
    total_terms = 0
    for idx, g in enumerate(generators):
        if not isinstance(g, Poly):
            raise InvalidGeneratorError(
                f"第 {idx} 个生成元不是 Poly，得到 {type(g).__name__}: {g!r}")
        if n is None:
            n = g.n
        elif g.n != n:
            raise InvalidGeneratorError(
                f"第 {idx} 个生成元变量数 {g.n} 与前面的 {n} 不一致")
        total_terms += len(g)
        if total_terms > MAX_TOTAL_TERMS:
            raise InvalidGeneratorError(
                f"总输入项数超过上限 {MAX_TOTAL_TERMS}（在 index={idx} 处累计 "
                f"{total_terms}）；请精简输入")
        gens.append(g)
    if n is None:
        # 空生成元列表：变量数无从推断，按 1 变量的零理想处理需要 n；
        # 要求显式语义，故此处拒绝，引导使用 groebner_basis_of_zero(n)。
        raise InvalidGeneratorError(
            "生成元序列为空：零理想需要变量数，请改用 groebner_basis_of_zero(n)")
    return tuple(gens)


def _zero_certs(m: int, n: int) -> Certificate:
    return tuple(Poly.zero(n) for _ in range(m))


def _unit_cert(j: int, m: int, n: int, coef: Fraction) -> Certificate:
    cert = list(_zero_certs(m, n))
    if coef != 0:
        cert[j] = Poly.constant(coef, n)
    return tuple(cert)


def _cert_linear_map(cert: Certificate, mul: Poly) -> Certificate:
    """证书整体乘一个（单项式）多项式。"""
    return tuple((mul * c) for c in cert)


def _cert_sub(a: Certificate, b: Certificate) -> Certificate:
    return tuple(x - y for x, y in zip(a, b))


def _monomial_times(n: int, exp: tuple[int, ...], coef: Fraction) -> Poly:
    return Poly._from_dict({exp: coef}, n)


# --------------------------------------------------------------------- 除法


def _reduce(dividend: Poly,
            divisors: Sequence[Poly],
            n: int,
            max_terms: int,
            cert0: Optional[Certificate] = None,
            divisor_certs: Optional[Sequence[Certificate]] = None,
            steps_left: Optional[list[int]] = None,
            key=grlex_key,
            ) -> tuple[list[Poly], Poly, Optional[Certificate]]:
    """带可选证书跟踪的多元除法（grlex，按除式次序取第一个可整除者）。

    恒等式：``dividend = sum q_i d_i + remainder``；
    若提供 ``cert0``（dividend 的证书）与 ``divisor_certs``，
    则余数证书 = ``cert0 - sum q_i * divisor_certs[i]``。

    两层预算，任一触顶抛 :class:`_BudgetExhausted`：

    - ``max_terms``：中间被除式的非零项数上限（防单项多项式膨胀）；
    - ``steps_left``：跨多次调用共享的“初等约化步数”余量（单元素 list，
      原地递减）。项数限制无法约束**项数不多但步数极多**的病态约化，
    步数预算保证工作量总有上界，使任何输入都会在有限时间内得到
    complete 或 incomplete 之一。

    key 是单项式排序键（默认 grlex）；饱和计算的消元序会传入其他键。
    """
    s = len(divisors)
    div_lms = [max(d.terms, key=key) for d in divisors]
    div_lcs = [d.terms[e] for d, e in zip(divisors, div_lms)]
    quotients = [Poly.zero(n) for _ in range(s)]
    p = dividend
    remainder = Poly.zero(n)

    while not p.is_zero():
        if len(p) > max_terms:
            raise _BudgetExhausted("约化中间多项式", len(p), max_terms)
        if steps_left is not None:
            steps_left[0] -= 1
            if steps_left[0] < 0:
                raise _BudgetExhausted("初等约化", 0, 0, kind="steps")
        p_terms = p.terms
        lm_p = max(p_terms, key=key)
        lc_p = p_terms[lm_p]
        chosen = -1
        for i, lm_d in enumerate(div_lms):
            if all(a >= b for a, b in zip(lm_p, lm_d)):
                chosen = i
                break
        if chosen >= 0:
            d = divisors[chosen]
            exp = tuple(a - b for a, b in zip(lm_p, div_lms[chosen]))
            coef = lc_p / div_lcs[chosen]
            term = _monomial_times(n, exp, coef)
            quotients[chosen] = quotients[chosen] + term
            p = p - term * d
        else:
            # 首项移入余数
            lead = _monomial_times(n, lm_p, lc_p)
            remainder = remainder + lead
            p = p - lead

    cert_r: Optional[Certificate] = None
    if cert0 is not None:
        assert divisor_certs is not None and len(divisor_certs) == s
        cert_r = cert0
        for q, c in zip(quotients, divisor_certs):
            cert_r = _cert_sub(cert_r, _cert_linear_map(c, q))
    return quotients, remainder, cert_r


def divide(f: Poly, divisors: Sequence[Poly]) -> DivisionResult:
    """多元多项式除法（余数依赖除式排列顺序；非 Groebner 基时余数不唯一）。

    除式中出现零多项式时抛 :class:`~groebner.errors.ZeroDivisorError`
    （携带位置），变量数不一致抛 :class:`~groebner.errors.InvalidGeneratorError`。
    """
    if not isinstance(f, Poly):
        raise InvalidPolynomialError(f"被除式必须是 Poly，得到 {type(f).__name__}")
    if not isinstance(divisors, Sequence) or isinstance(divisors, (str, bytes)):
        raise InvalidGeneratorError("divisors 必须是 Poly 的序列（如 list/tuple）")
    clean: list[Poly] = []
    for idx, d in enumerate(divisors):
        if not isinstance(d, Poly):
            raise InvalidGeneratorError(
                f"第 {idx} 个除式不是 Poly，得到 {type(d).__name__}")
        if d.n != f.n:
            raise InvalidGeneratorError(
                f"第 {idx} 个除式变量数 {d.n} 与被除式 {f.n} 不一致")
        if d.is_zero():
            raise ZeroDivisorError(f"第 {idx} 个除式是零多项式，不能作除式")
        clean.append(d)
    q, r, _ = _reduce(f, clean, f.n, UNBOUNDED_TERMS)
    return DivisionResult(tuple(q), r)


# ------------------------------------------------------------------- S 多项式


def s_polynomial(f: Poly, g: Poly) -> Poly:
    """S 多项式 ``L/lm(f) * f/lc(f) - L/lm(g) * g/lc(g)``，``L = lcm(lm f, lm g)``。"""
    if f.is_zero() or g.is_zero():
        raise ZeroDivisorError("S 多项式要求两个多项式均非零")
    if f.n != g.n:
        raise InvalidGeneratorError(
            f"变量数不一致：{f.n} != {g.n}")
    lcm = lcm_monomial(f.lm(), g.lm())
    tf = _monomial_times(f.n,
                         tuple(a - b for a, b in zip(lcm, f.lm())),
                         Fraction(1, 1) / f.lc())
    tg = _monomial_times(g.n,
                         tuple(a - b for a, b in zip(lcm, g.lm())),
                         Fraction(1, 1) / g.lc())
    return tf * f - tg * g


def _s_with_cert(i: int, j: int, basis: Sequence[Poly],
                 certs: Sequence[Certificate],
                 lms: Sequence[tuple[int, ...]],
                 lcs: Sequence[Fraction]) -> tuple[Poly, Certificate]:
    f, g = basis[i], basis[j]
    lcm = lcm_monomial(lms[i], lms[j])
    tf = _monomial_times(f.n,
                         tuple(a - b for a, b in zip(lcm, lms[i])),
                         Fraction(1, 1) / lcs[i])
    tg = _monomial_times(g.n,
                         tuple(a - b for a, b in zip(lcm, lms[j])),
                         Fraction(1, 1) / lcs[j])
    s = tf * f - tg * g
    cert = _cert_sub(_cert_linear_map(certs[i], tf),
                     _cert_linear_map(certs[j], tg))
    return s, cert


def _coprime_lm(f: Poly, g: Poly) -> bool:
    """首项单项式是否“逐分量不同时为正”（首项互素判据的前提）。"""
    return all(not (a > 0 and b > 0) for a, b in zip(f.lm(), g.lm()))


# ----------------------------------------------------------- Buchberger 主体


def _buchberger_core(gens: tuple[Poly, ...],
                     n: int,
                     key,
                     max_pairs: int,
                     max_terms: int,
                     max_steps: int
                     ) -> tuple[str, list[Poly], list[Certificate],
                                int, int, int, str]:
    """Buchberger 主循环（单项式次序由排序键 key 给出）。

    返回 (status, basis, certs, processed, skipped_coprime, skipped_lcm, reason)。
    status 为 "complete" 时 basis 是该次序下唯一的首一约化 Groebner 基；
    为 "incomplete" 时 basis 只是当前首一生成元集合（证书仍合法）。
    输入假定已校验（均为 Poly、变量数一致、预算为非负 int）。
    """
    m = len(gens)
    # 全程只构造新列表，结束时一次性发布；中途任何失败都不会产生对外副作用。
    basis: list[Poly] = []
    basis_certs: list[Certificate] = []
    basis_lms: list[tuple[int, ...]] = []
    basis_lcs: list[Fraction] = []
    # 待处理关键对集合（无序对，i < j）；配合正常选择策略使用。
    pair_set: set[tuple[int, int]] = set()
    skipped_coprime = 0
    skipped_lcm = 0
    processed = 0
    steps_left = [max_steps]

    def budget_message(exc: _BudgetExhausted, where: str) -> str:
        if exc.kind == "steps":
            used = max_steps - steps_left[0]
            return (f"初等约化步数预算耗尽：累计约 {used} 步"
                    f"（max_steps={max_steps}），{where}")
        return (f"项数预算耗尽：{exc.what}出现 {exc.terms} 个非零项 "
                f"（max_terms={exc.limit}），{where}")

    def add_element(poly: Poly, cert: Certificate) -> None:
        # 入库前先按当前基约化，再统一首一化（只做一次）。
        if poly.is_zero():
            return
        _, r, cert_r = _reduce(poly, basis, n, max_terms, cert,
                               basis_certs, steps_left, key)
        if r.is_zero():
            return
        inv = Fraction(1, 1) / r.lc()
        r = r.scale(inv)
        cert_r = tuple(c.scale(inv) for c in cert_r)
        k = len(basis)
        lm_r = max(r.terms, key=key)
        basis.append(r)
        basis_certs.append(cert_r)
        basis_lms.append(lm_r)
        basis_lcs.append(r.terms[lm_r])
        for i in range(k):
            pair_set.add((i, k))

    def pick_pair() -> tuple[int, int]:
        """正常选择策略：取 lcm(lm) 在当前次序下最小的关键对。"""
        def pkey(p: tuple[int, int]) -> tuple[int, tuple[int, ...]]:
            l = lcm_monomial(basis_lms[p[0]], basis_lms[p[1]])
            return key(l)
        return min(pair_set, key=pkey)

    def lcm_criterion(i: int, j: int) -> bool:
        """Buchberger 第二判据（lcm 判据）。

        若存在第三个当前基元素 k，其首项整除 lcm(lm_i, lm_j)，
        且 (i,k)、(j,k) 都已不在待处理集合（此前已处理/已剔除），
        则 S(i,j) 必约化为 0，可安全跳过。
        """
        lcm = lcm_monomial(basis_lms[i], basis_lms[j])
        for k in range(len(basis)):
            if k == i or k == j:
                continue
            lm_k = basis_lms[k]
            if all(a >= b for a, b in zip(lcm, lm_k)):
                ik = (i, k) if i < k else (k, i)
                jk = (j, k) if j < k else (k, j)
                if ik not in pair_set and jk not in pair_set:
                    return True
        return False

    def incomplete(what: str):
        return ("incomplete", basis, basis_certs,
                processed, skipped_coprime, skipped_lcm, what)

    try:
        for idx, g in enumerate(gens):
            if g.is_zero():
                continue
            add_element(g, _unit_cert(idx, m, n, Fraction(1)))

        while pair_set:
            i, j = pick_pair()
            pair_set.remove((i, j))
            processed += 1
            if processed > max_pairs:
                return incomplete(
                    f"关键对预算耗尽：预算允许处理 {max_pairs} 对，"
                    f"第 {processed} 对 ({i},{j}) 无法处理"
                    f"（此外仍有 {len(pair_set)} 个待处理关键对）")
            if all(not (a > 0 and b > 0)
                   for a, b in zip(basis_lms[i], basis_lms[j])):
                # Buchberger 第一判据：首项单项式互素 ⇒ S 多项式必约为 0。
                skipped_coprime += 1
                continue
            if lcm_criterion(i, j):
                # Buchberger 第二判据：lcm 被已处理的第三元素覆盖。
                skipped_lcm += 1
                continue
            s, cert_s = _s_with_cert(i, j, basis, basis_certs,
                                     basis_lms, basis_lcs)
            try:
                _, r, cert_r = _reduce(s, basis, n, max_terms,
                                       cert_s, basis_certs, steps_left, key)
            except _BudgetExhausted as exc:
                return incomplete(budget_message(exc, f"关键对 ({i},{j}) 未完成约化"))
            if not r.is_zero():
                add_element(r, cert_r)  # type: ignore[arg-type]
    except _BudgetExhausted as exc:
        # 初始生成元入库阶段触顶同样按 incomplete 返回。
        return incomplete(budget_message(exc, "生成元入库阶段"))

    try:
        reduced, reduced_certs = _reduce_to_reduced(
            basis, basis_certs, n, max_terms, steps_left, key)
    except _BudgetExhausted as exc:
        return incomplete(budget_message(exc, "最终约化未完成"))
    return ("complete", reduced, reduced_certs,
            processed, skipped_coprime, skipped_lcm, "")


def groebner_basis(generators: Iterable[Poly],
                   *,
                   max_pairs: int = DEFAULT_MAX_PAIRS,
                   max_terms: int = DEFAULT_MAX_TERMS,
                   max_steps: int = DEFAULT_MAX_STEPS) -> GroebnerResult:
    """计算首一约化 Groebner 基（固定 grlex 次序）。

    零生成元会被忽略（但在证书中保留其输入位置）；重复生成元、不同排列顺序
    均产生同一个首一约化基。三个预算任一耗尽都返回 incomplete：

    - max_pairs：处理的关键对数（含跳过的对）；
    - max_terms：约化中间多项式的非零项数；
    - max_steps：全部初等约化步数之和（含最终约化与成员判定的取余），
      用于约束“项数不多但步数极多”的病态约化，保证总有有限运行时间上界。
    """
    max_pairs = _check_budget(max_pairs, "max_pairs")
    max_terms = _check_budget(max_terms, "max_terms")
    max_steps = _check_budget(max_steps, "max_steps")
    gens = _validate_generators(generators)
    status, basis, certs, processed, sk_coprime, sk_lcm, reason = \
        _buchberger_core(gens, gens[0].n, grlex_key,
                         max_pairs, max_terms, max_steps)
    return GroebnerResult(
        status, tuple(basis), tuple(certs), gens,
        status == "complete", processed, sk_coprime, reason, sk_lcm)


def groebner_basis_of_zero(n: int,
                           *,
                           max_pairs: int = DEFAULT_MAX_PAIRS,
                           max_terms: int = DEFAULT_MAX_TERMS,
                           max_steps: int = DEFAULT_MAX_STEPS
                           ) -> GroebnerResult:
    """零理想（无生成元）的 Groebner 基：空基。

    预算参数仅为接口一致而校验；空基不做任何约化工作。
    """
    _check_budget(max_pairs, "max_pairs")
    _check_budget(max_terms, "max_terms")
    _check_budget(max_steps, "max_steps")
    if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= MAX_VARIABLES):
        raise InvalidPolynomialError(
            f"变量数 n 必须是 1..{MAX_VARIABLES} 的 int，得到 {n!r}")
    return GroebnerResult("complete", (), (), (), True, 0, 0, "")


def _reduce_to_reduced(basis: Sequence[Poly],
                       certs: Sequence[Certificate],
                       n: int,
                       max_terms: int,
                       steps_left: Optional[list[int]] = None,
                       key=grlex_key
                       ) -> tuple[list[Poly], list[Certificate]]:
    """把 Buchberger 基化为唯一的首一约化基，证书同步变换。

    步骤：删掉首项单项式被其他元素首项整除的冗余元素（极小化），
    再让每个元素对其余元素取余式；极小首一基保证余式首项不变、仍首一。
    """
    kept = list(range(len(basis)))
    changed = True
    while changed:
        changed = False
        for pos, idx in enumerate(kept):
            others = kept[:pos] + kept[pos + 1:]
            lms = {i: max(basis[i].terms, key=key) for i in kept}
            if any(all(a >= b for a, b in zip(lms[idx], lms[k]))
                   for k in others):
                kept.remove(idx)
                changed = True
                break

    reduced = [basis[i] for i in kept]
    reduced_certs = [certs[i] for i in kept]
    # 反复让每个元素对其余元素取余式，直到整体不动点。
    # 极小基保证每个元素的首项（及首一系数）不被触动，只化简尾部，
    # 而项序是良序，故每趟严格下降、过程必终止。
    changed = True
    while changed:
        changed = False
        for pos in range(len(reduced)):
            divisors = reduced[:pos] + reduced[pos + 1:]
            dcerts = reduced_certs[:pos] + reduced_certs[pos + 1:]
            _, r, cert_r = _reduce(reduced[pos], divisors, n, max_terms,
                                   reduced_certs[pos], dcerts, steps_left, key)
            if r != reduced[pos]:
                inv = Fraction(1, 1) / r.lc()
                reduced[pos] = r.scale(inv)
                reduced_certs[pos] = tuple(c.scale(inv) for c in cert_r)  # type: ignore[union-attr]
                changed = True

    order = sorted(range(len(reduced)),
                   key=lambda k: key(max(reduced[k].terms, key=key)),
                   reverse=True)
    return [reduced[k] for k in order], [reduced_certs[k] for k in order]


# --------------------------------------------------------------- 基的核验 API


def check_s_polynomials(basis: Sequence[Poly]) -> list[tuple[int, int, Poly]]:
    """对基中每一对计算 S 多项式并约化，返回所有**非零余式**。

    返回空列表当且仅当该集合是 Groebner 基（Buchberger 判据）。
    约化使用基自身（含每个元素自身在内不影响 S 多项式首项）。
    """
    for idx, g in enumerate(basis):
        if not isinstance(g, Poly):
            raise InvalidGeneratorError(f"第 {idx} 个基元素不是 Poly")
        if g.is_zero():
            raise InvalidGeneratorError(f"第 {idx} 个基元素是零多项式")
        if idx and g.n != basis[0].n:
            raise InvalidGeneratorError("基元素变量数不一致")
    nonzero: list[tuple[int, int, Poly]] = []
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            s = s_polynomial(basis[i], basis[j])
            _, r, _ = _reduce(s, list(basis), basis[0].n, UNBOUNDED_TERMS)
            if not r.is_zero():
                nonzero.append((i, j, r))
    return nonzero


def verify_groebner_basis(basis: Sequence[Poly]) -> bool:
    """便捷判据：所有最终 S 多项式余式为 0。"""
    return not check_s_polynomials(basis)


# ------------------------------------------------------------- 理想成员判定


def membership(f: Poly,
               generators: Iterable[Poly],
               *,
               max_pairs: int = DEFAULT_MAX_PAIRS,
               max_terms: int = DEFAULT_MAX_TERMS,
               max_steps: int = DEFAULT_MAX_STEPS) -> MembershipResult:
    """判定 ``f`` 是否属于生成元生成的理想。

    - 余式 0 总是成员的严格证明，直接给出组合证书；
    - 余式非零只有在 Groebner 基完整时才证明非成员；
    - 预算耗尽且余式非零时返回 ``"inconclusive"``。

    基构造与最后对 ``f`` 取余各至多消耗 ``max_steps`` 个初等约化步
    （总计有界），保证任何输入都会在有限时间内得到三种状态之一。
    """
    if not isinstance(f, Poly):
        raise InvalidPolynomialError(f"被判定多项式必须是 Poly，得到 {type(f).__name__}")
    max_pairs = _check_budget(max_pairs, "max_pairs")
    max_terms = _check_budget(max_terms, "max_terms")
    max_steps = _check_budget(max_steps, "max_steps")
    gens = _validate_generators(generators)
    if f.n != gens[0].n:
        raise InvalidGeneratorError(
            f"被判定多项式变量数 {f.n} 与生成元 {gens[0].n} 不一致")
    result = groebner_basis(gens, max_pairs=max_pairs,
                            max_terms=max_terms, max_steps=max_steps)
    n = f.n
    m = len(result.generators)

    if not result.basis:
        # 全部生成元为零：零理想，仅零多项式属于它。
        if f.is_zero():
            return MembershipResult(
                "member", Poly.zero(n), _zero_certs(m, n), result)
        status = "nonmember" if result.is_complete else "inconclusive"
        return MembershipResult(status, f, None, result)

    try:
        q, r, _ = _reduce(f, list(result.basis), n, max_terms,
                          steps_left=[max_steps])
    except _BudgetExhausted as exc:
        reason = result.reason or ""
        if exc.kind == "steps":
            tail = f"成员判定取余初等约化步数预算耗尽（max_steps={max_steps}）"
        else:
            tail = (f"成员判定取余项数预算耗尽：{exc.what}出现 {exc.terms} "
                    f"个非零项（max_terms={exc.limit}）")
        base = result if result.is_complete else GroebnerResult(
            "incomplete", result.basis, result.certificates,
            result.generators, False, result.pairs_processed,
            result.pairs_skipped_coprime,
            (reason + "；" if reason else "") + tail,
            result.pairs_skipped_lcm)
        return MembershipResult("inconclusive", f, None, base)

    if r.is_zero():
        # f = sum_i q_i * b_i，而 b_i = sum_j cert_i[j] * g_j，
        # 故 f 对原生成元的证书为 cert[j] = sum_i q_i * cert_i[j]。
        cert = list(_zero_certs(m, n))
        for qi, bi_cert in zip(q, result.certificates):
            for k in range(m):
                cert[k] = cert[k] + qi * bi_cert[k]
        return MembershipResult("member", r, tuple(cert), result)

    status = "nonmember" if result.is_complete else "inconclusive"
    return MembershipResult(status, r, None, result)


def membership_in_zero_ideal(f: Poly) -> MembershipResult:
    """对零理想（``groebner_basis_of_zero``）做成员判定的便捷入口。"""
    if not isinstance(f, Poly):
        raise InvalidPolynomialError(f"被判定多项式必须是 Poly，得到 {type(f).__name__}")
    gb = GroebnerResult("complete", (), (), (), True, 0, 0, "")
    if f.is_zero():
        return MembershipResult("member", Poly.zero(f.n), (), gb)
    return MembershipResult("nonmember", f, None, gb)


# ----------------------------------------------------------------- 理想饱和


@dataclass(frozen=True)
class SaturationResult:
    """饱和理想 I:f^inf = {g : 存在 k 使 f**k * g 属于 I} 的计算结果。

    - status == "complete"：basis 是饱和理想唯一的首一约化 Groebner 基
      （grlex，变量数与原理想相同，辅助变量 t 不会泄漏）；
      powers[i] 与 certificates[i] 给出逐基多项式的成员证书：
      f**powers[i] * basis[i] = sum_j certificates[i][j] * generators[j]，
      可用多项式加法/乘法独立展开核验；
    - status == "complete" 但 certificates 为 None：基本身有效，
      仅证书因项数超过 max_terms 预算而未给出（reason 说明）；
    - status == "incomplete"：预算耗尽，basis 为空，不声明任何结论。
    """

    status: str  # "complete" | "incomplete"
    basis: tuple[Poly, ...]
    certificates: Optional[tuple[Certificate, ...]]
    powers: Optional[tuple[int, ...]]
    generators: tuple[Poly, ...]
    f: Poly
    reduced: bool
    reason: str


def _elim_key_last(exp: tuple[int, ...]):
    """末位变量（辅助变量 t）最大的消元序：先比 t 的次数，再按 grlex 比其余。

    这是合法的消元序：任何含 t 的单项式都大于所有不含 t 的单项式，
    因此该次序下约化基中首项不含 t 的元素全体落在原多项式环中，
    且构成消元理想（即饱和理想）的 Groebner 基。
    """
    return (exp[-1], grlex_key(exp[:-1]))


def _lift_last(p: Poly) -> Poly:
    """把 n 元多项式嵌入 n+1 元环（新变量在末位，指数为 0）。"""
    return Poly._from_dict({e + (0,): c for e, c in p.terms.items()}, p.n + 1)


def _drop_last(p: Poly) -> Poly:
    """把不含末位变量的 n+1 元多项式降回 n 元环。"""
    return Poly._from_dict({e[:-1]: c for e, c in p.terms.items()}, p.n - 1)


def _pow_poly(base: Poly, k: int) -> Poly:
    out = Poly.constant(1, base.n)
    for _ in range(k):
        out = out * base
    return out


def _clear_t(cert: Poly, f: Poly, k: int, n: int) -> Poly:
    """把证书多项式 c(x, t) 变为 sum_a f**(k-a) * c_a(x)（c = sum_a t**a c_a）。

    由 1 - t*f = 0 即 t*f = 1，f**k * c(x, t) 模 (1 - t*f) 等于上式，
    且两边都不含 t，而 (1 - t*f) 的倍式中只有 0 不含 t，故等式精确成立。
    """
    slices: dict[int, dict[tuple[int, ...], Fraction]] = {}
    for e, c in cert.terms.items():
        slices.setdefault(e[-1], {})[e[:-1]] = c
    out = Poly.zero(n)
    for a, d in slices.items():
        out = out + _pow_poly(f, k - a) * Poly._from_dict(d, n)
    return out


def saturate(generators: Iterable[Poly],
             f: Poly,
             *,
             max_pairs: int = DEFAULT_MAX_PAIRS,
             max_terms: int = DEFAULT_MAX_TERMS,
             max_steps: int = DEFAULT_MAX_STEPS) -> SaturationResult:
    """计算饱和理想 I:f^inf（Rabinowitsch 技巧 + 消元序），I 由生成元给出。

    方法：在加入辅助变量 t 的扩环中计算 J = I + (1 - t*f) 关于消元序
    （t 最大，其余变量间仍为 grlex）的首一约化 Groebner 基，取其中不含 t
    的元素，再整理为原环上 grlex 的首一约化基。这不是一次理想商：
    例如 (x*y):x 的商是 (y)，而 (x**2*y):x 的商是 (x*y)，饱和才是 (y)。

    明确边界：

    - f 为零多项式：I:0^inf = (1)（排除 f=0 即排除整个空间）；
    - f 为非零常数：常数是单位，c**k * g 属于 I 当且仅当 g 属于 I，
      故饱和等于原理想 I 本身；
    - 零理想（生成元全为零或为空序列）：被非零 f 饱和仍是零理想；
    - 单位理想：饱和仍是单位理想。

    变量数上限：原理想至多 MAX_VARIABLES - 1 个变量（辅助变量 t 占一位）。
    预算语义与 groebner_basis 相同；任一阶段耗尽返回 incomplete。
    """
    max_pairs = _check_budget(max_pairs, "max_pairs")
    max_terms = _check_budget(max_terms, "max_terms")
    max_steps = _check_budget(max_steps, "max_steps")
    if not isinstance(f, Poly):
        raise InvalidPolynomialError(
            f"饱和多项式 f 必须是 Poly，得到 {type(f).__name__}")
    if isinstance(generators, Poly):
        raise InvalidGeneratorError(
            "generators 必须是 Poly 的序列（收到单个 Poly；请用 [p] 包裹）")
    if isinstance(generators, (str, bytes)) or not isinstance(generators, Iterable):
        raise InvalidGeneratorError(
            f"generators 必须是 Poly 的可迭代序列，得到 {type(generators).__name__}")
    gens: list[Poly] = []
    # 扩环系统 I + (1 - t*f) 的输入项数 = 原生成元项数 + len(f) + 1。
    total_terms = len(f) + 1
    for idx, g in enumerate(generators):
        if not isinstance(g, Poly):
            raise InvalidGeneratorError(
                f"第 {idx} 个生成元不是 Poly，得到 {type(g).__name__}: {g!r}")
        if g.n != f.n:
            raise InvalidGeneratorError(
                f"第 {idx} 个生成元变量数 {g.n} 与 f 的 {f.n} 不一致")
        total_terms += len(g)
        if total_terms > MAX_TOTAL_TERMS:
            raise InvalidGeneratorError(
                f"总输入项数（含 1 - t*f）超过上限 {MAX_TOTAL_TERMS}（在 index={idx} "
                f"处累计 {total_terms}）；请精简输入")
        gens.append(g)
    gens_t = tuple(gens)
    n = f.n
    m = len(gens_t)

    # f = 0：I:0^inf = (1)。证书取 k=1：f**1 * 1 = 0 恒属于任何理想。
    if f.is_zero():
        return SaturationResult("complete", (Poly.constant(1, n),),
                                (_zero_certs(m, n),), (1,), gens_t, f, True, "")
    # 零理想被非零 f 饱和仍是零理想（(1 - t*f) 与原环的交只有 0）。
    if all(g.is_zero() for g in gens_t):
        return SaturationResult("complete", (), (), (), gens_t, f, True, "")
    if n + 1 > MAX_VARIABLES:
        raise InvalidGeneratorError(
            f"饱和需要引入辅助变量 t：{n} + 1 超过变量数上限 {MAX_VARIABLES}")

    n_ext = n + 1
    t_var = Poly.variable(n_ext, n)
    one_minus_tf = Poly.constant(1, n_ext) - t_var * _lift_last(f)
    ext_gens = tuple(_lift_last(g) for g in gens_t) + (one_minus_tf,)
    status, basis_e, certs_e, _, _, _, reason = _buchberger_core(
        ext_gens, n_ext, _elim_key_last, max_pairs, max_terms, max_steps)
    if status != "complete":
        return SaturationResult("incomplete", (), None, None,
                                gens_t, f, False, reason)

    # 消元：该次序下首项不含 t 的基元素整体落在原环中。
    free: list[Poly] = []
    free_certs: list[Certificate] = []
    for b, c in zip(basis_e, certs_e):
        if max(b.terms, key=_elim_key_last)[-1] == 0:
            free.append(_drop_last(b))
            free_certs.append(c)
    if not free:
        return SaturationResult("complete", (), (), (), gens_t, f, True, "")

    # 整理为原环 grlex 下唯一的首一约化基（证书相对消元出的生成元）。
    gb = groebner_basis(free, max_pairs=max_pairs,
                        max_terms=max_terms, max_steps=max_steps)
    if not gb.is_complete:
        return SaturationResult("incomplete", (), None, None,
                                gens_t, f, False, gb.reason)

    # 组合证书：b = sum_i combo_i * h_i，h_i = sum_j ecert_ij * ext_j，
    # 故 b = sum_j c_j(x, t) ext_j；再按 t -> 1/f 清分母得到
    # f**k * b = sum_j cert_j(x) * g_j（1 - t*f 分量自动消去）。
    all_certs: list[Certificate] = []
    all_powers: list[int] = []
    for combo in gb.certificates:
        cert_ext = [Poly.zero(n_ext) for _ in ext_gens]
        for coef, ecert in zip(combo, free_certs):
            for j, c in enumerate(ecert):
                cert_ext[j] = cert_ext[j] + _lift_last(coef) * c
        k = 0
        for c in cert_ext[:m]:
            for e in c.terms:
                if e[-1] > k:
                    k = e[-1]
        cert_orig = list(_zero_certs(m, n))
        overflow = False
        for j in range(m):
            cert_orig[j] = _clear_t(cert_ext[j], f, k, n)
            if len(cert_orig[j]) > max_terms:
                overflow = True
        if overflow:
            return SaturationResult(
                "complete", gb.basis, None, None, gens_t, f, True,
                f"证书项数超过 max_terms={max_terms} 的预算，仅返回基本身")
        all_certs.append(tuple(cert_orig))
        all_powers.append(k)
    return SaturationResult("complete", gb.basis, tuple(all_certs),
                            tuple(all_powers), gens_t, f, True, "")
