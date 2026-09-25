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
    ZeroPolynomialError,
)
from .poly import (
    MAX_TOTAL_TERMS,
    MAX_VARIABLES,
    Poly,
    grlex_key,
    lcm_monomial,
)

#: 组合证书：长度等于输入生成元个数（含零生成元的位置），
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


@dataclass(frozen=True)
class IntersectionResult:
    """两个有限生成理想的交集 I 与 J 之交的计算结果。

    - status == "complete"：basis 是交集理想关于原变量 grlex 次序的
      一组首一 Groebner 基（不保证是约化基，故 reduced 为 False），
      每个基多项式同时带有相对 generators_i 与 generators_j
      的两份组合证书；
    - status == "incomplete"：消元过程预算耗尽。basis 只含已发现的
      不含消元变量的元素（每个都确实属于交集、证书合法），但不声明
      它们生成了整个交集。

    certificates_i[k] 满足 basis[k] = sum_j certificates_i[k][j] * I_j
    （原变量环中的恒等式，扩展证书已取 t=1）；certificates_j[k] 同理（t=0）。
    extended 是扩展环上 t*I + (1-t)*J 的 Groebner 基计算结果，供核查。
    """

    status: str  # "complete" | "incomplete"
    basis: tuple[Poly, ...]
    certificates_i: tuple[Certificate, ...]
    certificates_j: tuple[Certificate, ...]
    generators_i: tuple[Poly, ...]
    generators_j: tuple[Poly, ...]
    reduced: bool
    reason: str
    extended: GroebnerResult

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


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


def _lm_with(poly: Poly, key) -> tuple[int, ...]:
    """按给定单项式排序键取首项单项式（公开接口固定用 grlex_key）。"""
    if poly.is_zero():
        raise ZeroPolynomialError("零多项式没有首项单项式")
    return max(poly.terms, key=key)


def _lc_with(poly: Poly, key) -> Fraction:
    return poly.monomial_coefficient(_lm_with(poly, key))


def _elimination_key(exp: tuple[int, ...]) -> tuple:
    """消元块次序：消元变量 t（第 0 分量）大于一切原变量单项式，原变量块内 grlex。

    仅在 ideal_intersection 内部使用；公开的 grlex 接口不受影响。
    若某多项式的首项不含 t，则块次序保证它所有项都不含 t（消元性质）。
    """
    return (exp[0],) + grlex_key(exp[1:])


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
    """
    s = len(divisors)
    quotients = [Poly.zero(n) for _ in range(s)]
    p = dividend
    remainder = Poly.zero(n)
    div_lm = [_lm_with(d, key) for d in divisors]
    div_lc = [d.monomial_coefficient(e) for d, e in zip(divisors, div_lm)]

    while not p.is_zero():
        if len(p) > max_terms:
            raise _BudgetExhausted("约化中间多项式", len(p), max_terms)
        if steps_left is not None:
            steps_left[0] -= 1
            if steps_left[0] < 0:
                raise _BudgetExhausted("初等约化", 0, 0, kind="steps")
        lm_p = _lm_with(p, key)
        lc_p = p.monomial_coefficient(lm_p)
        chosen = -1
        for i in range(s):
            if all(a >= b for a, b in zip(lm_p, div_lm[i])):
                chosen = i
                break
        if chosen >= 0:
            d = divisors[chosen]
            exp = tuple(a - b for a, b in zip(lm_p, div_lm[chosen]))
            coef = lc_p / div_lc[chosen]
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
                 key=grlex_key) -> tuple[Poly, Certificate]:
    f, g = basis[i], basis[j]
    flm, glm = _lm_with(f, key), _lm_with(g, key)
    lcm = lcm_monomial(flm, glm)
    tf = _monomial_times(f.n,
                         tuple(a - b for a, b in zip(lcm, flm)),
                         Fraction(1, 1) / f.monomial_coefficient(flm))
    tg = _monomial_times(g.n,
                         tuple(a - b for a, b in zip(lcm, glm)),
                         Fraction(1, 1) / g.monomial_coefficient(glm))
    s = tf * f - tg * g
    cert = _cert_sub(_cert_linear_map(certs[i], tf),
                     _cert_linear_map(certs[j], tg))
    return s, cert


def _coprime_lm(f: Poly, g: Poly, key=grlex_key) -> bool:
    """首项单项式是否“逐分量不同时为正”（首项互素判据的前提）。"""
    return all(not (a > 0 and b > 0)
               for a, b in zip(_lm_with(f, key), _lm_with(g, key)))


# ----------------------------------------------------------- Buchberger 主体


def groebner_basis(generators: Iterable[Poly],
                   *,
                   max_pairs: int = DEFAULT_MAX_PAIRS,
                   max_terms: int = DEFAULT_MAX_TERMS,
                   max_steps: int = DEFAULT_MAX_STEPS,
                   _order_key=grlex_key) -> GroebnerResult:
    """计算首一约化 Groebner 基（固定 grlex 次序）。

    零生成元会被忽略（但在证书中保留其输入位置）；重复生成元、不同排列顺序
    均产生同一个首一约化基。三个预算任一耗尽都返回 ``status="incomplete"``：

    - ``max_pairs``：处理的关键对数（含跳过的对）；
    - ``max_terms``：约化中间多项式的非零项数；
    - ``max_steps``：全部初等约化步数之和（含最终约化与成员判定的取余），
      用于约束“项数不多但步数极多”的病态约化，保证总有有限运行时间上界。
    """
    max_pairs = _check_budget(max_pairs, "max_pairs")
    max_terms = _check_budget(max_terms, "max_terms")
    max_steps = _check_budget(max_steps, "max_steps")
    gens = _validate_generators(generators)
    n = gens[0].n
    m = len(gens)

    # 全程只构造新列表，结束时一次性发布；中途任何失败都不会产生对外副作用。
    basis: list[Poly] = []
    basis_certs: list[Certificate] = []
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
                               basis_certs, steps_left, _order_key)
        if r.is_zero():
            return
        inv = Fraction(1, 1) / r.lc()
        r = r.scale(inv)
        cert_r = tuple(c.scale(inv) for c in cert_r)
        k = len(basis)
        basis.append(r)
        basis_certs.append(cert_r)
        for i in range(k):
            pair_set.add((i, k))

    def pick_pair() -> tuple[int, int]:
        """正常选择策略：取 lcm(lm) 在 grlex 下最小的关键对。"""
        def pkey(p: tuple[int, int]) -> tuple[int, tuple[int, ...]]:
            l = lcm_monomial(_lm_with(basis[p[0]], _order_key),
                             _lm_with(basis[p[1]], _order_key))
            return _order_key(l)
        return min(pair_set, key=pkey)

    def lcm_criterion(i: int, j: int) -> bool:
        """Buchberger 第二判据（lcm 判据）。

        若存在第三个当前基元素 k，其首项整除 lcm(lm_i, lm_j)，
        且 (i,k)、(j,k) 都已不在待处理集合（此前已处理/已剔除），
        则 S(i,j) 必约化为 0，可安全跳过。
        """
        lcm = lcm_monomial(_lm_with(basis[i], _order_key),
                           _lm_with(basis[j], _order_key))
        for k in range(len(basis)):
            if k == i or k == j:
                continue
            lm_k = _lm_with(basis[k], _order_key)
            if all(a >= b for a, b in zip(lcm, lm_k)):
                ik = (i, k) if i < k else (k, i)
                jk = (j, k) if j < k else (k, j)
                if ik not in pair_set and jk not in pair_set:
                    return True
        return False

    def incomplete(what: str) -> GroebnerResult:
        return GroebnerResult(
            "incomplete", tuple(basis), tuple(basis_certs), gens,
            False, processed, skipped_coprime, what, skipped_lcm)

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
            if _coprime_lm(basis[i], basis[j], _order_key):
                # Buchberger 第一判据：首项单项式互素 ⇒ S 多项式必约为 0。
                skipped_coprime += 1
                continue
            if lcm_criterion(i, j):
                # Buchberger 第二判据：lcm 被已处理的第三元素覆盖。
                skipped_lcm += 1
                continue
            s, cert_s = _s_with_cert(i, j, basis, basis_certs, _order_key)
            try:
                _, r, cert_r = _reduce(s, basis, n, max_terms,
                                       cert_s, basis_certs, steps_left,
                                       _order_key)
            except _BudgetExhausted as exc:
                return incomplete(budget_message(exc, f"关键对 ({i},{j}) 未完成约化"))
            if not r.is_zero():
                add_element(r, cert_r)  # type: ignore[arg-type]
    except _BudgetExhausted as exc:
        # 初始生成元入库阶段触顶同样按 incomplete 返回。
        return incomplete(budget_message(exc, "生成元入库阶段"))

    skipped = skipped_coprime
    try:
        reduced, reduced_certs = _reduce_to_reduced(
            basis, basis_certs, n, max_terms, steps_left, _order_key)
    except _BudgetExhausted as exc:
        return incomplete(budget_message(exc, "最终约化未完成"))
    return GroebnerResult(
        "complete", tuple(reduced), tuple(reduced_certs), gens,
        True, processed, skipped, "", skipped_lcm)


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
            if any(all(a >= b for a, b in zip(_lm_with(basis[idx], key),
                                              _lm_with(basis[k], key)))
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
                   key=lambda k: key(_lm_with(reduced[k], key)), reverse=True)
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


# ------------------------------------------------------------- 理想交集


def _validate_intersection_side(gens: object, name: str) -> tuple[Poly, ...]:
    """交集某一侧的生成元校验；与 _validate_generators 类似但允许空序列。"""
    if isinstance(gens, Poly):
        raise InvalidGeneratorError(
            f"{name} 必须是 Poly 的序列（收到单个 Poly；请用 [p] 包裹）")
    if isinstance(gens, (str, bytes)) or not isinstance(gens, Iterable):
        raise InvalidGeneratorError(
            f"{name} 必须是 Poly 的可迭代序列，得到 {type(gens).__name__}")
    out: list[Poly] = []
    for idx, g in enumerate(gens):
        if not isinstance(g, Poly):
            raise InvalidGeneratorError(
                f"{name} 第 {idx} 个生成元不是 Poly，"
                f"得到 {type(g).__name__}: {g!r}")
        out.append(g)
    return tuple(out)


def _lift_t0(poly: Poly) -> Poly:
    """把 n 变量多项式嵌入 n+1 变量环（消元变量 t 为第 0 分量，取 t^0）。"""
    return Poly._from_dict({(0,) + e: c for e, c in poly.terms.items()},
                           poly.n + 1)


def _lift_t1(poly: Poly) -> Poly:
    """t * poly：嵌入 n+1 变量环并乘上消元变量 t。"""
    return Poly._from_dict({(1,) + e: c for e, c in poly.terms.items()},
                           poly.n + 1)


def _drop_t(poly: Poly) -> Poly:
    """去掉第 0 分量（要求多项式本就不含 t；调用方保证）。"""
    return Poly._from_dict({e[1:]: c for e, c in poly.terms.items()},
                           poly.n - 1)


def _eval_t_one(poly: Poly) -> Poly:
    """扩展环证书多项式取 t=1：同一原变量单项式的各 t 次幂系数求和。"""
    out: dict[tuple[int, ...], Fraction] = {}
    for e, c in poly.terms.items():
        x = e[1:]
        out[x] = out.get(x, Fraction(0)) + c
    return Poly._from_dict(out, poly.n - 1)


def _eval_t_zero(poly: Poly) -> Poly:
    """扩展环证书多项式取 t=0：只保留不含 t 的项。"""
    return Poly._from_dict({e[1:]: c for e, c in poly.terms.items()
                            if e[0] == 0}, poly.n - 1)


def ideal_intersection(gens_i, gens_j, *, max_pairs=DEFAULT_MAX_PAIRS,
                       max_terms=DEFAULT_MAX_TERMS,
                       max_steps=DEFAULT_MAX_STEPS) -> IntersectionResult:
    """计算两个有限生成理想的交集 I ∩ J 的一组生成元及双向组合证书。

    方法（消元定理）：引入额外消元变量 t，在 n+1 变量环中构造
    K = t*I + (1-t)*J，用“t 大于一切原变量、原变量块内 grlex”的
    块次序求 K 的 Groebner 基，则其中不含 t 的元素即生成交集，
    且其限制到原变量的次序恰为原内核的 grlex。块次序仅在本入口内部
    使用，公开的 grlex 接口行为不变。

    证书：扩展环中每个基多项式带证书
    b = sum_j A_j*(t*I_j) + sum_k C_k*((1-t)*J_k)；因 b 不含 t，
    取 t=1 得 b = sum_j A_j(1,x)*I_j，取 t=0 得
    b = sum_k C_k(0,x)*J_k——两份证书都在原变量环中，
    可直接独立展开核验。

    限制：原变量数 n 须满足 1 <= n <= 3（加 t 后不超过内核的 4 变量上限）；
    扩展生成元总项数同样受 MAX_TOTAL_TERMS 约束。三个预算覆盖整个
    消元过程（构造扩展基 + 最终约化）；任一耗尽时返回
    status="incomplete"，此时 basis 只含已发现的交元素，
    不声明它生成了整个交集。

    零理想（空序列或全零生成元）与单位理想按数学定义处理：
    与零理想的交是零理想（空基），与单位理想的交是另一侧本身。
    """
    max_pairs = _check_budget(max_pairs, "max_pairs")
    max_terms = _check_budget(max_terms, "max_terms")
    max_steps = _check_budget(max_steps, "max_steps")
    gens_i_t = _validate_intersection_side(gens_i, "I 的生成元")
    gens_j_t = _validate_intersection_side(gens_j, "J 的生成元")

    n: Optional[int] = None
    for g in gens_i_t + gens_j_t:
        if n is None:
            n = g.n
        elif g.n != n:
            raise InvalidGeneratorError(
                f"两侧生成元变量数不一致：出现 {g.n}，此前为 {n}")
    if n is None:
        raise InvalidGeneratorError(
            "两侧生成元序列都为空：零理想需要变量数，"
            "请至少在一侧给出（零）生成元以确定变量数")
    if n >= MAX_VARIABLES:
        raise InvalidGeneratorError(
            f"原变量数 n={n} 超限：ideal_intersection 要求 1 <= n <= "
            f"{MAX_VARIABLES - 1}（加消元变量 t 后不超过 {MAX_VARIABLES}）")

    def zero_result() -> IntersectionResult:
        ext = GroebnerResult("complete", (), (), (), True, 0, 0, "")
        return IntersectionResult(
            "complete", (), (), (), gens_i_t, gens_j_t, True, "", ext)

    # 任一侧为零理想（无生成元或全零）时交集为零理想。
    if not any(not g.is_zero() for g in gens_i_t):
        return zero_result()
    if not any(not g.is_zero() for g in gens_j_t):
        return zero_result()

    # 扩展环生成元：t*I_j 在前（对齐 I 的位置），(1-t)*J_k 在后。
    ext_gens: list[Poly] = []
    for g in gens_i_t:
        ext_gens.append(_lift_t1(g) if not g.is_zero() else Poly.zero(n + 1))
    for h in gens_j_t:
        ext_gens.append((_lift_t0(h) - _lift_t1(h))
                        if not h.is_zero() else Poly.zero(n + 1))

    ext = groebner_basis(ext_gens, max_pairs=max_pairs, max_terms=max_terms,
                         max_steps=max_steps, _order_key=_elimination_key)

    # 消元：块次序下首项不含 t 的基元素整体不含 t，它们生成交集。
    p = len(gens_i_t)
    basis_out: list[Poly] = []
    certs_i: list[Certificate] = []
    certs_j: list[Certificate] = []
    for b, cert in zip(ext.basis, ext.certificates):
        if _lm_with(b, _elimination_key)[0] != 0:
            continue
        basis_out.append(_drop_t(b))
        certs_i.append(tuple(_eval_t_one(cert[j]) for j in range(p)))
        certs_j.append(tuple(_eval_t_zero(cert[p + k])
                             for k in range(len(gens_j_t))))

    status = "complete" if ext.is_complete else "incomplete"
    return IntersectionResult(
        status, tuple(basis_out), tuple(certs_i), tuple(certs_j),
        gens_i_t, gens_j_t, False, ext.reason, ext)

