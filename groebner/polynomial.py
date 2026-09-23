"""稀疏有理系数多项式与固定 graded lex（grlex）项序。

项（term）用“指数元组”表示：长度等于变量数 n（n <= MAX_VARIABLES），
每个分量为非负 int 整数，例如 (1, 0, 2) 表示 x0 * x2**2。
系数统一为 fractions.Fraction（精确有理数，无浮点误差口径问题）。

grlex 次序（越大越靠前）：先比总次数，总次数相同按字典序（x0 > x1 > ...）。
"""

from __future__ import annotations

import numbers
from fractions import Fraction
from typing import Iterable, Iterator, Mapping, Sequence

MAX_VARIABLES = 4
_MAX_EXPONENT = 10**9  # 指数本身不设题面上限；此处只做合理性护栏，拒绝病态输入


class PolynomialError(ValueError):
    """非法多项式输入。错误信息指明出错位置，便于定位。"""


def _is_real_int(x: object) -> bool:
    """题面：整数参数拒绝 bool。"""
    return isinstance(x, int) and not isinstance(x, bool)


def _check_finite_number(x: object, where: str) -> Fraction:
    """把系数转换为 Fraction；拒绝 bool、NaN、Infinity 及非有限数值。

    Fraction 本身不接受 NaN/Infinity 字符串，但 float('nan')/('inf')
    会先在 float -> Fraction 转换处抛出；这里统一给出可定位错误。
    """
    if isinstance(x, bool):
        raise PolynomialError(f"{where}: bool 不是合法系数（整数参数不接受 bool）")
    if isinstance(x, Fraction):
        return x
    if _is_real_int(x):
        return Fraction(x)
    if isinstance(x, float):
        # 显式拦截 NaN / ±Infinity（Fraction(float('nan')) 本会抛 TypeError，
        # 但题面要求统一为可定位的 PolynomialError）。
        if x != x or x in (float("inf"), float("-inf")):
            raise PolynomialError(f"{where}: 系数必须有限，拒绝 NaN/Infinity")
        return Fraction(x)
    if isinstance(x, numbers.Rational):
        return Fraction(x.numerator, x.denominator)
    if isinstance(x, str):
        s = x.strip()
        if s.lower() in ("nan", "inf", "+inf", "-inf", "infinity", "-infinity"):
            raise PolynomialError(f"{where}: 系数字符串必须有限，拒绝 NaN/Infinity")
        try:
            return Fraction(s)
        except (ZeroDivisionError, ValueError, ArithmeticError) as exc:
            raise PolynomialError(f"{where}: 无法把 {x!r} 解析为有理系数") from exc
    raise PolynomialError(
        f"{where}: 不支持的系数类型 {type(x).__name__}（应为 int/Fraction/有限 float/有理数字符串）"
    )


def _check_nvars(nvars: object) -> int:
    if not _is_real_int(nvars):
        raise PolynomialError("nvars 必须是 int（拒绝 bool）")
    if not 1 <= nvars <= MAX_VARIABLES:
        raise PolynomialError(f"nvars 必须在 1..{MAX_VARIABLES} 之间，得到 {nvars}")
    return nvars


def check_exponent(exp: Sequence[int], nvars: int, where: str = "指数") -> tuple[int, ...]:
    """校验单个指数元组：长度 == nvars，各分量为非负 int（拒绝 bool）。"""
    if not isinstance(exp, (tuple, list)):
        raise PolynomialError(f"{where}: 指数必须是 tuple/list，得到 {type(exp).__name__}")
    if len(exp) != nvars:
        raise PolynomialError(f"{where}: 指数长度 {len(exp)} 与变量数 {nvars} 不一致: {exp!r}")
    out = []
    for i, e in enumerate(exp):
        if not _is_real_int(e):
            raise PolynomialError(f"{where}[{i}]: 指数必须是 int（拒绝 bool），得到 {e!r}")
        if e < 0:
            raise PolynomialError(f"{where}[{i}]: 指数必须非负，得到 {e}")
        if e > _MAX_EXPONENT:
            raise PolynomialError(f"{where}[{i}]: 指数 {e} 超过护栏上限 {_MAX_EXPONENT}")
        out.append(e)
    return tuple(out)


def grlex_key(exp: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """grlex 排序键：键越大项越大。总次数降序，平局时指数向量字典序降序。"""
    return (sum(exp), tuple(exp))


class Poly:
    """固定变量数的稀疏多项式：dict[指数元组 -> Fraction]，零项不保留。

    构造方式：
      - Poly.zero(nvars)
      - Poly.constant(c, nvars=1)
      - Poly.var(i, nvars)
      - Poly({(1, 0): 1, (0, 2): -1}, nvars=2)
      - Poly.from_terms([((1,0), 1), ...], nvars=2)
    多项式一旦构造即为不可变值对象（内部 dict 不对外暴露可变引用）。
    """

    __slots__ = ("_nvars", "_terms")

    def __init__(
        self,
        terms: Mapping[Sequence[int], object] | Iterable[tuple[Sequence[int], object]] | None = None,
        nvars: int | None = None,
    ) -> None:
        if nvars is None:
            raise PolynomialError("构造 Poly 必须显式提供 nvars")
        n = _check_nvars(nvars)
        data: dict[tuple[int, ...], Fraction] = {}
        if terms is None:
            iterable: Iterable[tuple[Sequence[int], object]] = ()
        elif isinstance(terms, Mapping):
            iterable = terms.items()
        else:
            iterable = terms
        for item in iterable:
            try:
                exp, coef = item
            except (TypeError, ValueError) as exc:
                raise PolynomialError(f"项必须是 (指数, 系数) 对，得到 {item!r}") from exc
            e = check_exponent(exp, n, where="多项式项")
            c = _check_finite_number(coef, where=f"项 {e} 的系数")
            if c:
                data[e] = data.get(e, Fraction(0)) + c
        # 删除合并后变零的项
        self._terms: dict[tuple[int, ...], Fraction] = {e: c for e, c in data.items() if c != 0}
        self._nvars = n

    # ----- 基本构造子 -----
    @classmethod
    def zero(cls, nvars: int) -> "Poly":
        n = _check_nvars(nvars)
        p = cls.__new__(cls)
        p._nvars = n
        p._terms = {}
        return p

    @classmethod
    def constant(cls, c: object, nvars: int) -> "Poly":
        n = _check_nvars(nvars)
        coef = _check_finite_number(c, where="常数项系数")
        p = cls.__new__(cls)
        p._nvars = n
        p._terms = {tuple([0] * n): coef} if coef != 0 else {}
        return p

    @classmethod
    def var(cls, index: int, nvars: int) -> "Poly":
        n = _check_nvars(nvars)
        if not _is_real_int(index):
            raise PolynomialError("变量下标必须是 int（拒绝 bool）")
        if not 0 <= index < n:
            raise PolynomialError(f"变量下标 {index} 超出 [0, {n - 1}]")
        p = cls.__new__(cls)
        p._nvars = n
        e = [0] * n
        e[index] = 1
        p._terms = {tuple(e): Fraction(1)}
        return p

    @classmethod
    def from_terms(cls, pairs: Iterable[tuple[Sequence[int], object]], nvars: int) -> "Poly":
        return cls(list(pairs), nvars=nvars)

    # ----- 基本性质 -----
    @property
    def nvars(self) -> int:
        return self._nvars

    def is_zero(self) -> bool:
        return not self._terms

    def is_constant(self) -> bool:
        return all(sum(e) == 0 for e in self._terms)

    def __len__(self) -> int:
        """非零项数。"""
        return len(self._terms)

    def terms(self) -> list[tuple[tuple[int, ...], Fraction]]:
        """按 grlex 从大到小返回 [(指数, 系数), ...] 的快照。"""
        return sorted(self._terms.items(), key=lambda ec: grlex_key(ec[0]), reverse=True)

    def items(self) -> "Iterator[tuple[tuple[int, ...], Fraction]]":
        return iter(self._terms.items())

    def leading_monomial(self) -> tuple[int, ...]:
        if not self._terms:
            raise PolynomialError("零多项式没有首项")
        return max(self._terms, key=grlex_key)

    def leading_coefficient(self) -> Fraction:
        if not self._terms:
            raise PolynomialError("零多项式没有首项系数")
        return self._terms[self.leading_monomial()]

    def leading_term(self) -> tuple[tuple[int, ...], Fraction]:
        m = self.leading_monomial()
        return m, self._terms[m]

    def coefficient(self, exp: Sequence[int]) -> Fraction:
        e = check_exponent(exp, self._nvars, where="coefficient 查询指数")
        return self._terms.get(e, Fraction(0))

    # ----- 内部工具 -----
    def _same_n(self, other: "Poly", op: str) -> None:
        if not isinstance(other, Poly):
            raise PolynomialError(f"{op}: 操作数必须是 Poly，得到 {type(other).__name__}")
        if other._nvars != self._nvars:
            raise PolynomialError(
                f"{op}: 变量数不一致 {self._nvars} != {other._nvars}"
            )

    # ----- 算术 -----
    def __add__(self, other: object) -> "Poly":
        if not isinstance(other, Poly):
            c = _check_finite_number(other, where="加法标量")
            other = Poly.constant(c, self._nvars)
        self._same_n(other, "加法")
        data = dict(self._terms)
        for e, c in other._terms.items():
            data[e] = data.get(e, Fraction(0)) + c
        return self._from_dict({e: c for e, c in data.items() if c != 0}, self._nvars)

    def __radd__(self, other: object) -> "Poly":
        return self.__add__(other)

    def __sub__(self, other: object) -> "Poly":
        if not isinstance(other, Poly):
            c = _check_finite_number(other, where="减法标量")
            other = Poly.constant(c, self._nvars)
        self._same_n(other, "减法")
        data = dict(self._terms)
        for e, c in other._terms.items():
            data[e] = data.get(e, Fraction(0)) - c
        return self._from_dict({e: c for e, c in data.items() if c != 0}, self._nvars)

    def __rsub__(self, other: object) -> "Poly":
        c = _check_finite_number(other, where="减法标量")
        return Poly.constant(c, self._nvars) - self

    def __neg__(self) -> "Poly":
        return self._from_dict({e: -c for e, c in self._terms.items()}, self._nvars)

    def __mul__(self, other: object) -> "Poly":
        if isinstance(other, Poly):
            self._same_n(other, "乘法")
            data: dict[tuple[int, ...], Fraction] = {}
            for e1, c1 in self._terms.items():
                for e2, c2 in other._terms.items():
                    e = tuple(a + b for a, b in zip(e1, e2))
                    data[e] = data.get(e, Fraction(0)) + c1 * c2
            return self._from_dict({e: c for e, c in data.items() if c != 0}, self._nvars)
        # 标量乘法
        c = _check_finite_number(other, where="标量乘法系数")
        if c == 0:
            return Poly.zero(self._nvars)
        return self._from_dict({e: coef * c for e, coef in self._terms.items()}, self._nvars)

    def __rmul__(self, other: object) -> "Poly":
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> "Poly":
        """非负整数次幂（快速幂）；指数必须是 int 且非负（拒绝 bool）。"""
        if not _is_real_int(exponent):
            raise PolynomialError("幂指数必须是非负 int（拒绝 bool）")
        if exponent < 0:
            raise PolynomialError("多项式不支持负整数次幂")
        result = Poly.constant(1, self._nvars)
        base = self
        k = exponent
        while k:
            if k & 1:
                result = result * base
            base = base * base
            k >>= 1
        return result

    def __truediv__(self, other: object) -> "Poly":
        """仅支持除以非零有理标量（多项式除法见 engine.divide）。"""
        c = _check_finite_number(other, where="标量除法分母")
        if c == 0:
            raise PolynomialError("不能除以零标量")
        return self._from_dict({e: coef / c for e, coef in self._terms.items()}, self._nvars)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self._nvars == other._nvars and self._terms == other._terms

    def __hash__(self) -> int:
        return hash((self._nvars, tuple(sorted(self._terms.items()))))

    @classmethod
    def _from_dict(cls, data: dict[tuple[int, ...], Fraction], nvars: int) -> "Poly":
        p = cls.__new__(cls)
        p._nvars = nvars
        p._terms = data
        return p

    def scale_to_monic(self) -> "Poly":
        """除以首项系数，得到首一多项式；零多项式原样返回。"""
        if not self._terms:
            return self
        lc = self.leading_coefficient()
        if lc == 1:
            return self
        return self / lc

    def is_monic(self) -> bool:
        return self.is_zero() or self.leading_coefficient() == 1

    def copy(self) -> "Poly":
        return self._from_dict(dict(self._terms), self._nvars)

    # ----- 显示 -----
    _VAR_NAMES = ("x", "y", "z", "w")

    def _name(self, i: int) -> str:
        return self._VAR_NAMES[i] if i < len(self._VAR_NAMES) else f"x{i}"

    def _term_str(self, exp: tuple[int, ...], coef: Fraction, first: bool) -> str:
        sign = "+" if coef > 0 else "-"
        abs_c = abs(coef)
        mono = "".join(
            (self._name(i) if k == 1 else f"{self._name(i)}^{k}")
            for i, k in enumerate(exp)
            if k
        )
        if mono == "":
            body = str(abs_c)
        elif abs_c == 1:
            body = mono
        else:
            body = f"{abs_c}*{mono}"
        if first:
            return body if coef > 0 else f"-{body}"
        return f"{sign} {body}"

    def __str__(self) -> str:
        if not self._terms:
            return "0"
        ordered = self.terms()
        parts = [self._term_str(ordered[0][0], ordered[0][1], True)]
        for e, c in ordered[1:]:
            parts.append(self._term_str(e, c, False))
        return " ".join(parts)

    def __repr__(self) -> str:
        if not self._terms:
            return f"Poly.zero({self._nvars})"
        inner = ", ".join(f"{list(e)}: {c!r}" for e, c in self.terms())
        return f"Poly({{{inner}}}, nvars={self._nvars})"


def lcm_monomial(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """两个单项式的最小公倍式（指数逐位取 max）。"""
    return tuple(x if x >= y else y for x, y in zip(a, b))


def divides_monomial(a: Sequence[int], b: Sequence[int]) -> bool:
    """单项式 a 是否整除 b。"""
    return all(x <= y for x, y in zip(a, b))


def monomial_factor(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """b / a 的指数差；要求 a 整除 b。"""
    if not divides_monomial(a, b):
        raise PolynomialError(f"单项式 {tuple(a)} 不整除 {tuple(b)}")
    return tuple(y - x for x, y in zip(a, b))
