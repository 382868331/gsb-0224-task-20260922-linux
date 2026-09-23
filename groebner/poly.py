"""稀疏有理系数多项式与固定 graded lex（次数优先、并列时按变量 x1>x2>...）次序。

多项式用不可变值对象 :class:`Poly` 表示：

- 内部存储为 ``dict[tuple[int, ...], Fraction]``，只保留非零系数的单项式；
- 指数向量长度恒等于变量数 ``n``（``n`` 满足 ``1 <= n <= 4``）；
- 系数全部为精确的 :class:`fractions.Fraction`，全程无浮点，因此不存在舍入误差；
- 空字典表示零多项式，``Poly.zero(n)`` 是其规范构造方式。

单项式次序（grlex）::

    alpha > beta  当且仅当  |alpha| > |beta|，
    或 |alpha| == |beta| 且 alpha 在字典序下大于 beta（x1 > x2 > ...）。

例如二元情形 ``x^2 > xy > y^2 > x > y > 1``。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, Iterator, Mapping, Sequence, Union

from .errors import InvalidPolynomialError, ZeroPolynomialError

MAX_VARIABLES = 4
MAX_TOTAL_TERMS = 100

Coefficient = Union[int, Fraction]


def _reject_bool(value: object, what: str, term: object = None,
                 index: object = None) -> None:
    # bool 是 int 的子类：按契约，整数类输入一律拒绝 bool，避免 True 被当成 1。
    if isinstance(value, bool):
        loc = _location(term, index)
        raise InvalidPolynomialError(
            f"{what} 不接受 bool（得到 {value!r}）{loc}；布尔值不会被静默当作 0/1")


def _reject_nonfinite(value: float, what: str, term: object = None,
                      index: object = None) -> None:
    if not math.isfinite(value):
        loc = _location(term, index)
        raise InvalidPolynomialError(
            f"{what} 必须有限，拒绝 NaN/Infinity（得到 {value!r}）{loc}")


def _location(term: object, index: object) -> str:
    parts = []
    if isinstance(index, int) and not isinstance(index, bool):
        parts.append(f"输入位置 index={index}")
    if term is not None:
        parts.append(f"单项式指数={tuple(term)!r}")
    return f"（{'，'.join(parts)}）" if parts else ""


def _check_exponent(exp: object, n: int, index: object = None) -> tuple[int, ...]:
    """校验单个指数向量，返回规范化后的元组。"""
    if not isinstance(exp, (tuple, list)):
        raise InvalidPolynomialError(
            f"单项式指数必须是 tuple/list 长度为 {n}，得到 {exp!r}"
            f"{_location(None, index)}")
    if len(exp) != n:
        raise InvalidPolynomialError(
            f"单项式指数长度必须等于变量数 {n}，得到长度 {len(exp)}："
            f"{tuple(exp)!r}{_location(None, index)}")
    out = []
    for k, e in enumerate(exp):
        _reject_bool(e, f"第 {k} 个指数", exp, index)
        if not isinstance(e, int):
            raise InvalidPolynomialError(
                f"指数必须是 int，第 {k} 个分量得到 {e!r}"
                f"{_location(exp, index)}")
        if e < 0:
            raise InvalidPolynomialError(
                f"指数必须非负，第 {k} 个分量得到 {e}"
                f"{_location(exp, index)}")
        out.append(e)
    return tuple(out)


def _check_coefficient(c: object, term: object = None,
                       index: object = None) -> Fraction:
    """校验并转换系数；bool、NaN、Infinity 一律拒绝。"""
    _reject_bool(c, "系数", term, index)
    if isinstance(c, Fraction):
        return c
    if isinstance(c, int):
        return Fraction(c)
    if isinstance(c, float):
        _reject_nonfinite(c, "系数", term, index)
        return Fraction(c)
    raise InvalidPolynomialError(
        f"系数必须是 int / fractions.Fraction / 有限 float，得到 {c!r}"
        f"{_location(term, index)}")


def grlex_key(exp: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """grlex 排序键；键越大，单项式越大。"""
    return (sum(exp), exp)


class Poly:
    """不可变稀疏多项式。

    构造方式：

    - ``Poly({(1, 0): 1, (0, 2): -2}, n=2)`` 表示 x - 2*y**2；
    - :meth:`constant`、:meth:`variable`、:meth:`zero` 命名构造器；
    - 同类项自动合并，零系数项自动剔除；
    - 传入 ``terms`` 为空时得到零多项式。

    所有输入在构造时严格校验（变量数、指数、系数），
    构造成功的 :class:`Poly` 对其余算法永远满足不变式。
    """

    __slots__ = ("_terms", "_n")

    def __init__(self,
                 terms: Mapping[Sequence[int], Coefficient]
                       | Iterable[tuple[Sequence[int], Coefficient]]
                       | None = None,
                 *,
                 n: int | None = None):
        _reject_bool(n, "变量数 n")
        if n is None:
            if terms is None:
                raise InvalidPolynomialError(
                    "必须指定变量数 n（terms 为空时无法推断）")
        elif not isinstance(n, int) or not (1 <= n <= MAX_VARIABLES):
            raise InvalidPolynomialError(
                f"变量数 n 必须是 1..{MAX_VARIABLES} 的 int，得到 {n!r}")

        if terms is None:
            self._terms: dict[tuple[int, ...], Fraction] = {}
            self._n = n  # type: ignore[assignment]
            return

        if isinstance(terms, Mapping):
            items: Iterable[tuple[object, object]] = terms.items()
        elif isinstance(terms, Iterable) and not isinstance(terms, (str, bytes)):
            items = terms
        else:
            raise InvalidPolynomialError(
                f"terms 必须是映射或 (指数, 系数) 可迭代对象，得到 {terms!r}")

        merged: dict[tuple[int, ...], Fraction] = {}
        inferred_n = n
        for idx, (raw_exp, raw_c) in enumerate(items):
            if isinstance(raw_exp, int) and not isinstance(raw_exp, bool) \
                    and inferred_n is None:
                raise InvalidPolynomialError(
                    "无法从未经分组的键推断变量数，请显式传 n")
            if inferred_n is None:
                if not isinstance(raw_exp, (tuple, list)):
                    raise InvalidPolynomialError(
                        f"单项式指数必须是 tuple/list，得到 {raw_exp!r}"
                        f"{_location(None, idx)}")
                inferred_n = len(raw_exp)
                if not (1 <= inferred_n <= MAX_VARIABLES):
                    raise InvalidPolynomialError(
                        f"变量数必须在 1..{MAX_VARIABLES}，得到 {inferred_n}"
                        f"{_location(None, idx)}")
            exp = _check_exponent(raw_exp, inferred_n, idx)
            coef = _check_coefficient(raw_c, exp, idx)
            if coef:
                merged[exp] = merged.get(exp, Fraction(0)) + coef
                if not merged[exp]:
                    del merged[exp]

        if inferred_n is None:
            # 空 terms 且未给 n 的情形上面已拦截；这里仅为类型检查器。
            raise InvalidPolynomialError("无法推断变量数，请显式传 n")
        self._terms = merged
        self._n = inferred_n

    # ------------------------------------------------------------------ 基本

    @property
    def n(self) -> int:
        """变量数。"""
        return self._n

    @property
    def terms(self) -> dict[tuple[int, ...], Fraction]:
        """单项式 -> 系数的副本（修改不影响对象）。"""
        return dict(self._terms)

    def __len__(self) -> int:
        """非零项数。"""
        return len(self._terms)

    def is_zero(self) -> bool:
        return not self._terms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self._n == other._n and self._terms == other._terms

    def __hash__(self) -> int:
        return hash((self._n, tuple(sorted(self._terms.items()))))

    def __iter__(self) -> Iterator[tuple[tuple[int, ...], Fraction]]:
        """按 grlex 降序迭代 ``(指数, 系数)``。"""
        for exp in sorted(self._terms, key=grlex_key, reverse=True):
            yield exp, self._terms[exp]

    def __repr__(self) -> str:
        if self.is_zero():
            return f"Poly.zero({self._n})"
        body = ", ".join(f"{exp}: {_fmt_frac(self._terms[exp])}"
                         for exp in self._terms)
        return f"Poly({{{body}}}, n={self._n})"

    def monomial_coefficient(self, exp: Sequence[int]) -> Fraction:
        """取指定单项式的系数（不存在则为 0）；同样校验指数。"""
        return self._terms.get(_check_exponent(exp, self._n), Fraction(0))

    # ------------------------------------------------------------- 首项相关

    def lm(self) -> tuple[int, ...]:
        """首项单项式（leading monomial），零多项式抛 :class:`ZeroPolynomialError`。"""
        if not self._terms:
            raise ZeroPolynomialError("零多项式没有首项单项式")
        return max(self._terms, key=grlex_key)

    def lc(self) -> Fraction:
        """首项系数。"""
        if not self._terms:
            raise ZeroPolynomialError("零多项式没有首项系数")
        return self._terms[self.lm()]

    def lt(self) -> tuple[tuple[int, ...], Fraction]:
        """返回 ``(首项单项式, 首项系数)``。"""
        exp = self.lm()
        return exp, self._terms[exp]

    # ----------------------------------------------------------------- 运算

    def _check_same(self, other: "Poly", op: str) -> None:
        if not isinstance(other, Poly):
            raise TypeError(f"{op} 只支持 Poly 与 Poly，得到 {type(other).__name__}")
        if other._n != self._n:
            raise InvalidPolynomialError(
                f"{op} 要求两个多项式变量数相同：{self._n} != {other._n}")

    def __add__(self, other: "Poly") -> "Poly":
        self._check_same(other, "加法")
        out = dict(self._terms)
        for exp, c in other._terms.items():
            new = out.get(exp, Fraction(0)) + c
            if new:
                out[exp] = new
            else:
                out.pop(exp, None)
        return Poly._from_dict(out, self._n)

    def __sub__(self, other: "Poly") -> "Poly":
        self._check_same(other, "减法")
        out = dict(self._terms)
        for exp, c in other._terms.items():
            new = out.get(exp, Fraction(0)) - c
            if new:
                out[exp] = new
            else:
                out.pop(exp, None)
        return Poly._from_dict(out, self._n)

    def __neg__(self) -> "Poly":
        return Poly._from_dict({e: -c for e, c in self._terms.items()}, self._n)

    def __mul__(self, other: "Poly | int | Fraction | float") -> "Poly":
        if isinstance(other, Poly):
            self._check_same(other, "乘法")
            out: dict[tuple[int, ...], Fraction] = {}
            for e1, c1 in self._terms.items():
                for e2, c2 in other._terms.items():
                    e = tuple(a + b for a, b in zip(e1, e2))
                    new = out.get(e, Fraction(0)) + c1 * c2
                    if new:
                        out[e] = new
                    else:
                        out.pop(e, None)
            return Poly._from_dict(out, self._n)
        # bool 在 scale 的系数校验中被拒绝；float 经 Fraction 精确化（NaN/Inf 拒绝）。
        if isinstance(other, (int, Fraction, float)) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    def __radd__(self, other: "Poly") -> "Poly":
        return self.__add__(other)

    def __rsub__(self, other: "Poly") -> "Poly":
        return other.__sub__(self)

    def __rmul__(self, other: "int | Fraction") -> "Poly":
        return self.scale(other)

    def scale(self, c: Coefficient | float) -> "Poly":
        """乘以标量；标量同样拒绝 bool/NaN/Infinity，乘 0 得零多项式。"""
        cf = _check_coefficient(c)
        if cf == 0:
            return Poly.zero(self._n)
        return Poly._from_dict({e: v * cf for e, v in self._terms.items()}, self._n)

    def normalize_monic(self) -> "Poly":
        """除以自身首项系数，使其首一；零多项式原样返回。"""
        if self.is_zero():
            return Poly.zero(self._n)
        return self.scale(Fraction(1, 1) / self.lc())

    # ------------------------------------------------------------- 显示辅助

    def to_string(self, names: Sequence[str] | None = None) -> str:
        """人类可读形式，如 ``x**2 - 2*x*y + 1``。变量名仅用于展示。"""
        if names is not None:
            if len(names) != self._n:
                raise InvalidPolynomialError(
                    f"变量名个数 {len(names)} != 变量数 {self._n}")
        else:
            names = tuple(f"x{i + 1}" for i in range(self._n))
        if self.is_zero():
            return "0"
        parts: list[str] = []
        for exp, c in self:
            if not parts:
                sign = "-" if c < 0 else ""
            else:
                sign = "- " if c < 0 else "+ "
            ac = abs(c)
            if sum(exp) == 0:
                term = _fmt_frac(ac)
            else:
                factors = []
                for name, k in zip(names, exp):
                    if k == 1:
                        factors.append(name)
                    elif k:
                        factors.append(f"{name}**{k}")
                mono = "*".join(factors)
                term = mono if ac == 1 else f"{_fmt_frac(ac)}*{mono}"
            parts.append(f"{sign}{term}".strip())
        return " ".join(parts)

    def __str__(self) -> str:
        return self.to_string()

    # ------------------------------------------------------------- 内部/构造

    @classmethod
    def _from_dict(cls, d: dict[tuple[int, ...], Fraction], n: int) -> "Poly":
        """绕过校验的内部构造器：键/系数已由算法保证合法。"""
        obj = object.__new__(cls)
        obj._terms = {e: c for e, c in d.items() if c != 0}
        obj._n = n
        return obj

    @classmethod
    def zero(cls, n: int) -> "Poly":
        _reject_bool(n, "变量数 n")
        if not isinstance(n, int) or not (1 <= n <= MAX_VARIABLES):
            raise InvalidPolynomialError(
                f"变量数 n 必须是 1..{MAX_VARIABLES} 的 int，得到 {n!r}")
        return cls._from_dict({}, n)

    @classmethod
    def constant(cls, value: Coefficient | float, n: int) -> "Poly":
        """常数多项式（可为 0）。"""
        cf = _check_coefficient(value)
        if cf == 0:
            return cls.zero(n)
        return cls._from_dict({tuple([0] * n): cf}, n)

    @classmethod
    def variable(cls, n: int, index: int) -> "Poly":
        """变量 ``x_{index+1}``；``index`` 从 0 开始。"""
        _reject_bool(n, "变量数 n")
        _reject_bool(index, "变量下标 index")
        if not isinstance(n, int) or not (1 <= n <= MAX_VARIABLES):
            raise InvalidPolynomialError(
                f"变量数 n 必须是 1..{MAX_VARIABLES} 的 int，得到 {n!r}")
        if not isinstance(index, int) or not (0 <= index < n):
            raise InvalidPolynomialError(
                f"变量下标 index 必须是 0..{n - 1} 的 int，得到 {index!r}")
        exp = [0] * n
        exp[index] = 1
        return cls._from_dict({tuple(exp): Fraction(1)}, n)


def _fmt_frac(c: Fraction) -> str:
    if c.denominator == 1:
        return str(c.numerator)
    return f"({c.numerator}/{c.denominator})"


def lcm_monomial(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """两个单项式的最小公倍式（逐分量取 max）。"""
    return tuple(max(x, y) for x, y in zip(a, b))
