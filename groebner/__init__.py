"""多变量多项式的约化 Groebner 基库（有理数系数，固定 graded lex 次序）。

公开接口：

- 数据结构：:class:`Poly`，常量 :data:`MAX_VARIABLES`、:data:`MAX_TOTAL_TERMS`；
- 基本算法：:func:`divide`、:func:`s_polynomial`；
- Groebner 基：:func:`groebner_basis`、:func:`groebner_basis_of_zero`、
  :func:`check_s_polynomials`、:func:`verify_groebner_basis`；
- 理想成员：:func:`membership`、:func:`membership_in_zero_ideal`；
- 理想交集：``ideal_intersection``（消元变量块次序仅在该入口内部使用）；
- 结果对象：:class:`DivisionResult`、:class:`GroebnerResult`、
  :class:`MembershipResult`；
- 异常：:class:`GroebnerError`、:class:`InvalidPolynomialError`、
  :class:`InvalidGeneratorError`、:class:`ZeroPolynomialError`、
  :class:`ZeroDivisorError`。

全程精确有理数运算，标准库实现，不依赖任何符号计算第三方库。
"""

from __future__ import annotations

from .errors import (
    GroebnerError,
    InvalidGeneratorError,
    InvalidPolynomialError,
    ZeroDivisorError,
    ZeroPolynomialError,
)
from .groebner import (
    DEFAULT_MAX_PAIRS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TERMS,
    DivisionResult,
    GroebnerResult,
    IntersectionResult,
    MembershipResult,
    check_s_polynomials,
    divide,
    groebner_basis,
    groebner_basis_of_zero,
    ideal_intersection,
    membership,
    membership_in_zero_ideal,
    s_polynomial,
    verify_groebner_basis,
)
from .poly import MAX_TOTAL_TERMS, MAX_VARIABLES, Poly, grlex_key, lcm_monomial

__all__ = [
    "Poly",
    "MAX_VARIABLES",
    "MAX_TOTAL_TERMS",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_TERMS",
    "DEFAULT_MAX_STEPS",
    "grlex_key",
    "lcm_monomial",
    "divide",
    "s_polynomial",
    "groebner_basis",
    "groebner_basis_of_zero",
    "ideal_intersection",
    "check_s_polynomials",
    "verify_groebner_basis",
    "membership",
    "membership_in_zero_ideal",
    "DivisionResult",
    "GroebnerResult",
    "IntersectionResult",
    "MembershipResult",
    "GroebnerError",
    "InvalidPolynomialError",
    "InvalidGeneratorError",
    "ZeroPolynomialError",
    "ZeroDivisorError",
]
