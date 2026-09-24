"""多变量多项式的约化 Groebner 基库（有理数系数）。

支持两种单项式次序：graded lex（``"grlex"``，默认）与纯字典序
（``"lex"``，用于消元）。

公开接口：

- 数据结构：:class:`Poly`，常量 :data:`MAX_VARIABLES`、:data:`MAX_TOTAL_TERMS`；
- 基本算法：:func:`divide`、:func:`s_polynomial`；
- Groebner 基：:func:`groebner_basis`、:func:`groebner_basis_of_zero`、
  :func:`check_s_polynomials`、:func:`verify_groebner_basis`；
- lex 消元：:func:`eliminate`；
- 理想成员：:func:`membership`、:func:`membership_in_zero_ideal`；
- 结果对象：:class:`DivisionResult`、:class:`GroebnerResult`、
  :class:`MembershipResult`、:class:`EliminationResult`；
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
    MAX_LEX_VARIABLES,
    DivisionResult,
    EliminationResult,
    GroebnerResult,
    MembershipResult,
    ORDER_KEYS,
    check_s_polynomials,
    divide,
    eliminate,
    groebner_basis,
    groebner_basis_of_zero,
    membership,
    membership_in_zero_ideal,
    s_polynomial,
    verify_groebner_basis,
)
from .poly import MAX_TOTAL_TERMS, MAX_VARIABLES, Poly, grlex_key, lcm_monomial, lex_key

__all__ = [
    "Poly",
    "MAX_VARIABLES",
    "MAX_LEX_VARIABLES",
    "MAX_TOTAL_TERMS",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_TERMS",
    "DEFAULT_MAX_STEPS",
    "ORDER_KEYS",
    "grlex_key",
    "lex_key",
    "lcm_monomial",
    "divide",
    "s_polynomial",
    "groebner_basis",
    "groebner_basis_of_zero",
    "check_s_polynomials",
    "verify_groebner_basis",
    "eliminate",
    "membership",
    "membership_in_zero_ideal",
    "DivisionResult",
    "GroebnerResult",
    "MembershipResult",
    "EliminationResult",
    "GroebnerError",
    "InvalidPolynomialError",
    "InvalidGeneratorError",
    "ZeroPolynomialError",
    "ZeroDivisorError",
]
