"""多变量多项式的约化 Groebner 基库（有理数系数，固定 graded lex 次序）。

公开接口：

- 数据结构：:class:`Poly`，常量 :data:`MAX_VARIABLES`、:data:`MAX_TOTAL_TERMS`；
- 基本算法：:func:`divide`、:func:`s_polynomial`；
- Groebner 基：:func:`groebner_basis`、:func:`groebner_basis_of_zero`、
  :func:`check_s_polynomials`、:func:`verify_groebner_basis`；
- 理想成员：:func:`membership`、:func:`membership_in_zero_ideal`；
- 次数有界 syzygy：:func:`bounded_syzygies`、:func:`relation_value`、
  :func:`monomials_up_to_degree`、:class:`BoundedSyzygyResult`；
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
    MembershipResult,
    check_s_polynomials,
    divide,
    groebner_basis,
    groebner_basis_of_zero,
    membership,
    membership_in_zero_ideal,
    s_polynomial,
    verify_groebner_basis,
)
from .poly import MAX_TOTAL_TERMS, MAX_VARIABLES, Poly, grlex_key, lcm_monomial
from .syzygy import (
    DEFAULT_MAX_UNKNOWNS,
    BoundedSyzygyResult,
    bounded_syzygies,
    monomials_up_to_degree,
    relation_value,
)

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
    "check_s_polynomials",
    "verify_groebner_basis",
    "membership",
    "membership_in_zero_ideal",
    "bounded_syzygies",
    "relation_value",
    "monomials_up_to_degree",
    "BoundedSyzygyResult",
    "DEFAULT_MAX_UNKNOWNS",
    "DivisionResult",
    "GroebnerResult",
    "MembershipResult",
    "GroebnerError",
    "InvalidPolynomialError",
    "InvalidGeneratorError",
    "ZeroPolynomialError",
    "ZeroDivisorError",
]
