"""groebner: 有理系数稀疏多项式与约化 Groebner 基（graded lex，变量 <= 4）。

公开接口见包文档与 README。仅使用 Python 3.14 标准库（fractions）。
"""

from .polynomial import (
    Poly,
    PolynomialError,
    MAX_VARIABLES,
    grlex_key,
)
from .engine import (
    GroebnerBudgets,
    GroebnerBasisResult,
    DivisionResult,
    MembershipResult,
    MAX_INPUT_TERMS,
    divide,
    remainder,
    s_polynomial,
    groebner_basis,
    reduced_basis,
    ideal_membership,
    membership_with_basis,
)

# 题面统一称为“可定位的非法输入错误”：PolynomialError 的别名
GroebnerInputError = PolynomialError

__all__ = [
    "Poly",
    "PolynomialError",
    "GroebnerInputError",
    "MAX_VARIABLES",
    "MAX_INPUT_TERMS",
    "GroebnerBudgets",
    "GroebnerBasisResult",
    "DivisionResult",
    "MembershipResult",
    "grlex_key",
    "divide",
    "remainder",
    "s_polynomial",
    "groebner_basis",
    "reduced_basis",
    "ideal_membership",
    "membership_with_basis",
]
