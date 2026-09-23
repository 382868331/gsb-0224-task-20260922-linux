"""异常类型。

所有本模块抛出的异常都继承自 :class:`GroebnerError`，便于调用方统一捕获。
非法输入同时继承 ``ValueError``，符合 Python 对“非法参数值”的惯例。
"""

from __future__ import annotations


class GroebnerError(Exception):
    """本库所有异常的公共基类。"""


class InvalidPolynomialError(ValueError, GroebnerError):
    """多项式的数据本身非法。

    常见来源：系数为 NaN/Infinity 或 bool、指数为负/非整数/bool、
    指数向量长度与变量数不一致、变量数超出 1..4 范围。

    错误消息中会尽量携带 ``term``（出问题的指数向量）与 ``index``
    （在输入序列中的位置），保证错误“可定位”。
    """


class InvalidGeneratorError(ValueError, GroebnerError):
    """生成元序列不满足本库契约。

    常见来源：元素不是 :class:`~groebner.poly.Poly`、变量数彼此不一致、
    总输入项数超过 100、预算参数不是正整数（且不是 bool）。
    """


class ZeroPolynomialError(GroebnerError):
    """对零多项式索取首项/首项系数/首项单项式。"""


class ZeroDivisorError(ValueError, GroebnerError):
    """除法的除式列表中含有零多项式（零多项式不能作除式）。"""
