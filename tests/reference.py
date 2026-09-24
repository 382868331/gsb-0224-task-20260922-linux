"""测试专用的独立参考实现。

刻意与被测库 ``groebner`` **完全独立**：

- 多项式直接用 ``dict[tuple[int,...], Fraction]`` 表示；
- 加法/乘法/约化/S 多项式/Buchberger 全部在这里另写一遍朴素实现；
- 不导入 ``groebner`` 的任何算法函数，因此它给出的期望值
  不可能“抄自”被测核心。

它只用 Python 标准库（fractions）。功能够用即止，不做输入校验、
不做预算控制——那是被测库的职责，本文件只负责独立地给出数学期望。
"""

from __future__ import annotations

from fractions import Fraction


# --------------------------------------------------------------- dict 多项式运算

def _clean(d):
    return {e: c for e, c in d.items() if c != 0}


def add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, Fraction(0)) + c
    return _clean(out)


def neg(a):
    return _clean({e: -c for e, c in a.items()})


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    out = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            e = tuple(x + y for x, y in zip(e1, e2))
            out[e] = out.get(e, Fraction(0)) + c1 * c2
    return _clean(out)


def smul(c, a):
    if c == 0:
        return {}
    return {e: c * v for e, v in a.items()}


def is_zero(a):
    return not a


def key_grlex(exp):
    """grlex 排序键：总次数优先，并列按指数向量字典序（x1 优先）。"""
    return (sum(exp), exp)


def key_lex(exp):
    """纯 lex 排序键：指数向量自身的字典序（x1 > x2 > ...）。"""
    return tuple(exp)


# 旧名字保留给既有调用；默认 grlex。
key = key_grlex


def lm(a, key=key_grlex):
    return max(a, key=key)


def lc(a, key=key_grlex):
    return a[lm(a, key)]


def divides(u, v):
    return all(x <= y for x, y in zip(u, v))


def lcm_exp(u, v):
    return tuple(max(x, y) for x, y in zip(u, v))


def monom(n, exp, coef=Fraction(1)):
    return {tuple(exp): Fraction(coef)}


# ------------------------------------------------------------------- 除法/约化

def remainder(f, divisors, key=key_grlex):
    """f 对除式列表反复取首项约化得到的余式（依赖除式顺序，与教科书一致）。"""
    r = {}
    p = f
    while p:
        e, c = lm(p, key), lc(p, key)
        hit = None
        for g in divisors:
            if divides(lm(g, key), e):
                hit = g
                break
        if hit is None:
            r = add(r, monom(len(e), e, c))
            p = sub(p, monom(len(e), e, c))
        else:
            ge, gc = lm(hit, key), lc(hit, key)
            shift = tuple(a - b for a, b in zip(e, ge))
            qterm = monom(len(e), shift, c / gc)
            p = sub(p, mul(qterm, hit))
    return r


def divide_full(f, divisors, key=key_grlex):
    """返回 (商列表, 余式)，满足 f = sum q_i g_i + r。"""
    if f:
        n = len(lm(f, key))
    elif divisors:
        n = len(lm(divisors[0], key))
    else:
        n = 0
    qs = [{} for _ in divisors]
    r = {}
    p = f
    while p:
        e, c = lm(p, key), lc(p, key)
        hit = None
        for i, g in enumerate(divisors):
            if divides(lm(g, key), e):
                hit = i
                break
        if hit is None:
            r = add(r, monom(n, e, c))
            p = sub(p, monom(n, e, c))
        else:
            g = divisors[hit]
            shift = tuple(a - b for a, b in zip(e, lm(g, key)))
            qterm = monom(n, shift, c / lc(g, key))
            qs[hit] = add(qs[hit], qterm)
            p = sub(p, mul(qterm, g))
    return qs, r


# ------------------------------------------------------------------- S 多项式

def s_poly(f, g, key=key_grlex):
    l = lcm_exp(lm(f, key), lm(g, key))
    tf = monom(len(l), tuple(a - b for a, b in zip(l, lm(f, key))),
               Fraction(1) / lc(f, key))
    tg = monom(len(l), tuple(a - b for a, b in zip(l, lm(g, key))),
               Fraction(1) / lc(g, key))
    return sub(mul(tf, f), mul(tg, g))


def _monic(f, key=key_grlex):
    return smul(Fraction(1) / lc(f, key), f)


# --------------------------------------------------------------- Buchberger

def reduced_groebner(gens, key=key_grlex):
    """朴素 Buchberger + 首一约化，返回按首项降序的 dict 列表。

    与被测实现刻意不同：不做首项互素跳过、不用队列，
    每轮对当前所有对取余式（余式按**活动基**约化，保证新首项严格
    扩大首项理想，过程必终止），直到一轮没有新元素。

    ``key`` 选择单项式次序（默认 grlex，可传 :func:`key_lex`）。
    """
    basis = []
    for g in gens:
        if not g:
            continue
        r = remainder(g, basis, key)
        if r:
            basis.append(_monic(r, key))

    while True:
        added = False
        size = len(basis)
        for i in range(size):
            for j in range(i + 1, size):
                r = remainder(s_poly(basis[i], basis[j], key), basis, key)
                if r:
                    basis.append(_monic(r, key))
                    added = True
        if not added:
            break

    # 同首项只保留一个（极小化前先去重，避免重复首项互相“整除”）
    uniq, seen = [], set()
    for b in basis:
        e = lm(b, key)
        if e not in seen:
            seen.add(e)
            uniq.append(b)

    # 极小化：首项被其他不同的首项严格整除者删除
    minimal = [
        b for b in uniq
        if not any(b is not k
                   and lm(b, key) != lm(k, key)
                   and divides(lm(k, key), lm(b, key))
                   for k in uniq)
    ]

    # 首一约化到不动点
    reduced = [dict(b) for b in minimal]
    changed = True
    while changed:
        changed = False
        for pos in range(len(reduced)):
            others = reduced[:pos] + reduced[pos + 1:]
            r = remainder(reduced[pos], others, key)
            r = _monic(r, key) if r else r
            if r != reduced[pos]:
                reduced[pos] = r
                changed = True

    reduced.sort(key=lambda b: key(lm(b, key)), reverse=True)
    return reduced


def ideal_remainder(f, gens, key=key_grlex):
    """成员判定的独立期望：对独立算出的约化基取余式。"""
    basis = reduced_groebner(gens, key)
    if not basis:
        return dict(f)
    return remainder(f, basis, key)


def expand(cert_dicts, gen_dicts):
    """独立展开证书：sum_j cert_j * gen_j，返回 dict 多项式。"""
    total = {}
    for c, g in zip(cert_dicts, gen_dicts):
        total = add(total, mul(c, g))
    return total


# ------------------------------------------------------------------- lex 消元


def permute_poly(d, perm):
    """变量置换：new_exp[pos] = old_exp[perm[pos]]。"""
    return {tuple(e[k] for k in perm): c for e, c in d.items()}


def elimination_basis(gens, eliminate_idx, n):
    """消元的独立期望：lex 基过滤后映回原变量坐标。

    与被测的 eliminate 刻意独立地实现同一条数学路径（消元定理），
    但复用本文件自己的朴素 Buchberger，且不导入被测代码。
    返回按 grlex 首项降序排列的 dict 列表（与被测输出顺序约定一致）。
    """
    elim = tuple(sorted(set(eliminate_idx)))
    rest = tuple(k for k in range(n) if k not in elim)
    perm = elim + rest
    inv = [0] * n
    for pos, k in enumerate(perm):
        inv[k] = pos

    work = [permute_poly(g, perm) for g in gens]
    basis = reduced_groebner(work, key=key_lex)
    out = []
    for b in basis:
        if all(all(e[p] == 0 for p in range(len(elim))) for e in b):
            out.append(permute_poly(b, inv))
    out.sort(key=lambda b: key_grlex(lm(b, key_grlex)), reverse=True)
    return out
