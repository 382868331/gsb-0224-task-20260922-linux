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


def key(exp):
    """grlex 排序键：总次数优先，并列按指数向量字典序（x1 优先）。"""
    return (sum(exp), exp)


def lm(a):
    return max(a, key=key)


def lc(a):
    return a[lm(a)]


def divides(u, v):
    return all(x <= y for x, y in zip(u, v))


def lcm_exp(u, v):
    return tuple(max(x, y) for x, y in zip(u, v))


def monom(n, exp, coef=Fraction(1)):
    return {tuple(exp): Fraction(coef)}


# ------------------------------------------------------------------- 除法/约化

def remainder(f, divisors):
    """f 对除式列表反复取首项约化得到的余式（依赖除式顺序，与教科书一致）。"""
    r = {}
    p = f
    while p:
        e, c = lm(p), lc(p)
        hit = None
        for g in divisors:
            if divides(lm(g), e):
                hit = g
                break
        if hit is None:
            r = add(r, monom(len(e), e, c))
            p = sub(p, monom(len(e), e, c))
        else:
            ge, gc = lm(hit), lc(hit)
            shift = tuple(a - b for a, b in zip(e, ge))
            qterm = monom(len(e), shift, c / gc)
            p = sub(p, mul(qterm, hit))
    return r


def divide_full(f, divisors):
    """返回 (商列表, 余式)，满足 f = sum q_i g_i + r。"""
    if f:
        n = len(lm(f))
    elif divisors:
        n = len(lm(divisors[0]))
    else:
        n = 0
    qs = [{} for _ in divisors]
    r = {}
    p = f
    while p:
        e, c = lm(p), lc(p)
        hit = None
        for i, g in enumerate(divisors):
            if divides(lm(g), e):
                hit = i
                break
        if hit is None:
            r = add(r, monom(n, e, c))
            p = sub(p, monom(n, e, c))
        else:
            g = divisors[hit]
            shift = tuple(a - b for a, b in zip(e, lm(g)))
            qterm = monom(n, shift, c / lc(g))
            qs[hit] = add(qs[hit], qterm)
            p = sub(p, mul(qterm, g))
    return qs, r


# ------------------------------------------------------------------- S 多项式

def s_poly(f, g):
    l = lcm_exp(lm(f), lm(g))
    tf = monom(len(l), tuple(a - b for a, b in zip(l, lm(f))), Fraction(1) / lc(f))
    tg = monom(len(l), tuple(a - b for a, b in zip(l, lm(g))), Fraction(1) / lc(g))
    return sub(mul(tf, f), mul(tg, g))


def _monic(f):
    return smul(Fraction(1) / lc(f), f)


# --------------------------------------------------------------- Buchberger

def reduced_groebner(gens):
    """朴素 Buchberger + 首一约化，返回按首项降序的 dict 列表。

    与被测实现刻意不同：不做首项互素跳过、不用队列，
    每轮对当前所有对取余式（余式按**活动基**约化，保证新首项严格
    扩大首项理想，过程必终止），直到一轮没有新元素。
    """
    basis = []
    for g in gens:
        if not g:
            continue
        r = remainder(g, basis)
        if r:
            basis.append(_monic(r))

    while True:
        added = False
        size = len(basis)
        for i in range(size):
            for j in range(i + 1, size):
                r = remainder(s_poly(basis[i], basis[j]), basis)
                if r:
                    basis.append(_monic(r))
                    added = True
        if not added:
            break

    # 同首项只保留一个（极小化前先去重，避免重复首项互相“整除”）
    uniq, seen = [], set()
    for b in basis:
        if lm(b) not in seen:
            seen.add(lm(b))
            uniq.append(b)

    # 极小化：首项被其他不同的首项严格整除者删除
    minimal = [
        b for b in uniq
        if not any(b is not k and lm(b) != lm(k) and divides(lm(k), lm(b))
                   for k in uniq)
    ]

    # 首一约化到不动点
    reduced = [dict(b) for b in minimal]
    changed = True
    while changed:
        changed = False
        for pos in range(len(reduced)):
            others = reduced[:pos] + reduced[pos + 1:]
            r = remainder(reduced[pos], others)
            r = _monic(r) if r else r
            if r != reduced[pos]:
                reduced[pos] = r
                changed = True

    reduced.sort(key=lambda b: key(lm(b)), reverse=True)
    return reduced


def ideal_remainder(f, gens):
    """成员判定的独立期望：对独立算出的约化基取余式。"""
    basis = reduced_groebner(gens)
    if not basis:
        return dict(f)
    return remainder(f, basis)


def expand(cert_dicts, gen_dicts):
    """独立展开证书：sum_j cert_j * gen_j，返回 dict 多项式。"""
    total = {}
    for c, g in zip(cert_dicts, gen_dicts):
        total = add(total, mul(c, g))
    return total


# ------------------------------------------------- 独立线性代数（syzygy 对照）

def monomials_le(n, d):
    """独立枚举次数 <= d 的单项式指数（递归另一写法），grlex 降序。"""
    out = []

    def rec(pos, rem, acc):
        if pos == n:
            out.append(tuple(acc))
            return
        for e in range(rem + 1):
            rec(pos + 1, rem - e, acc + [e])

    rec(0, d, [])
    out.sort(key=key, reverse=True)
    return out


def _rref_independent(matrix):
    """独立另写的精确行最简消元，返回 (rref, 主元列列表)。"""
    rows = [list(r) for r in matrix]
    ncols = len(rows[0]) if rows else 0
    pivots = []
    r = 0
    for c in range(ncols):
        piv = None
        for k in range(r, len(rows)):
            if rows[k][c] != 0:
                piv = k
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = Fraction(1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for k in range(len(rows)):
            if k != r and rows[k][c] != 0:
                f = rows[k][c]
                rows[k] = [v - f * w for v, w in zip(rows[k], rows[r])]
        pivots.append(c)
        r += 1
    return rows, pivots


def bounded_syzygy_expectation(gen_dicts, n, d):
    """独立重建有界 syzygy 线性系统，返回 (秩, 未知量数, 零空间基向量列表)。

    基向量是长度 = 未知量数的 Fraction 列表，列序为
    (生成元下标 i, 单项式指数)，单项式按 grlex 降序——与被测库约定相同，
    但枚举、建矩阵、消元全部在此独立实现。
    """
    mons = monomials_le(n, d)
    m = len(gen_dicts)
    num = m * len(mons)
    equations = {}
    for i, g in enumerate(gen_dicts):
        for j, exp in enumerate(mons):
            col = i * len(mons) + j
            for e_f, c in g.items():
                beta = tuple(a + b for a, b in zip(exp, e_f))
                row = equations.setdefault(beta, {})
                row[col] = row.get(col, Fraction(0)) + c
    matrix = []
    for beta in sorted(equations, key=key, reverse=True):
        row = [Fraction(0)] * num
        for col, c in equations[beta].items():
            row[col] = c
        matrix.append(row)
    rref, pivots = _rref_independent(matrix)
    pivot_set = set(pivots)
    basis = []
    for free in range(num):
        if free in pivot_set:
            continue
        vec = [Fraction(0)] * num
        vec[free] = Fraction(1)
        for row_idx, pcol in enumerate(pivots):
            vec[pcol] = -rref[row_idx][free]
        lead = next(v for v in vec if v != 0)
        if lead != 1:
            vec = [v / lead for v in vec]
        basis.append(vec)
    return len(pivots), num, basis
