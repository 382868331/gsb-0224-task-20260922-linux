"""测试专用独立参考实现（不得 import 被测核心 groebner 来产生期望值）。

这里用最朴素的 dict[指数元组 -> Fraction] 与教科书算法独立实现：
  - 多项式加法/乘法/标量乘、S 多项式、教科书多元除法余式；
  - 无任何判据的朴素 Buchberger；
  - 独立的首一约化基与完整范式；
  - 纯 dict 的组合证书展开器。
与 groebner 包共用 fractions.Fraction（标准库数值类型，不属于被测核心）。
"""

from __future__ import annotations

from fractions import Fraction

TermMap = dict[tuple[int, ...], Fraction]


def clean(d: TermMap) -> TermMap:
    return {e: c for e, c in d.items() if c != 0}


def ref_add(a: TermMap, b: TermMap) -> TermMap:
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, Fraction(0)) + c
    return clean(out)


def ref_sub(a: TermMap, b: TermMap) -> TermMap:
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, Fraction(0)) - c
    return clean(out)


def ref_smul(a: TermMap, c: Fraction) -> TermMap:
    return clean({e: v * c for e, v in a.items()})


def ref_mul(a: TermMap, b: TermMap) -> TermMap:
    out: TermMap = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            e = tuple(i + j for i, j in zip(e1, e2))
            out[e] = out.get(e, Fraction(0)) + c1 * c2
    return clean(out)


def ref_deg(e: tuple[int, ...]) -> int:
    return sum(e)


def ref_lt(p: TermMap) -> tuple[tuple[int, ...], Fraction]:
    """grlex 首项：总次数最高，平局指数向量字典序最大。"""
    assert p
    e = max(p, key=lambda m: (ref_deg(m), m))
    return e, p[e]


def ref_divides(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(i <= j for i, j in zip(a, b))


def ref_s_poly(f: TermMap, g: TermMap) -> TermMap:
    mf, cf = ref_lt(f)
    mg, cg = ref_lt(g)
    L = tuple(max(i, j) for i, j in zip(mf, mg))
    sf = tuple(j - i for i, j in zip(mf, L))
    sg = tuple(j - i for i, j in zip(mg, L))
    left = ref_smul(ref_shift(f, sf), cg)
    right = ref_smul(ref_shift(g, sg), cf)
    return ref_sub(left, right)


def ref_shift(p: TermMap, delta: tuple[int, ...]) -> TermMap:
    return {tuple(a + b for a, b in zip(e, delta)): c for e, c in p.items()}


def ref_divrem(f: TermMap, ds: list[TermMap]) -> tuple[list[TermMap], TermMap]:
    """教科书多元除法（Cox Little O'Shea 版），返回 (商列表, 余式)。

    首项能整除当前首项就消，否则该首项移入余式。除数构成 GB 时余式为唯一范式。
    """
    qs = [{} for _ in ds]
    r: TermMap = {}
    p = dict(f)
    while p:
        m, c = ref_lt(p)
        hit = -1
        for i, d in enumerate(ds):
            md, _ = ref_lt(d)
            if ref_divides(md, m):
                hit = i
                break
        if hit >= 0:
            d = ds[hit]
            md, cd = ref_lt(d)
            delta = tuple(b - a for a, b in zip(md, m))
            gamma = c / cd
            mono = {delta: gamma}
            qs[hit] = ref_add(qs[hit], mono)
            p = ref_sub(p, ref_mul(mono, d))
        else:
            r = ref_add(r, {m: c})
            p = ref_sub(p, {m: c})
    return qs, r


def ref_rem(f: TermMap, ds: list[TermMap]) -> TermMap:
    return ref_divrem(f, ds)[1]


def ref_full_nf(p: TermMap, ds: list[TermMap]) -> TermMap:
    """完整范式：每步消去当前最大的“可约”项（独立于被测实现的写法）。"""
    work = dict(p)
    while work:
        target = None
        for m in sorted(work, key=lambda e: (ref_deg(e), e), reverse=True):
            if any(ref_divides(ref_lt(d)[0], m) for d in ds):
                target = m
                break
        if target is None:
            break
        c = work[target]
        for d in ds:
            md, cd = ref_lt(d)
            if ref_divides(md, target):
                delta = tuple(b - a for a, b in zip(md, target))
                mono = {delta: c / cd}
                work = ref_sub(work, ref_mul(mono, d))
                break
    return work


def ref_buchberger(gens: list[TermMap]) -> list[TermMap]:
    """无判据、无跳过的朴素 Buchberger（用于独立对照最终基）。"""
    G = [dict(g) for g in gens if g]

    def pairs(size: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(size) for j in range(i + 1, size)]

    pending = pairs(len(G))
    guard = 0
    while pending:
        i, j = pending.pop(0)
        s = ref_s_poly(G[i], G[j])
        _, r = ref_divrem(s, G)
        if r:
            G.append(r)
            k = len(G) - 1
            pending += [(t, k) for t in range(k)]
        guard += 1
        if guard > 100000:
            raise RuntimeError("参考实现步数异常")
    return G


def ref_monic(p: TermMap) -> TermMap:
    _, c = ref_lt(p)
    return ref_smul(p, Fraction(1) / c)


def ref_reduced_basis(gens: list[TermMap]) -> list[TermMap]:
    """独立计算唯一首一约化基：先朴素 Buchberger，再做首一约化。"""
    H = [ref_monic(g) for g in ref_buchberger(gens)]
    while True:
        changed = False
        i = 0
        while i < len(H):
            others = [h for k, h in enumerate(H) if k != i]
            r = ref_full_nf(H[i], others)
            if not r:
                del H[i]
                changed = True
                continue
            r = ref_monic(r)
            if r != H[i]:
                H[i] = r
                changed = True
            i += 1
        if not changed:
            break
    H.sort(key=lambda h: ref_lt(h)[0], reverse=True)
    return H


def ref_expand(combo: dict[int, TermMap], gens: list[TermMap]) -> TermMap:
    """纯 dict 展开 Σ combo[i] * gens[i]，用于独立证书检查。"""
    total: TermMap = {}
    for i, q in combo.items():
        total = ref_add(total, ref_mul(q, gens[i]))
    return total
