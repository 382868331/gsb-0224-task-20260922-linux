"""固定输入的现场演示：真实计算，不使用 sleep，不预录结果。

运行：``python demo.py``（约 1 秒内完成）。

演示内容：
1. 正常结果：圆与抛物线理想
   g1 = x**2 + y**2 - 1, g2 = x - y**2
   - 计算 grlex 下的首一约化 Groebner 基；
   - 独立展开每个基多项式的组合证书，逐字核验；
   - 核验最终基的每一对 S 多项式余式都是 0；
   - 展示消元关系 y**4 + y**2 - 1（只含 y）属于该理想并打印其证书；
   - 成员 / 非成员判定与余式。
2. 实际触发的拒绝边界：NaN 系数被拒绝（抛 InvalidPolynomialError）。
3. 预算耗尽返回 incomplete，与“证明非成员”严格区分。
"""

from __future__ import annotations

import math

from groebner import (
    InvalidPolynomialError,
    Poly,
    check_s_polynomials,
    groebner_basis,
    ideal_intersection,
    membership,
)

NAMES = ("x", "y")


def show(p: Poly) -> str:
    return p.to_string(NAMES)


def expand(cert, gens) -> Poly:
    """证书独立展开：sum_j cert[j] * g_j（只用多项式算术，不复用算法结论）。"""
    total = Poly.zero(gens[0].n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def main() -> None:
    print("=" * 70)
    print("多变量多项式的约化 Groebner 基演示（有理数系数，graded lex 次序）")
    print("=" * 70)

    # ---------------------------------------------------------- 1. 正常结果
    g1 = Poly({(2, 0): 1, (0, 2): 1, (0, 0): -1}, n=2)  # x**2 + y**2 - 1
    g2 = Poly({(1, 0): 1, (0, 2): -1}, n=2)             # x - y**2
    gens = [g1, g2]
    print("\n[输入生成元]")
    print(f"  g1 = {show(g1)}")
    print(f"  g2 = {show(g2)}")

    res = groebner_basis(gens)
    assert res.status == "complete"
    print(f"\n[首一约化 Groebner 基]  状态={res.status}，"
          f"处理关键对 {res.pairs_processed} 个"
          f"（其中首项互素直接跳过 {res.pairs_skipped_coprime} 个）")
    for i, b in enumerate(res.basis):
        print(f"  b{i} = {show(b)}")

    print("\n[组合证书核验] 对每个 b_i 独立展开 b_i = c1*g1 + c2*g2：")
    for i, (b, cert) in enumerate(zip(res.basis, res.certificates)):
        rebuilt = expand(cert, gens)
        ok = rebuilt == b
        print(f"  b{i}: ({show(cert[0])})*g1 + ({show(cert[1])})*g2"
              f" = {show(rebuilt)}   [{'一致' if ok else '不一致'}]")
        assert ok

    bad_pairs = check_s_polynomials(list(res.basis))
    print(f"\n[Buchberger 判据核验] 最终基所有 S 多项式余式为 0："
          f"{'通过' if not bad_pairs else '失败'}（检查了 "
          f"{len(res.basis) * (len(res.basis) - 1) // 2} 对）")
    assert not bad_pairs

    # 消元关系：由 x = y**2 代入 x**2 + y**2 = 1 得 y**4 + y**2 - 1 = 0
    elim = Poly({(0, 4): 1, (0, 2): 1, (0, 0): -1}, n=2)
    m = membership(elim, gens)
    print("\n[消元关系]")
    print(f"  询问 f = {show(elim)} 是否属于理想：{m.status}，余式 = {show(m.remainder)}")
    assert m.status == "member"
    c1, c2 = m.certificate
    print(f"  组合证书：f = ({show(c1)})*g1 + ({show(c2)})*g2")
    assert expand(m.certificate, gens) == elim
    print("  独立展开证书与 f 完全一致（精确有理数，无浮点容差）。")

    # 非成员
    outsider = Poly({(1, 0): 1, (0, 0): 1}, n=2)  # x + 1
    mo = membership(outsider, gens)
    print("\n[非成员判定]")
    print(f"  h = {show(outsider)}：{mo.status}，非零余式 = {show(mo.remainder)}")
    assert mo.status == "nonmember"

    # --------------------------------------------- 2. 两个有限生成理想的交
    print("\n" + "=" * 70)
    print("两个理想的交：内部 t 消元，输出原变量环的双组合证书")
    print("=" * 70)
    xv = Poly.variable(1, 0)
    one1 = Poly.constant(1, 1)
    i_gen = (xv - one1) * (xv - one1)
    j_gen = (xv - one1) * (xv + one1)
    inter = ideal_intersection([i_gen], [j_gen])
    assert inter.status == "complete"
    print("  I = <(x-1)**2>, J = <(x-1)(x+1)>")
    for h, (cert_i, cert_j) in zip(inter.basis, inter.certificates):
        rebuilt_i = expand(cert_i, [i_gen])
        rebuilt_j = expand(cert_j, [j_gen])
        print(f"  交集生成元 h = {h.to_string(('x',))}")
        print(f"    I 侧证书：h = ({cert_i[0].to_string(('x',))}) * (x-1)**2 "
              f"= {rebuilt_i.to_string(('x',))}")
        print(f"    J 侧证书：h = ({cert_j[0].to_string(('x',))}) * "
              f"(x-1)(x+1) = {rebuilt_j.to_string(('x',))}")
        assert rebuilt_i == h and rebuilt_j == h
    tiny_inter = ideal_intersection([i_gen], [j_gen], max_pairs=0)
    print(f"  max_pairs=0 时状态：{tiny_inter.status}（basis="
          f"{tuple(map(str, tiny_inter.basis))}，不发布部分不含 t 项）")
    assert tiny_inter.status == "incomplete" and not tiny_inter.basis

    # ------------------------------------------------- 3. 实际触发的拒绝边界
    print("\n[输入拒绝边界] 系数为 NaN 必须被拒绝（不静默纠正）：")
    try:
        Poly({(1, 0): math.nan}, n=2)
    except InvalidPolynomialError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 系数未被拒绝")

    # 再演示一个可定位错误：bool 不允许冒充整数
    try:
        Poly({(True, 0): 1}, n=2)
    except InvalidPolynomialError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("bool 指数未被拒绝")

    # ----------------------------- 4. 预算耗尽（incomplete）≠ 证明非成员
    print("\n[预算语义] 极小 max_pairs 下：")
    tiny = groebner_basis(gens, max_pairs=0)
    print(f"  status={tiny.status}, reduced={tiny.reduced}")
    print(f"  reason={tiny.reason}")
    assert tiny.status == "incomplete" and not tiny.reduced
    probe = Poly({(3, 3): 1, (0, 0): 1}, n=2)
    mt = membership(probe, gens, max_pairs=0)
    print(f"  对 {show(probe)} 的判定状态为 {mt.status}（预算耗尽时不得宣称 nonmember）")
    assert mt.status in ("inconclusive", "member")

    print("\n全部演示计算完成。")


if __name__ == "__main__":
    main()
