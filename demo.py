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

    # ------------------------------------------------- 2. 实际触发的拒绝边界
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

    # ----------------------------- 3. 预算耗尽（incomplete）≠ 证明非成员
    print("\n[预算语义] 极小 max_pairs 下：")
    tiny = groebner_basis(gens, max_pairs=0)
    print(f"  status={tiny.status}, reduced={tiny.reduced}")
    print(f"  reason={tiny.reason}")
    assert tiny.status == "incomplete" and not tiny.reduced
    probe = Poly({(3, 3): 1, (0, 0): 1}, n=2)
    mt = membership(probe, gens, max_pairs=0)
    print(f"  对 {show(probe)} 的判定状态为 {mt.status}（预算耗尽时不得宣称 nonmember）")
    assert mt.status in ("inconclusive", "member")

    # ----------------------------- 4. 两个理想的交集（消元变量 t）
    print("\n[理想交集] I = (x*(x+y), y*(x+y))，J = ((x+y)*(x-1))，求 I 与 J 的交：")
    i_gens = [
        Poly({(2, 0): 1, (1, 1): 1}, n=2),   # x*(x+y)
        Poly({(1, 1): 1, (0, 2): 1}, n=2),   # y*(x+y)
    ]
    j_gens = [Poly({(2, 0): 1, (1, 0): -1, (1, 1): 1, (0, 1): -1}, n=2)]  # (x+y)*(x-1)
    inter = ideal_intersection(i_gens, j_gens)
    print(f"  状态={inter.status}（消元变量 t 的块次序仅在该入口内部使用）")
    assert inter.status == "complete"
    for k, b in enumerate(inter.basis):
        print(f"  交集生成元 h{k} = {show(b)}")

    print("\n[交集双向证书核验] 每个 h_k 分别展开为 I、J 生成元的组合：")
    for k, (b, ci, cj) in enumerate(
            zip(inter.basis, inter.certificates_i, inter.certificates_j)):
        ok_i = expand(ci, i_gens) == b
        ok_j = expand(cj, j_gens) == b
        print(f"  h{k}: 属于 I 的证书展开 {'一致' if ok_i else '不一致'}，"
              f"属于 J 的证书展开 {'一致' if ok_j else '不一致'}")
        assert ok_i and ok_j

    print("\n[交集成员复核] 每个 h_k 对 I、J 各自的 Groebner 基余式都为 0：")
    gb_i = groebner_basis(i_gens)
    gb_j = groebner_basis(j_gens)
    for k, b in enumerate(inter.basis):
        ri = membership(b, i_gens)
        rj = membership(b, j_gens)
        print(f"  h{k}: 对 I 余式={show(ri.remainder)}，对 J 余式={show(rj.remainder)}")
        assert ri.status == "member" and rj.status == "member"
    assert gb_i.is_complete and gb_j.is_complete

    print("\n[交集边界] 与零理想的交为空基；与单位理想的交等于另一侧：")
    z = ideal_intersection([Poly.zero(2)], i_gens)
    u = ideal_intersection([Poly.constant(1, 2)], i_gens)
    print(f"  (0) ∩ I：状态={z.status}，生成元个数={len(z.basis)}")
    print(f"  (1) ∩ I：状态={u.status}，生成元={[show(b) for b in u.basis]}")
    assert z.basis == () and u.status == "complete"

    tiny_i = ideal_intersection(i_gens, j_gens, max_pairs=0)
    print(f"  零预算下：status={tiny_i.status}（只报 incomplete，不自称完整交集）")
    assert tiny_i.status == "incomplete" and not tiny_i.reduced

    print("\n全部演示计算完成。")


if __name__ == "__main__":
    main()
