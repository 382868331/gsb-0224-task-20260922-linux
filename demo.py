"""固定输入的现场演示：真实计算，不使用 sleep，不预录结果。

运行：``python demo.py``（1 秒内完成）。

演示内容：
1. 正常结果：圆与抛物线理想
   g1 = x**2 + y**2 - 1, g2 = x - y**2
   - 计算 grlex 下的首一约化 Groebner 基；
   - 独立展开每个基多项式的组合证书，逐字核验；
   - 核验最终基的每一对 S 多项式余式都是 0；
   - 展示消元关系 y**4 + y**2 - 1（只含 y）属于该理想并打印其证书；
   - 成员 / 非成员判定与余式。
2. lex 次序与消元（本轮新增）：
   - 同一系统的 lex 基，用消元入口 eliminate 消去 x，直接得到
     不含 x 的消元基 {y**4 + y**2 - 1}（手算核对）；
   - 三变量代入链 x**2=y, y**2=z, z=1：消去 x 得 {y**2-1, z-1}，
     并演示消去“非前缀”变量 z 时坐标自动处理；
   - 参数曲线 x=t, y=t**2 消去 t 得隐式方程 x**2 - y；
   - 每个消元基元素的组合证书独立展开核验。
3. 实际触发的拒绝边界：NaN 系数与非法消去下标被拒绝（抛异常）。
4. 预算耗尽返回 incomplete，消元入口不发布半成品，与“证明无解”严格区分。
"""

from __future__ import annotations

import math

from groebner import (
    InvalidGeneratorError,
    InvalidPolynomialError,
    Poly,
    check_s_polynomials,
    eliminate,
    groebner_basis,
    membership,
)

NAMES = ("x", "y")


def show(p: Poly) -> str:
    return p.to_string(NAMES)


def show3(p: Poly) -> str:
    return p.to_string(("x", "y", "z"))


def expand(cert, gens) -> Poly:
    """证书独立展开：sum_j cert[j] * g_j（只用多项式算术，不复用算法结论）。"""
    total = Poly.zero(gens[0].n)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def main() -> None:
    print("=" * 70)
    print("约化 Groebner 基演示（有理数系数；graded lex 与 lex 消元）")
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

    # --------------------------------------------- 2. lex 次序与消元（新增）
    print("\n" + "-" * 70)
    print("[lex 次序] 同一系统 x1 > x2 字典序下的首一约化基：")
    rl = groebner_basis(gens, order="lex")
    assert rl.status == "complete" and rl.order == "lex"
    for i, b in enumerate(rl.basis):
        print(f"  b{i} = {show(b)}")
    assert check_s_polynomials(list(rl.basis), order="lex") == []

    print("\n[消元 eliminate] 消去 x（下标 0），直接返回不含 x 的基元素：")
    e = eliminate(gens, [0])
    assert e.status == "complete"
    for b, cert in zip(e.elimination_basis, e.certificates):
        print(f"  消元基元素 = {show(b)}")
        rebuilt = expand(cert, gens)
        print(f"    证书展开 = ({show(cert[0])})*g1 + ({show(cert[1])})*g2"
              f" = {show(rebuilt)}   [{'一致' if rebuilt == b else '不一致'}]")
        assert rebuilt == b
    # 手算核对：x = y**2 代入 x**2 + y**2 = 1 得 y**4 + y**2 - 1
    assert [b.terms for b in e.elimination_basis] == [
        {(0, 4): 1, (0, 2): 1, (0, 0): -1}]

    print("\n[三变量代入链] g1 = x**2 - y, g2 = y**2 - z, g3 = z - 1")
    a = Poly({(2, 0, 0): 1, (0, 1, 0): -1}, n=3)
    b = Poly({(0, 2, 0): 1, (0, 0, 1): -1}, n=3)
    c = Poly({(0, 0, 1): 1, (0, 0, 0): -1}, n=3)
    chain = [a, b, c]
    ex = eliminate(chain, [0])  # 消去 x
    print("  消去 x：" + ", ".join(show3(q) for q in ex.elimination_basis))
    assert [q.terms for q in ex.elimination_basis] == [
        {(0, 2, 0): 1, (0, 0, 0): -1},
        {(0, 0, 1): 1, (0, 0, 0): -1},
    ]
    ez = eliminate(chain, [2])  # 消去非前缀变量 z
    print("  消去 z（非前缀，坐标自动处理）："
          + ", ".join(show3(q) for q in ez.elimination_basis))
    assert [q.terms for q in ez.elimination_basis] == [
        {(2, 0, 0): 1, (0, 1, 0): -1},
        {(0, 2, 0): 1, (0, 0, 0): -1},
    ]
    for q, cert in zip(ez.elimination_basis, ez.certificates):
        assert expand(cert, chain) == q

    print("\n[隐式化] 参数曲线 x = t, y = t**2：g1 = x - t, g2 = y - t**2")
    gx = Poly({(1, 0, 0): 1, (0, 0, 1): -1}, n=3)   # x - t
    gy = Poly({(0, 1, 0): 1, (0, 0, 2): -1}, n=3)   # y - t**2
    et = eliminate([gx, gy], [2])  # 消去 t（下标 2）
    print("  消去 t 得隐式方程：" + ", ".join(
        q.to_string(("x", "y", "t")) for q in et.elimination_basis))
    assert [q.terms for q in et.elimination_basis] == [
        {(2, 0, 0): 1, (0, 1, 0): -1}]
    for q, cert in zip(et.elimination_basis, et.certificates):
        assert expand(cert, [gx, gy]) == q
    print("  证书独立展开核验一致（精确有理数，无浮点容差）。")

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

    # 消元入口自身的参数错误：越界下标 / 裸 int / 未知次序
    try:
        eliminate(gens, [5])
    except InvalidGeneratorError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("越界消去下标未被拒绝")
    try:
        eliminate(gens, 0)
    except InvalidGeneratorError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("裸 int 消去下标未被拒绝")
    try:
        groebner_basis(gens, order="lexx")
    except InvalidGeneratorError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("未知次序未被拒绝")

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

    # 消元入口在预算耗尽时同样不允许发布“半成品”消元基
    et_bad = eliminate(gens, [0], max_pairs=0)
    print(f"  eliminate 预算耗尽：status={et_bad.status}，"
          f"消元基长度={len(et_bad.elimination_basis)}（必须为 0，不冒充完成）")
    assert et_bad.status == "incomplete"
    assert et_bad.elimination_basis == ()

    print("\n全部演示计算完成。")


if __name__ == "__main__":
    main()
