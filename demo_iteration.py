"""saturate（I:f^inf 饱和理想）的固定输入现场演示：真实计算，不用 sleep、不预录。

运行：``python demo_iteration.py``（正常与边界结果合计约 1 秒内，远低于 90 秒）。

演示内容：
1. 正常结果：xy=0 排除 x=0 分量 -> y=0；x**2*y 排除 x=0 -> y=0
   （并展示一次理想商 (x**2*y):x = (x*y) 并不是饱和）；
   每个输出多项式打印有限 k 与 f**k*g 属于原理想的证书，并独立展开核验。
2. 边界结果：f=0 -> 单位理想；非零常数 f -> 原理想；
   零理想 -> 零理想；单位理想 -> 单位理想。
3. 预算耗尽 -> incomplete（与正常结论严格区分）。
"""

from __future__ import annotations

from groebner import Poly, divide, saturate

NAMES = ("x", "y")


def show(p: Poly) -> str:
    return p.to_string(NAMES)


def pow_poly(base: Poly, k: int) -> Poly:
    out = Poly.constant(1, base.n)
    for _ in range(k):
        out = out * base
    return out


def expand(cert, gens) -> Poly:
    """证书独立展开：sum_j cert[j] * gens[j]（只用多项式算术）。"""
    total = Poly.zero(gens[0].n) if gens else Poly.zero(2)
    for c, g in zip(cert, gens):
        total = total + c * g
    return total


def report(title: str, gens, f: Poly) -> None:
    print("-" * 70)
    print(title)
    print("  I = (" + ", ".join(show(g) for g in gens if not g.is_zero()) + ")"
          if any(not g.is_zero() for g in gens) else "  I = (0)（零理想）")
    print(f"  f = {show(f)}")
    r = saturate(gens, f)
    if r.status != "complete":
        print(f"  结果：incomplete（{r.reason}）")
        return
    if not r.basis:
        print("  饱和理想 = 零理想 (0)")
        return
    print("  饱和理想 I:f^inf 的首一约化基：")
    for b, k, cert in zip(r.basis, r.powers, r.certificates):
        lhs = pow_poly(f, k) * b
        rhs = expand(cert, gens)
        ok = "OK" if lhs == rhs else "不符！"
        print(f"    g = {show(b)}，k = {k}，证书展开核验 f**k*g == sum cert_j*I_j：{ok}")
        print(f"      f**k*g = {show(lhs)}")


def main() -> None:
    x = Poly.variable(2, 0)
    y = Poly.variable(2, 1)

    print("=" * 70)
    print("饱和理想 I:f^inf 演示（Rabinowitsch 技巧 + 消元序，精确有理数）")
    print("=" * 70)

    print("[1] 正常结果：分量去除")
    report("xy = 0 的零点集是两条坐标轴，排除 x = 0 分量后剩 y = 0：",
           [x * y], x)
    report("x**2*y = 0：一次理想商 (x**2*y):x = (x*y) 仍含 x 分量，",
           [x * x * y], x)
    # 对照：一次商确实不是饱和
    q = divide(x * x * y, [x]).quotients[0]
    print(f"  对照：一次理想商生成元为 {show(q)}，而饱和基为 y——二者不同。")

    print()
    print("[2] 边界结果")
    report("f = 0：排除 f=0 即排除整个空间，剩余为空集，理想为 (1)：",
           [x * y], Poly.zero(2))
    report("f 为非零常数：常数是单位，饱和 = 原理想本身：",
           [x * y], Poly.constant(3, 2))
    report("零理想被非零 f 饱和仍是零理想：",
           [Poly.zero(2)], x)
    report("单位理想饱和仍是单位理想：",
           [Poly.constant(1, 2)], x)

    print()
    print("[3] 预算耗尽：incomplete（不宣称任何结论）")
    r = saturate([x * y], x, max_steps=0)
    print(f"  status = {r.status}，reason = {r.reason}")

    print()
    print("全部演示均为现场真实计算完成。")


if __name__ == "__main__":
    main()

