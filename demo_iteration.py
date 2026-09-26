"""saturate(I, f) 固定输入演示：正常情形与边界情形，全部真实计算。

运行：python demo_iteration.py
"""

from groebner import Poly, saturate


def show(title, result, names):
    print(f"== {title} ==")
    print(f"状态: {result.status}")
    if result.reason:
        print(f"说明: {result.reason}")
    if not result.basis:
        print("饱和理想的基: (空，零理想)" if result.is_complete else "基: 无")
    for i, b in enumerate(result.basis):
        cert = result.certificates[i]
        if cert is None:
            note = "证书超出预算未给出"
        else:
            note = f"k={cert.k}，即 f^{cert.k}*g 属于原理想（证书可独立展开核验）"
        print(f"  g{i + 1} = {b.to_string(names)}    [{note}]")
    print()


def main():
    x = Poly.variable(2, 0)
    y = Poly.variable(2, 1)

    # 正常 1：xy = 0 排除 x = 0 的分量，只剩 y = 0。
    show("I = (x*y), f = x：排除 x=0 分量", saturate([x * y], x), ("x", "y"))

    # 正常 2：一次理想商不够——(x^2*y):x = (x*y) 仍含 x=0 分量，
    # 无限次饱和才得到 (y)。
    show("I = (x^2*y), f = x：需要 k=2 的真正饱和",
         saturate([x * x * y], x), ("x", "y"))

    # 正常 3：两个分量的交，(x*z, y*z) = (z) 交 (x, y)，排除 z=0 剩 (x, y)。
    z = Poly.variable(3, 2)
    x3 = Poly.variable(3, 0)
    y3 = Poly.variable(3, 1)
    show("I = (x*z, y*z), f = z：保留 z!=0 上的约束",
         saturate([x3 * z, y3 * z], z), ("x", "y", "z"))

    # 边界 1：f = 0，任何 g 都满足 f*g = 0 属于 I，饱和为单位理想。
    show("边界：f = 0，饱和为单位理想 (1)",
         saturate([x * y], Poly.zero(2)), ("x", "y"))

    # 边界 2：f 为非零常数（可逆元），饱和等于原理想。
    show("边界：f = 3 非零常数，饱和等于原理想",
         saturate([x * y], Poly.constant(3, 2)), ("x", "y"))

    # 边界 3：零理想在整环中饱和仍为零理想。
    show("边界：零理想，饱和仍为零理想", saturate([], x), ("x", "y"))

    # 边界 4：单位理想饱和仍为单位理想。
    show("边界：单位理想，饱和仍为单位理想",
         saturate([Poly.constant(1, 2), x], x), ("x", "y"))

    # 边界 5：零预算，incomplete 与“证明结果”严格分开。
    show("边界：零预算，返回 incomplete",
         saturate([x * y], x, max_pairs=0, max_terms=0, max_steps=0),
         ("x", "y"))


if __name__ == "__main__":
    main()

