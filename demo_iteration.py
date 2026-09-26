"""bounded_syzygies 的现场演示：真实计算，不使用 sleep，不预录结果。

运行：python demo_iteration.py（远低于 90 秒，实测约 0.1 秒）。

演示内容：
1. 正常结果：(x, y) 在 d=1 求出 Koszul 关系 (y, -x)，并逐条独立展开核验；
2. 正常结果：三个二次生成元的两条 Koszul 关系；
3. 边界：d 太小（d=0）时精确证明不存在非平凡关系，不捏造；
4. 边界：零生成元 / 重复生成元 / 常数生成元的处理；
5. 边界：非法 max_degree（bool）被可定位地拒绝。
"""

from __future__ import annotations

import time

from groebner import (
    InvalidGeneratorError,
    Poly,
    bounded_syzygies,
    relation_polynomial,
)

NAMES = ("x", "y")


def show_relation(coeffs, gens):
    lhs = " + ".join(
        f"({a.to_string(NAMES)})*({f.to_string(NAMES)})"
        for a, f in zip(coeffs, gens))
    total = relation_polynomial(coeffs, gens)
    print(f"    {lhs}")
    print(f"    独立展开求和 = {total.to_string(NAMES)}  （必须为 0）")


def main():
    started = time.perf_counter()
    x = Poly.variable(2, 0)
    y = Poly.variable(2, 1)

    print("=== 1. 正常：f = (x, y)，d = 1 ===")
    result = bounded_syzygies([x, y], 1)
    print(f"  未知数 {result.num_unknowns}，方程 {result.num_equations}，"
          f"秩 {result.rank}，解空间维数 {result.dimension}")
    print(f"  维数公式核验：{result.dimension} == "
          f"{result.num_unknowns} - {result.rank} -> "
          f"{result.dimension == result.num_unknowns - result.rank}")
    for coeffs in result.basis:
        show_relation(coeffs, result.generators)

    print("=== 2. 正常：f = (x**2, x*y, y**2)，d = 1 ===")
    gens2 = [x * x, x * y, y * y]
    result2 = bounded_syzygies(gens2, 1)
    print(f"  解空间维数 {result2.dimension}（两条 Koszul 关系）")
    for coeffs in result2.basis:
        show_relation(coeffs, gens2)

    print("=== 3. 边界：d = 0 太小，精确证明无关系 ===")
    result3 = bounded_syzygies([x, y], 0)
    print(f"  维数 {result3.dimension}，基 {result3.basis}（空基 = 不存在，"
          f"不是没找到）")

    print("=== 4. 边界：零 / 重复 / 常数生成元 ===")
    zero = Poly.zero(2)
    r_zero = bounded_syzygies([zero, x], 1)
    print(f"  [0, x]：维数 {r_zero.dimension}（零生成元的系数列全部自由）")
    r_dup = bounded_syzygies([x, x], 1)
    print(f"  [x, x]：维数 {r_dup.dimension}，含常数关系 (1, -1)：")
    for coeffs in r_dup.basis:
        if all(sum(e) == 0 for c in coeffs for e in c.terms):
            show_relation(coeffs, [x, x])
    one = Poly.constant(1, 2)
    r_const = bounded_syzygies([one, x], 1)
    print(f"  [1, x]：维数 {r_const.dimension}（常数生成元迫使 "
          f"a0 = -a1*x）")
    for coeffs in r_const.basis:
        show_relation(coeffs, [one, x])

    print("=== 5. 边界：非法 max_degree 被拒绝 ===")
    try:
        bounded_syzygies([x, y], True)
    except InvalidGeneratorError as exc:
        print(f"  拒绝 bool 次数：{exc}")

    elapsed = time.perf_counter() - started
    print(f"=== 全部真实计算完成，用时 {elapsed:.3f} 秒（预算 90 秒）===")


if __name__ == "__main__":
    main()
