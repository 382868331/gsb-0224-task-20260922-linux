"""bounded_syzygies 的现场演示：真实计算，不用 sleep，不预录结果。

运行：``python demo_iteration.py``（远低于 90 秒）。

演示内容：
1. 正常结果：[x, y] 在 d=1 找到规范关系 (y, -x)；
   [x**2 - y, x*y - 1] 在 d=2 找到非平凡关系；
   每条关系独立逐多项式相乘求和核验为 0，并用矩阵秩等式说明维数完备。
2. 边界结果：d 太小时没有关系（不捏造）；零生成元、重复生成元、
   常数生成元；非法次数上限被可定位地拒绝。
"""

from __future__ import annotations

from groebner import (
    InvalidGeneratorError,
    Poly,
    bounded_syzygies,
    relation_value,
)

NAMES = ("x", "y")


def show(p: Poly) -> str:
    return p.to_string(NAMES)


def show_relation(rel) -> str:
    return "(" + ", ".join(show(a) for a in rel) + ")"


def report(gens, d, label):
    res = bounded_syzygies(gens, d)
    print(f"\n[{label}] 生成元 = [{', '.join(show(g) for g in gens)}]，d = {d}")
    print(f"  未知量 {res.num_unknowns} 个，系数矩阵秩 {res.rank}，"
          f"解空间维数 = {res.num_unknowns} - {res.rank} = {res.dimension}")
    assert res.dimension == res.num_unknowns - res.rank  # 秩等式 = 维数完备
    if not res.basis:
        print("  该次数上限下不存在非平凡关系（维数 0，不捏造）。")
        return res
    for k, rel in enumerate(res.basis):
        value = relation_value(rel, gens)  # 独立逐多项式相乘求和
        print(f"  关系 {k}: {show_relation(rel)}")
        print(f"    独立展开 sum a_i*f_i = {show(value)}   "
              f"[{'为零，一致' if value.is_zero() else '非零，失败'}]")
        assert value.is_zero()
    return res


def main() -> None:
    print("=" * 70)
    print("次数有界 syzygy 演示：sum a_i*f_i = 0, deg a_i <= d（精确有理数）")
    print("=" * 70)

    x = Poly.variable(2, 0)
    y = Poly.variable(2, 1)

    # ------------------------------------------------ 1. 正常结果
    report([x, y], 1, "正常：x, y")
    f1 = Poly({(2, 0): 1, (0, 1): -1}, n=2)   # x**2 - y
    f2 = Poly({(1, 1): 1, (0, 0): -1}, n=2)   # x*y - 1
    report([f1, f2], 2, "正常：x**2 - y, x*y - 1")

    # ------------------------------------------------ 2. 边界结果
    report([x, y], 0, "边界：d 太小")
    report([Poly.zero(2), x], 1, "边界：含零生成元")
    report([f1, f1], 0, "边界：重复生成元")
    report([Poly.constant(2, 2), x], 1, "边界：常数生成元")

    print("\n[边界：非法输入拒绝] 次数上限 d 为 bool/负数必须被拒绝：")
    for bad in (True, -1):
        try:
            bounded_syzygies([x, y], bad)
        except InvalidGeneratorError as exc:
            print(f"  d={bad!r} -> 捕获 {type(exc).__name__}: {exc}")
        else:
            raise AssertionError(f"d={bad!r} 未被拒绝")

    print("\n全部演示计算完成。")


if __name__ == "__main__":
    main()
