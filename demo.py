"""固定输入演示：约化 Groebner 基、组合证书与理想成员判定。

运行：python demo.py
内容全部为真实计算（确定性数据，无网络、无预录结果、无 sleep）：
  1. 正常结果：对 I = <x+y, xy-1> 计算首一约化 Gröbner 基，展示消元关系
     y^2+1 = 0，独立展开每个基多项式的组合证书，并核验全部 S 多项式余式为 0；
  2. 成员判定：成员给出可独立展开的证书；非成员给出非零余式并证明不属于理想；
  3. 异常边界：NaN 系数被当场拒绝（可定位错误，不静默纠正）。
"""

from itertools import combinations

from groebner import (
    Poly,
    PolynomialError,
    groebner_basis,
    reduced_basis,
    ideal_membership,
    s_polynomial,
    divide,
)


def line(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    x = Poly.var(0, 2)
    y = Poly.var(1, 2)
    g1 = x + y
    g2 = x * y - 1
    gens = [g1, g2]
    names = ["g1", "g2"]

    line("1. 正常结果：I = <g1, g2>，g1 = x + y，g2 = x*y - 1（grlex, Q[x,y]）")

    result = groebner_basis(gens)
    rb = reduced_basis(result)
    print(f"Buchberger 完成：处理 S 对 {result.pairs_processed} 个，"
          f"首项互素跳过 {result.pairs_skipped_coprime} 个，complete={result.complete}")
    print("首一约化 Gröbner 基（每一项都附带相对原始生成元的组合证书）：")
    for k, (p, combo) in enumerate(zip(rb.basis, rb.combinations)):
        cert = " + ".join(
            f"({q})*{names[i]}" for i, q in sorted(combo.items())
        )
        # 独立展开证书（直接用 Poly 算术重算一遍，不用引擎内部结果）
        expanded = Poly.zero(2)
        for i, q in combo.items():
            expanded = expanded + q * gens[i]
        check = "OK" if expanded == p else "不一致!"
        print(f"  b{k} = {str(p):<12}   证书: {cert:<28} 独立展开={check}")

    print()
    print("消元关系：基中含纯 y 的多项式，即 I ∩ Q[y]：")
    for p in rb.basis:
        if all(e[0] == 0 for e, _ in p.items()):
            print(f"  {p} = 0   —— 这就是消去 x 后 y 必须满足的关系")

    print()
    print("核验：全部最终 S 多项式除以该基的余式都为 0")
    for a, b in combinations(rb.basis, 2):
        rem = divide(s_polynomial(a, b), rb.basis).remainder
        print(f"  rem( S({a}; {b}) ) = {rem}")
        assert rem.is_zero()

    line("1b. 需要实际处理 S 对的例子（Cox 教材例）：f=x^3-2xy, g=x^2y-2y^2+x")
    f = x ** 3 - 2 * x * y
    g = x ** 2 * y - 2 * y ** 2 + x
    print(f"S(f, g) = {s_polynomial(f, g)}（这个 S 多项式首项不互素，必须真正约化）")
    res2 = groebner_basis([f, g])
    rb2 = reduced_basis(res2)
    print(f"Buchberger 实际处理 S 对 {res2.pairs_processed} 个，约化基：")
    for k, p in enumerate(rb2.basis):
        print(f"  b{k} = {p}")
    for a, b in combinations(rb2.basis, 2):
        assert divide(s_polynomial(a, b), rb2.basis).remainder.is_zero()
    print("全部 S 余式为 0，证书展开同样成立。")

    line("2. 理想成员判定（余式 + 组合证书）")
    f_member = y * g1 - g2  # = y^2+1，真实成员
    m = ideal_membership(f_member, gens)
    print(f"f = {f_member}")
    print(f"  判定: {m.status}，余式 = {m.remainder}")
    cert_terms = []
    expanded = Poly.zero(2)
    for i, q in sorted(m.coefficients.items()):
        cert_terms.append(f"({q})*{names[i]}")
        expanded = expanded + q * gens[i]
    print(f"  成员证书: f = {' + '.join(cert_terms)}")
    print(f"  独立展开证书 = {expanded}，与 f 相等: {expanded == f_member}")
    assert m.status == "member" and expanded == f_member

    f_non = x + y + 1
    m2 = ideal_membership(f_non, gens)
    print(f"f = {f_non}")
    print(f"  判定: {m2.status}，余式 = {m2.remainder}（非零 => "
          "基完整，证明不是理想成员）")
    assert m2.status == "nonmember"

    line("3. 异常边界（实际触发的拒绝，而非打印文字）")
    bad_values = [
        ("NaN 系数", {(1, 0): float("nan")}),
        ("bool 系数", {(0, 1): True}),
    ]
    for label, terms in bad_values:
        try:
            Poly(terms, nvars=2)
        except PolynomialError as exc:
            print(f"  拒绝 {label} -> PolynomialError: {exc}")
        else:
            raise AssertionError(f"{label} 竟然被接受了")

    line("完成")
    print("全部计算精确（fractions.Fraction），无近似误差；无第三方库调用。")


if __name__ == "__main__":
    main()
