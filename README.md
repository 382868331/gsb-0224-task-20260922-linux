# 多变量多项式的约化 Gröbner 基（有理系数，graded lex）

一个从零实现的、可导入的离线符号代数小题库：在 Q[x0, …, x_{n-1}]（n ≤ 4）上用
固定 **graded lex（grlex）** 项序实现稀疏多项式、多元除法、S 多项式、Buchberger
算法与唯一的**首一约化 Gröbner 基**，并记录每个基多项式相对**原始输入生成元**的
多项式线性组合（成员证书）。支持理想成员判定及其组合证书，支持预算耗尽时返回
`incomplete`（绝不把未完成基称作最终基）。

仅使用 Python 3.14.7 **标准库**（`fractions`、`dataclasses`、`itertools` 等），
不调用任何符号计算库或第三方同类实现，不需要网络/账号/密钥/Docker/数据库。

## 运行环境与命令

- Python：**3.14.7**（不使用任何第三方包，装好 Python 即可）。
- 平台：Linux 原生。

```bash
# 执行全部必要测试（含独立参考实现对照与固定种子随机属性测试）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.05 秒内完成，真实计算，无 sleep / 无预录结果）
python demo.py
```

## 快速上手

```python
from fractions import Fraction
from groebner import Poly, groebner_basis, reduced_basis, ideal_membership

x, y = Poly.var(0, 2), Poly.var(1, 2)
g1, g2 = x + y, x * y - 1

res = groebner_basis([g1, g2])          # Buchberger
rb  = reduced_basis(res)                # 唯一首一约化基
print(res.complete, len(rb.basis))      # True 2 ; 基为 {y^2 + 1, x + y}

# 每个基多项式的组合证书：rb.basis[k] == Σ rb.combinations[k][i]*g_i
for p, combo in zip(rb.basis, rb.combinations):
    assert sum((q * [g1, g2][i] for i, q in combo.items()), Poly.zero(2)) == p

# 理想成员判定（余式 + 证书）
m = ideal_membership(y * g1 - g2, [g1, g2])
print(m.status, m.remainder)            # member 0
m2 = ideal_membership(x + y + 1, [g1, g2])
print(m2.status, m2.remainder)          # nonmember 1
```

## 公开数据结构与接口

### `Poly`（`groebner.polynomial`，不可变值对象）

- 表示：`dict[指数元组 -> Fraction]` 的稀疏形式，零项不保留；变量数在构造时固定。
- 构造：
  - `Poly.zero(nvars)`、`Poly.constant(c, nvars)`、`Poly.var(i, nvars)`；
  - `Poly({(1, 0): 1, (0, 2): -Fraction(1,3)}, nvars=2)`；
  - `Poly.from_terms([((1, 0), 1), ...], nvars=2)`（重复指数会合并）。
- 项序与首项：`grlex_key(e)`、`p.leading_monomial()`、`p.leading_coefficient()`、
  `p.leading_term()`、`p.terms()`（按 grlex 降序的快照）。
- 算术：`+ - * / **`（`/` 仅为除以非零有理标量；多项式除法见 `divide`），
  支持与有限有理数标量混写（`x + 1`、`2*y`）。
- 其它：`p.nvars`、`len(p)`（非零项数）、`p.is_zero()`、`p.is_constant()`、
  `p.scale_to_monic()`、`p.is_monic()`、`p.coefficient(exp)`、`str(p)`
  （变量名 x, y, z, w）。
- 常量：`MAX_VARIABLES = 4`。

### 算法接口（`groebner.engine`）

| 接口 | 说明 |
| --- | --- |
| `GroebnerBudgets(max_pairs=10**6, max_terms=10**6)` | 搜索预算（正 int，拒绝 bool） |
| `groebner_basis(generators, budgets=...) -> GroebnerBasisResult` | Buchberger；零生成元丢弃但保留下标 |
| `reduced_basis(result, budgets=None) -> GroebnerBasisResult` | 首一约化基（唯一）；拒绝 incomplete 基 |
| `divide(f, divisors) -> DivisionResult` | 固定 grlex 多元除法；`.quotients/.remainder/.identity()` |
| `remainder(f, divisors)` | 除法余式便捷封装 |
| `s_polynomial(f, g)` | S 多项式 |
| `ideal_membership(f, generators, budgets=...) -> MembershipResult` | 一步完成基计算 + 成员判定 |
| `membership_with_basis(f, result, budgets=None)` | 基于已有结果判定 |

- `GroebnerBasisResult` 字段：`complete`、`basis`、`combinations`（每个基多项式
  的 `dict[int, Poly]` 证书）、`generators`、`nvars`、`reduced`、`monic`、
  `pairs_processed`、`pairs_skipped_coprime`、`peak_total_terms`、`budget`、
  `incomplete_reason`；方法 `certificate(index)`。
- `MembershipResult` 字段：
  - `status`：`"member"`（余式 0，`coefficients` 为成员证书）；
    `"nonmember"`（仅基完整时给出，余式非零即**证明非成员**）；
    `"unknown"`（预算耗尽、基不完整且余式非零——**不下结论**）。
  - `remainder`、`coefficients: dict[int, Poly]`、`member`（True/False/None）、
    `certificate`（仅 member 时非 None）。
  - 恒等式：`f - remainder == Σ coefficients[i] * generators[i]`。
- 输入上限常量：`MAX_INPUT_TERMS = 100`（全部生成元非零项总数）。

## 输入上限与错误语义

- 变量数 `1 ≤ nvars ≤ 4`；生成元非零项总数 `≤ 100`，超限抛 `PolynomialError`。
- 指数：长度必须等于变量数、非负 `int`（拒绝 `bool`，拒绝负指数）。
- 系数：`int` / `fractions.Fraction` / 有限 `float` / 标准库 `numbers.Rational`
  / 可解析的有理数字符串。
  - **拒绝 NaN 与 ±Infinity**（`float("nan")`、`float("inf")`、`"nan"`、
    `"inf"` 等），统一抛 `PolynomialError`。
  - **所有整数形参拒绝 `bool`**（`True` 不等于 1）：变量数、变量下标、指数、
    幂指数、预算等。
- 非法输入一律抛 `PolynomialError(ValueError)`，错误信息带位置（如
  `多项式项[0]: 指数必须非负`、`项 (0, 0) 的系数: 拒绝 NaN/Infinity`、
  `generators[2] ...`），**不静默纠正**。
- 变量数不一致的多项式混合运算、零多项式作除数/作 S 多项式参数均报错。
- 有状态/批处理语义：算法结果均为新对象，预算耗尽只返回带 `incomplete_reason`
  的不可变结果，不修改任何输入，也不存在“改了一半”的外部状态。

## 预算耗尽 vs 证明无解（严格分开）

- `GroebnerBudgets.max_pairs`：最多**真正做约化**的 S 对数量（首项互素而按
  Buchberger 第一判据跳过的不计入，单独在 `pairs_skipped_coprime` 计数）。
- `GroebnerBudgets.max_terms`：约化中“基总项数 + 工作多项式项数”的峰值上限。
- 预算耗尽：`complete=False`、`reduced=False`、`monic=False`，带
  `incomplete_reason`；已产出的基成员及其证书仍全部可用（恒等式成立），
  但**不能**称其为最终基（`reduced_basis` 对 incomplete 基直接报错）。
- “证明不是成员”只可能发生在 `complete=True` 的基上；incomplete 基上余式非零
  的结果一律标为 `unknown`。余式为 0 的成员结论不依赖预算，incomplete 时也可靠。

## 浮点 / 误差口径

- 全部内部计算使用 `fractions.Fraction` **精确有理数**运算，**不存在浮点舍入
  误差，也没有容差（tolerance）**；因此没有“用宽容差掩盖边界错误”的空间。
- 仅在**输入端**接受有限 `float`，并用 `Fraction(float)` 取其**精确二进制值**
  （如 `0.1` 会变成 `3602879701896397/36028797018963968`），之后全程精确运算。
  需要十进制语义时请传 `Fraction("0.1")` 或字符串 `"0.1"`。
- 演示与测试的所有核验都是精确等式断言（证书展开逐系数相等、S 余式恰为 0）。

## 算法与设计取舍

1. **项序固定 grlex**：总次数优先、平局指数向量字典序（x0 > x1 > …）。不实现
   lex/grevlex 等可切换项序（超出本题契约）。
2. **Buchberger**：FIFO 处理关键对；采用第一判据（首项单项式互素则跳过 S）。
   正确性另由测试中“全部最终 S 多项式余式为 0”独立核验把关；未实现第二判据/
   Gebauer–Möller 等工程优化（在 4 变量、≤100 输入项的契约规模内无必要）。
3. **约化策略**：每步消去当前“最大的可约项”，得到完整范式（不是只做到首项
   不可约），从而约化基中每个成员的任何项都不被其他成员首项整除。
4. **组合证书**：每个带标签基多项式与它的 `dict[int, Poly]` 同步维护——
   初始约化、S 对构造（`S = m_f*f - m_g*g`）、约化步（`p <- p - m*g`）、
   首一化、约化基删除/替换，每一步都同步变换证书；删除成员时其证书行一并删除。
   测试用独立参考展开器逐成员核验恒等式。
5. **零生成元**：不参与基，但**保留输入下标位置**，证书下标与调用方输入一致。
6. **确定性**：对处理顺序（FIFO）与最终输出（按首项 grlex 排序）均做固定化，
   不同输入顺序产生相同的首一约化基（有测试覆盖）。
7. **独立性**：`tests/reference.py` 是与被测核心无代码共享的朴素实现
   （dict + Fraction，无判据 Buchberger、独立约化基、独立展开器），期望值全部
   来自它；`DivisionResult.identity()` 一类的“往返一致”只作补充断言。

## 已知限制

- 仅限 Q 上的多项式：不支持 Z/pZ、有限域、幂级数或非交换情形。
- 变量数上限 4、输入非零项总数上限 100（接口边界，非性能建议）。
- 仅 grlex 一种项序；不宣称与任何行业标准 CAS 全功能兼容。
- 未做 Groebner 走查（walk）、签名基（F5/G2V）等加速；预算默认值很大，
  刻意构造的大例子请显式传小预算观察 `incomplete`。
- 不提供文件 I/O、网络或 CLI 产品；`demo.py` 是唯一的固定输入演示入口。

## 目录

```
groebner/__init__.py      # 公开接口导出
groebner/polynomial.py    # Poly、grlex、单项式工具、输入校验
groebner/engine.py        # 除法 / S 多项式 / Buchberger / 约化基 / 成员判定
tests/reference.py        # 独立朴素参考实现（不调用被测核心）
tests/test_polynomial.py  # 数据结构 / 项序 / 校验测试
tests/test_engine.py      # 算法 / 证书 / 预算 / 错误语义测试
demo.py                   # 固定输入演示
requirements.txt          # 无第三方依赖（仅注释说明）
```
