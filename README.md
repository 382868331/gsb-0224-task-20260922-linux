# 多变量多项式的约化 Gröbner 基库（Python 标准库）

固定 **graded lex（grlex，次数优先、并列按 x1 > x2 > … 字典序）** 次序、
**有理数精确系数**的稀疏多变量多项式库，实现除法、S 多项式、Buchberger
算法、唯一的首一约化 Gröbner 基，并为每个基多项式记录它相对输入生成元的
**多项式线性组合证书**。支持理想成员判定（余式 + 组合证书）与预算耗尽的
`incomplete` 语义。
本轮新增 **理想饱和** `saturate(I, f)`：计算 `I:f^∞`（排除 `f=0` 分量后的
代数约束），用 Rabinowitsch 技巧（辅助变量 `t` 与 `1 - t*f`）加**消元序**
实现，并为每个输出多项式给出有限 `k` 与 `f**k * g ∈ I` 的组合证书。

**纯 Python 3.14.7 标准库实现**（只用 `fractions`、`dataclasses`、
`collections` 等），不调用任何符号计算第三方库，离线运行，无网络/账号/
数据库/云依赖。

## 运行环境与命令

- Python：3.14.7（代码不使用 3.14 专属语法，3.10+ 亦可运行，但以 3.14.7 为准）
- 依赖：**无**（不需要 `pip install` 任何包；`requirements.txt` 仅为占位说明）

```bash
# 必要测试（在仓库根目录执行）
python -m unittest discover -s tests -v

# 固定输入现场演示（真实计算，约 0.1 秒，远低于 8 秒要求）
python demo.py

# 饱和功能演示（真实计算正常 + 边界结果，约 0.1 秒，远低于 90 秒）
python demo_iteration.py
```

准备虚拟环境（可选；库只用标准库，venv 仅为隔离）：

```bash
python3.14 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

## 目录结构

```
groebner/
  __init__.py     # 公开接口导出
  poly.py         # Poly：稀疏有理系数多项式、grlex 次序、四则运算
  groebner.py     # 除法、S 多项式、Buchberger、首一约化基、成员判定、证书、饱和
  errors.py       # 异常类型
tests/
  reference.py    # 独立朴素参考实现（dict 多项式 + 另写的 Buchberger），
                  # 不导入被测核心，专门用于给出独立数学期望（含饱和参考）
  test_poly.py    # 数据结构/次序/运算/除法/S 多项式测试
  test_groebner.py# Gröbner 基/证书/成员判定/预算/边界测试
  test_saturate.py# 饱和：手算分量去除、独立参考对照、证书、边界、预算
demo.py           # 固定输入演示
demo_iteration.py # 饱和功能演示（本轮新增）
```

## 公开接口

### 多项式 `Poly`

```python
from groebner import Poly
from fractions import Fraction

p = Poly({(2, 0): 1, (1, 1): -2, (0, 0): 3}, n=2)  # x**2 - 2*x*y + 3
p = Poly([((1, 0), 1), ((1, 0), 1)], n=2)          # 也接受 (指数, 系数) 序列，同类项合并
Poly.zero(2)                       # 零多项式
Poly.constant(Fraction(1, 2), 2)   # 常数
Poly.variable(2, 0)                # x（下标从 0 开始）
```

- 内部存储 `dict[tuple[int,...], Fraction]`，空字典为零多项式；`Poly` 不可变。
- 系数接受 `int`、`fractions.Fraction`、**有限** `float`
  （`float` 在构造时经 `Fraction(value)` 精确化为有理数）。
- 运算：`p + q`、`p - q`、`p * q`、`-p`、`p.scale(c)`、
  `p.normalize_monic()`；`p.lm() / p.lc() / p.lt()`（零多项式抛
  `ZeroPolynomialError`）；`p.terms`（副本）、`len(p)`、按 grlex 降序迭代；
  `p.to_string(("x","y"))` 人类可读显示。

### 除法与 S 多项式

```python
from groebner import divide, s_polynomial
res = divide(f, [g1, g2])
res.quotients      # (q1, q2)，恒满足 f = q1*g1 + q2*g2 + res.remainder
res.remainder      # 余式；非 Gröbner 基时依赖除式排列顺序（数学事实，非缺陷）
s = s_polynomial(g1, g2)
```

除式含零多项式抛 `ZeroDivisorError`（错误信息带位置）。

### Gröbner 基

```python
from groebner import groebner_basis, check_s_polynomials, groebner_basis_of_zero

r = groebner_basis([g1, g2], max_pairs=2000, max_terms=300, max_steps=5000)
r.status            # "complete" 或 "incomplete"
r.basis             # complete 时为唯一的首一约化基（按首项降序）
r.certificates      # 每个基多项式 b_i 的证书 (c_i1,...,c_im)：
                    #   b_i = sum_j c_ij * 输入生成元_j
r.reduced           # complete 时恒为 True；incomplete 时恒为 False
r.pairs_processed, r.pairs_skipped_coprime, r.pairs_skipped_lcm, r.reason
check_s_polynomials(list(r.basis))   # 返回所有非零 S 多项式余式；[] 即通过判据
groebner_basis_of_zero(3)            # 零理想（无生成元），空基
```

- 零生成元被忽略，但在证书中**保留其输入位置**（该位置证书恒为零多项式）。
- 重复生成元、任意输入排列顺序、非首一倍数，都产生同一个首一约化基。
- 关键对用集合管理，采用**正常选择策略**（grlex 下 lcm 最小的对优先），
  并用 Buchberger 第一判据（首项单项式互素）与第二判据（lcm 判据）跳过
  必约为 0 的 S 多项式；跳过数分别在 `pairs_skipped_coprime`、
  `pairs_skipped_lcm` 中计数。两判据的正确性最终由完整基的逐对 S 多项式
  核验独立保证（测试与独立参考均复算）。
- 证书贯穿“初始首一化 → S 多项式约化 → 极小化 → 尾部约化”全过程同步维护。

### 理想成员判定

```python
from groebner import membership
m = membership(f, [g1, g2])
m.status       # "member" | "nonmember" | "inconclusive"
m.remainder    # 对最终基的余式
m.certificate  # member 时：f = sum_j certificate[j] * g_j；否则为 None
```

- `member`：余式为 0，证书可独立展开核验；
- `nonmember`：基完整（complete）且余式非零，非零余式即为非成员证据；
- `inconclusive`：预算在判定前耗尽且余式非零——**不允许**在此时宣称非成员。

### 理想饱和 `saturate`

```python
from groebner import saturate

r = saturate([x * y], x)          # I = (x*y)，f = x：排除 x=0 分量
r.status                          # "complete" | "incomplete"
r.basis                           # I:f^∞ 的首一约化 Gröbner 基（grlex，原变量环）
r.powers                          # 每个基多项式 g_i 对应的有限 k_i
r.certificates                    # f**k_i * g_i = sum_j cert_ij * 输入生成元_j
```

- 数学定义：`I:f^∞ = {g : 存在 k ≥ 0 使 f**k * g ∈ I}`，即排除 `f=0` 分量后
  剩余部分的代数约束。实现用 Rabinowitsch 技巧：扩环中加入辅助变量 `t` 与
  `1 - t*f`，关于 **t 最大的消元序**（先比 t 次数，再按 grlex）求约化基，
  取不含 `t` 的元素即得饱和；**不是**一次理想商（`(x²y):x = (xy)`，而饱和
  是 `(y)`）。辅助变量 `t` 不会泄漏到任何公开多项式中。
- 明确边界：`f = 0` → 单位理想 `(1)`（排除整个空间）；`f` 为非零常数 →
  原理想本身（常数是单位）；零理想（含空生成元序列）→ 零理想；
  单位理想 → 单位理想。
- 证书：每个输出 `g_i` 带有限 `k_i` 与相对**原始输入生成元**的组合证书，
  满足 `f**k_i * g_i = Σ_j cert_ij * g_j`，可独立展开核验。若证书项数超过
  `max_terms` 预算，则 `certificates`/`powers` 为 `None`、`reason` 单独说明，
  但 `basis` 仍是有效的完整结果。
- 变量数限制：原理想至多 `MAX_VARIABLES - 1 = 3` 个变量（辅助变量 `t` 占一位），
  超出抛 `InvalidGeneratorError`；预算语义与 `groebner_basis` 相同，
  耗尽返回 `status="incomplete"`，不宣称任何结论。

### 异常

| 异常 | 触发情形 |
|---|---|
| `InvalidPolynomialError` | 系数 NaN/Infinity、系数/指数为 `bool`、指数为负或非整数、指数长度与变量数不符、变量数越界等；消息带 `index=` 与指数向量，可定位 |
| `InvalidGeneratorError` | 生成元不是 `Poly`、变量数不一致、总输入项数 > 100、预算为负或非整数（含 `bool`）等 |
| `ZeroPolynomialError` | 对零多项式取首项/首项系数 |
| `ZeroDivisorError` | 除式或 S 多项式参数含零多项式 |

均继承 `GroebnerError`；前两者同时继承 `ValueError`。

## 输入上限（接口边界）

- 变量数 `n`：`1 <= n <= 4`；
- 所有输入生成元的非零项数之和：`<= 100`（超过抛 `InvalidGeneratorError`，
  消息标注累计位置）；
- 单项式次序：固定 grlex，不提供其他次序；
- 预算：三个非负整数关键字参数，`0` 合法（零预算立即 `incomplete`；
  拒绝负数、浮点与 `bool`）。默认 `max_pairs=2000`、`max_terms=300`、
  `max_steps=5000`，均可用关键字调大以处理更大的合法系统：
  - `max_pairs`：处理的关键对数（含跳过的对）；
  - `max_terms`：任一约化中间多项式允许的最大非零项数；
  - `max_steps`：全部约化累计的初等约化步数（含最终约化；成员判定取余
    另有一份等额预算）。它约束“项数不多但步数极多”的病态约化，使运行
    时间总有有限上界。

题面上限是接口边界；`demo.py` 只使用小规模固定输入，不穷举上限规模。

## 误差口径

**库内不存在浮点运算，也没有任何容差（tolerance）。** 所有系数、商与证书
都是 `fractions.Fraction` 精确有理数，相等判定是精确判定；不存在
“用宽容差掩盖边界错误”的可能。若用户以 `float` 提供系数，它在构造时被
`Fraction(x)` 按其二进制精确值转换（例如 `0.1` 变为 `3602879701896397/2**55`），
如需十进制语义请直接传 `Fraction("0.1")` 或整数。

## 设计取舍

1. **首一化与约化策略**：每个新元素入库前先按当前基约化再首一化；
   全部关键对处理完后，先删冗余首项（极小化），再对各元素尾部反复约化
   到不动点。项序是良序，每趟严格下降，过程必终止；结果是唯一的首一约化基。
2. **关键对管理**：正常选择策略（grlex 下 lcm 最小的对优先）+ Buchberger
   第一/第二判据剪枝。这是教科书标准、可独立核验的优化；未实现更重的
   Gebauer–Möller 安装或 sugar 次序（它们不改变数学结果，只影响速度），
   以保持实现小而可证。最终正确性不靠“信任判据”，而由完整基逐对
   S 多项式余式为 0 直接核验，测试与独立参考两边都复算。
3. **预算即工作量上界**：仅有关键对数不够——一次约化内部的初等步数可能
   无界。因此同时以 `max_pairs`/`max_terms`/`max_steps` 三个预算约束工作量，
   保证任何合法输入都在有限时间内得到 `complete` 或 `incomplete` 之一。
   默认值按“典型合法输入实际消耗约几十步、个位数项数”留出约两个数量级
   余量；更大的合法系统可调大预算。耗尽时返回不可变结果对象，不抛异常、
   不留部分状态；`incomplete` 基仍张成同一理想、证书仍合法，但**绝不**
   自称 Gröbner 基。
4. **证书是一等输出**：成员判定返回相对**原始输入生成元**（含零生成元位置）
   的组合，而不只是相对基的商，便于第三方独立验证。
5. **公开的 `divide` 与 S 多项式核验**不提供 incomplete 通道：多元除法在
   数学上必终止（项序良序），内部仅有一个防御性极高项数上限。

## 验证方式（独立性说明）

- `tests/reference.py` 是一份**完全独立的朴素实现**：多项式用裸 `dict`，
  加法/乘法/约化/S 多项式/Buchberger/约化基全部另写，且不使用互素跳过、
  采用不同的对枚举结构；它**不导入** `groebner` 的任何算法代码。
  测试中的期望值（约化基、余式、S 多项式）都由它独立给出。
- 证书检查为**独立展开**：`sum_j cert_j * g_j` 只经过多项式加法/乘法
  重算，与证书的产生路径无关。
- “往返一致”（如除法恒等式重算）只作为补充断言。
- 最终基逐对用被测的 S 多项式判据**和**参考实现分别复算余式为 0。

## 已知限制 / 非目标

- 只支持有理数域 Q、grlex 一种次序、1–4 个变量、总输入项 ≤ 100；
  不宣称兼容其他项序、域（有限域/代数扩域）或行业标准 CAS 行为。
- 不实现 Gröbner 扇、模块/非交换 Gröbner 基、根理想、消元序专门接口
  （消元序仅在 `saturate` 内部用于消去辅助变量 `t`，不对外提供任意次序接口；
  grlex 下消元关系可通过成员判定与证书展示，见 `demo.py`）。
- 饱和 `saturate` 因需引入辅助变量，原理想限 3 个变量以内；证书在
  `max_terms` 内给出，超出时单独说明（基仍有效）。
- 中间多项式项数与初等约化步数都可能很大；受 `max_pairs`、`max_terms`、
  `max_steps` 三重预算约束，超出时结果为 `incomplete`/`inconclusive`，
  需要调用方调大对应预算重算。
- 不含 CLI、服务端、数据库或任何网络功能；终端命令仅用于测试与演示。
