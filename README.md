# 多变量多项式的约化 Gröbner 基库（Python 标准库）

**有理数精确系数**的稀疏多变量多项式库，支持两种单项式次序：

- **graded lex（grlex，默认）**：次数优先、并列按 x1 > x2 > … 字典序；
- **lex（纯字典序，x1 > x2 > …）**：配合 `eliminate` 做消元——指定要
  消去的变量，直接返回基中不含这些变量的元素（消元理想
  `I ∩ k[剩余变量]` 的首一约化基）。

在两种次序下都实现除法、S 多项式、Buchberger 算法、唯一的首一约化
Gröbner 基，并为每个基多项式记录它相对输入生成元的**多项式线性组合
证书**。支持理想成员判定（余式 + 组合证书）与预算耗尽的 `incomplete` 语义。

**纯 Python 3.14.7 标准库实现**（只用 `fractions`、`dataclasses`、
`collections` 等），不调用任何符号计算第三方库，离线运行，无网络/账号/
数据库/云依赖。

## 运行环境与命令

- Python：3.14.7（代码不使用 3.14 专属语法，3.10+ 亦可运行，但以 3.14.7 为准）
- 依赖：**无**（不需要 `pip install` 任何包；`requirements.txt` 仅为占位说明）

```bash
# 必要测试（在仓库根目录执行）
python -m unittest discover -s tests -v

# 固定输入现场演示（真实计算，约 0.1 秒，远低于 90 秒要求）
python demo.py
```

## 目录结构

```
groebner/
  __init__.py     # 公开接口导出
  poly.py         # Poly：稀疏有理系数多项式、grlex/lex 次序键、四则运算
  groebner.py     # 除法、S 多项式、Buchberger（两种次序）、首一约化基、
                  # 成员判定、证书、lex 消元入口 eliminate
  errors.py       # 异常类型
tests/
  reference.py    # 独立朴素参考实现（dict 多项式 + 另写的 Buchberger，
                  # grlex/lex 两序），不导入被测核心，专门给出独立数学期望
  test_poly.py    # 数据结构/次序/运算/除法/S 多项式测试
  test_groebner.py# grlex 基/证书/成员判定/预算/边界测试（旧测试保留）
  test_elimination.py # lex 次序、消元的正常/边界/失败测试（本轮新增）
demo.py           # 固定输入演示（grlex + lex 消元）
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
  `p.normalize_monic()`；`p.lm() / p.lc() / p.lt()`（固定 grlex 的首项；
  零多项式抛 `ZeroPolynomialError`）、`p.leading(key)`（按 `grlex_key` 或
  `lex_key` 取首项，供其他次序使用）；`p.terms`（副本）、`len(p)`、
  按 grlex 降序迭代；`p.to_string(("x","y"))` 人类可读显示。

### 单项式次序

```python
from groebner import grlex_key, lex_key
# grlex_key((a,b)) = (a+b, (a,b))：次数优先；lex_key((a,b)) = (a,b)：纯字典序
# lex 下二元单项式：x**2 > x*y > x > y**2 > y > 1
```

### 除法与 S 多项式

```python
from groebner import divide, s_polynomial
res = divide(f, [g1, g2])                 # 默认 order="grlex"
res = divide(f, [g1, g2], order="lex")    # lex 除法
res.quotients      # (q1, q2)，恒满足 f = q1*g1 + q2*g2 + res.remainder
res.remainder      # 余式；非 Gröbner 基时依赖除式排列顺序（数学事实，非缺陷）
s = s_polynomial(g1, g2)                 # order 关键字同样可选 "grlex"/"lex"
```

除式含零多项式抛 `ZeroDivisorError`（错误信息带位置）；未知 `order`
抛 `InvalidGeneratorError`。

### Gröbner 基

```python
from groebner import groebner_basis, check_s_polynomials, groebner_basis_of_zero

r = groebner_basis([g1, g2])                    # 默认 order="grlex"（旧行为）
r = groebner_basis([g1, g2], order="lex")       # lex 次序（至多 3 变量）
r.status            # "complete" 或 "incomplete"
r.order             # "grlex" 或 "lex"
r.basis             # complete 时为唯一的首一约化基（按该次序首项降序）
r.certificates      # 每个基多项式 b_i 的证书 (c_i1,...,c_im)：
                    #   b_i = sum_j c_ij * 输入生成元_j
r.reduced           # complete 时恒为 True；incomplete 时恒为 False
r.pairs_processed, r.pairs_skipped_coprime, r.pairs_skipped_lcm, r.reason
check_s_polynomials(list(r.basis))               # 默认 grlex；lex 基传 order="lex"
groebner_basis_of_zero(3)                        # 零理想（无生成元），空基
```

- 不传 `order` 时一切与旧版完全一致（grlex）；原有调用无需改动。
- 零生成元被忽略，但在证书中**保留其输入位置**（该位置证书恒为零多项式）。
- 重复生成元、任意输入排列顺序、非首一倍数，都产生同一个首一约化基。
- 关键对用集合管理，采用**正常选择策略**（当前次序下 lcm 最小的对优先），
  并用 Buchberger 第一判据（首项单项式互素）与第二判据（lcm 判据）跳过
  必约为 0 的 S 多项式；跳过数分别在 `pairs_skipped_coprime`、
  `pairs_skipped_lcm` 中计数。两判据的正确性最终由完整基的逐对 S 多项式
  核验独立保证（测试与独立参考均复算）。
- 证书贯穿“初始首一化 → S 多项式约化 → 极小化 → 尾部约化”全过程同步维护。

### lex 消元：`eliminate`

```python
from groebner import eliminate

e = eliminate(generators, [0])            # 消去 x1（下标从 0 起）
e = eliminate(generators, (0, 2))         # 同时消去 x1、x3
e.status                  # "complete" | "incomplete"
e.eliminate               # 被消去变量下标元组，如 (0,)、(0, 2)
e.elimination_basis       # complete：不含被消去变量的基元素（消元理想
                          #   I ∩ k[剩余变量] 的唯一首一约化基，可能为空）
e.certificates            # 每个消元基元素相对原始输入生成元的组合证书
e.groebner                # 完整 lex Groebner 基结果（EliminationResult 透传
                          #   status/is_complete/reason）
```

- **数学依据（消元定理）**：取 lex 次序使被消去变量排在最前，则 lex
  Gröbner 基中不含这些变量的元素恰好生成消元理想 `I ∩ k[剩余变量]`。
- 被消去变量**不必是前缀**（如在 `(x,y,z)` 中消去 `z`）：库内部自动把
  变量重排为“先消去、后保留”再计算，返回时映回原变量坐标，调用方无需
  手动换元。此时 `e.groebner.basis` 是按内部置换次序的基（坐标仍是原
  坐标），不要对它直接用标准 `x1>x2>...` 的 `check_s_polynomials`；
  `elimination_basis` 本身在保留变量环上是 lex 基，可直接
  `check_s_polynomials(list(e.elimination_basis), order="lex")`。
- 空消元基是合法且有意义的结果：消元理想为零（原理想不含任何只涉及
  剩余变量的非零关系，例如参数曲线消去参数之外的方向）。若理想不相容
  （如 `(x, x-1)`），消元基含常数 `1`。
- 预算耗尽时 `status="incomplete"`、`elimination_basis` 与 `certificates`
  为空元组——**不发布未核验的半成品**；中间基及其合法证书仍在
  `e.groebner` 中，调大预算重算即可。

手算例子（均在测试与 `demo.py` 中核对）：

| 生成元 | 消去 | 消元基 |
|---|---|---|
| `x²+y²−1, x−y²` | `x` | `y⁴+y²−1` |
| `x−2y, y−3` | `y` / `x` | `x−6` / `y−3` |
| `xy−1, xz−1` | `x` | `y−z` |
| `x−t, y−t²` | `t`（非前缀） | `x²−y` |
| `x²−y, y²−z, z−1` | `x` / `z` | `{y²−1, z−1}` / `{x²−y, y²−1}` |

### 理想成员判定

```python
from groebner import membership
m = membership(f, [g1, g2])                # 默认 order="grlex"
m = membership(f, [g1, g2], order="lex")   # lex 下判定
m.status       # "member" | "nonmember" | "inconclusive"
m.remainder    # 对最终基的余式
m.certificate  # member 时：f = sum_j certificate[j] * g_j；否则为 None
```

- `member`：余式为 0，证书可独立展开核验；
- `nonmember`：基完整（complete）且余式非零，非零余式即为非成员证据；
- `inconclusive`：预算在判定前耗尽且余式非零——**不允许**在此时宣称非成员。

### 异常

| 异常 | 触发情形 |
|---|---|
| `InvalidPolynomialError` | 系数 NaN/Infinity、系数/指数为 `bool`、指数为负或非整数、指数长度与变量数不符、变量数越界等；消息带 `index=` 与指数向量，可定位 |
| `InvalidGeneratorError` | 生成元不是 `Poly`、变量数不一致、总输入项数 > 100、预算为负或非整数（含 `bool`）、未知 `order`、`eliminate` 下标非法（裸 int / 空 / 越界 / 重复 / `bool`）、lex 变量数超过 3 等 |
| `ZeroPolynomialError` | 对零多项式取首项/首项系数 |
| `ZeroDivisorError` | 除式或 S 多项式参数含零多项式 |

均继承 `GroebnerError`；前两者同时继承 `ValueError`。

## 输入上限（接口边界）

- 变量数 `n`：grlex 为 `1 <= n <= 4`；**lex（及 `eliminate`）本轮限定
  `1 <= n <= 3`**，超过抛 `InvalidGeneratorError`；
- 所有输入生成元的非零项数之和：`<= 100`（超过抛 `InvalidGeneratorError`，
  消息标注累计位置）；
- 单项式次序：`"grlex"`（默认）与 `"lex"` 两种；`order` 必须是这两个
  字符串之一，其他值（含大小写变体）一律拒绝；
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
2. **关键对管理**：正常选择策略（当前次序下 lcm 最小的对优先）+ Buchberger
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
4. **证书是一等输出**：成员判定与消元返回相对**原始输入生成元**（含零
   生成元位置）的组合，而不只是相对基的商，便于第三方独立验证。
5. **公开的 `divide` 与 S 多项式核验**不提供 incomplete 通道：多元除法在
   数学上必终止（项序良序），内部仅有一个防御性极高项数上限。
6. **消元用纯 lex + 变量置换实现**：`eliminate` 不引入新的项序类型，只把
   “被消去变量置于 lex 最前”这一步用确定性的变量重排完成，算完映回原
   坐标；因此 grlex 路径完全不受影响，消元的正确性就是消元定理本身，
   可由独立参考（另写的 lex Buchberger + 过滤）逐项复算。

## 验证方式（独立性说明）

- `tests/reference.py` 是一份**完全独立的朴素实现**：多项式用裸 `dict`，
  加法/乘法/约化/S 多项式/Buchberger/约化基全部另写（grlex 与 lex 两个
  次序），且不使用互素跳过、采用不同的对枚举结构；它**不导入** `groebner`
  的任何算法代码。测试中的期望值（约化基、余式、S 多项式、消元基）都由
  它独立给出。
- lex/消元的期望除随机系统与独立参考逐单项式比对之外，还用手算例子钉死
  （代入关系 `x−6`、隐式化 `x²−y`、不相容系统的常数 `1`、空消元理想等）。
- 证书检查为**独立展开**：`sum_j cert_j * g_j` 只经过多项式加法/乘法
  重算，与证书的产生路径无关。
- “往返一致”（如除法恒等式重算）只作为补充断言。
- 最终基逐对用被测的 S 多项式判据**和**参考实现分别复算余式为 0。

## 已知限制 / 非目标

- 只支持有理数域 Q 与两种次序（grlex、lex）；grlex 限 1–4 个变量，
  lex/消元本轮限 **1–3 个变量**、面向小次数系统；总输入项 ≤ 100。
  不宣称兼容其他项序（分块/权序等）、域（有限域/代数扩域）或行业标准
  CAS 行为。
- 本轮迭代只做“lex 次序 + 指定消去变量并返回消元基”；**不**包含块序、
  理想交专门接口、多素数/模重建与证书加速，也没有单列的大规模性能指标。
- 不实现 Gröbner 扇、模块/非交换 Gröbner 基、根理想。
- lex 下中间多项式系数与项数可能比 grlex 膨胀得更快；仍受
  `max_pairs`/`max_terms`/`max_steps` 三重预算约束，超出时结果为
  `incomplete`/`inconclusive`（消元入口此时不发布半成品消元基），
  需要调用方调大对应预算重算。
- 不含 CLI、服务端、数据库或任何网络功能；终端命令仅用于测试与演示。
