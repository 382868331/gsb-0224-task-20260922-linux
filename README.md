# 多变量多项式的约化 Gröbner 基库（Python 标准库）

固定 **graded lex（grlex，次数优先、并列按 x1 > x2 > … 字典序）** 次序、
**有理数精确系数**的稀疏多变量多项式库，实现除法、S 多项式、Buchberger
算法、唯一的首一约化 Gröbner 基，并为每个基多项式记录它相对输入生成元的
**多项式线性组合证书**。支持理想成员判定（余式 + 组合证书）、两个有限生成
理想的交（内部 `t` 消元 + 原环双证书）与预算耗尽的 `incomplete` 语义。

**纯 Python 3.14.7 标准库实现**（只用 `fractions`、`dataclasses`、
`collections` 等），不调用任何符号计算第三方库，离线运行，无网络/账号/
数据库/云依赖。

## 运行环境与命令

- Python：3.14.7（代码不使用 3.14 专属语法，3.10+ 亦可运行，但以 3.14.7 为准）
- 依赖：**无**（不需要 `pip install` 任何包；`requirements.txt` 仅为占位说明）

```bash
# 必要测试（在仓库根目录执行）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入现场演示（真实计算，通常远低于 90 秒）
./.venv/bin/python demo.py
```

## 目录结构

```
groebner/
  __init__.py     # 公开接口导出
  poly.py         # Poly：稀疏有理系数多项式、grlex 次序、四则运算
  groebner.py     # 除法、S 多项式、Buchberger、首一约化基、成员判定、证书
  errors.py       # 异常类型
tests/
  reference.py    # 独立朴素参考实现（dict 多项式 + 另写的 Buchberger），
                  # 不导入被测核心，专门用于给出独立数学期望
  test_poly.py    # 数据结构/次序/运算/除法/S 多项式测试
  test_groebner.py# Gröbner 基/证书/成员判定/预算/边界测试
  test_intersection.py# ideal_intersection 消元、双证书、边界/预算测试
demo.py           # 固定输入演示
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

### 两个有限生成理想的交

```python
from groebner import ideal_intersection

r = ideal_intersection([f1, f2], [g1, g2])
r.status        # "complete" | "incomplete"
r.basis         # I ∩ J 在原变量环 Q[x] 中的首一约化基
r.certificates  # 每个交集生成元 h 的两份证书：
                #   (a1,...), (b1,...)，分别满足
                #   h = sum_i a_i*f_i，h = sum_j b_j*g_j
r.elimination   # k[x,t] 中 tI + (1-t)J 的块次序 Groebner 基结果
```

- 算法只用于两个有限生成理想：在内部加入消元变量 ``t``，计算
  ``<t*f_i, (1-t)*g_j>``，块次序为 ``t >> x1,...,xn``，再提取不含 ``t``
  的首一约化基元素。块次序仅存在于该入口内部；公开的
  ``groebner_basis`` 仍是固定 graded lex，不新增公开项序参数。
- 返回证书是原变量环中分别相对 ``I``、``J`` 的两份组合证书，不是只在
  ``Q[x,t]`` 中成立的扩展表达式。
- 原变量最多 3 个；连同 ``t`` 后总共最多 4 个，仍受原内核变量上限约束。
- 零理想可用空序列（变量数可由另一侧推断）或 ``[Poly.zero(n)]``；
  两个序列同时为空时无法推断变量数，会抛 ``InvalidGeneratorError``。
- 单位理想、重复生成元与非首一倍数按普通 Groebner 基流程正确去重；
  交集为单位理想时返回常数 ``1``。
- ``max_pairs``、``max_terms``、``max_steps`` 覆盖提升后消元 Groebner
  基的完整计算。预算耗尽时只能得到 ``"incomplete"``；结果不发布任何
  当前碰巧不含 ``t`` 的部分项，避免把中间结果误称为完整交集。
- 该入口不是一般符号代数前端：不接受字符串表达式、无穷生成理想、
  其他消元变量名或其他域。

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

- 公开 `groebner_basis` 只支持有理数域 Q、grlex 一种次序、1–4 个变量、
  总输入项 ≤ 100；`ideal_intersection` 只接受 1–3 个原变量（内部加 `t`
  后至多 4 个），每一侧总输入项 ≤ 100。
- 不做一般符号代数前端，不接受字符串表达式或任意符号变量名；不宣称兼容
  其他公开项序、域（有限域/代数扩域）或行业标准 CAS 行为。
- 不实现 Gröbner 扇、模块/非交换 Gröbner 基、根理想、通用消元序接口；
  块次序仅封装在两个理想交集这一固定入口内。
- 中间多项式项数与初等约化步数都可能很大；受 `max_pairs`、`max_terms`、
  `max_steps` 三重预算约束，超出时结果为 `incomplete`/`inconclusive`，
  需要调用方调大对应预算重算。
- 不含 CLI、服务端、数据库或任何网络功能；终端命令仅用于测试与演示。
