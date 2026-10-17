# 上交建模 AB 矩阵 MATLAB 计算记录

> 基于上交开源文档 10 状态 × 4 控制模型，通过 MATLAB 符号推导 + jacobian 计算 A、B 矩阵，最终 lqrd 得到 LQR 增益 K。

---

## 1. 整体流程

```
syms 全部变量(不赋数值)
  → equ1~equ5 (5条线性化动力学方程)
  → solve 解 5 个二阶导 (ddot_θ_w_l/r, ddot_θ_l_l/r, ddot_θ_b)
  → 用 solve 输出定义 s̈、φ̈ (运动学合成量)
  → 组装 dot_x (10×1)
  → jacobian(dot_x, x) → A 符号矩阵
  → jacobian(dot_x, u) → B 符号矩阵
  → subs 代入固定物理参数 / 腿长循环
  → double 转数值 → lqrd → K
  → fit 拟合 K(l) 多项式
```

---

## 2. solve 的 5 个未知数不包含 s̈、φ̈

### 问题

上交文档的 5 条动力学方程 (3.11~3.15) 中，自变量是 5 个刚体角加速度：
`ddot_θ_w_l, ddot_θ_w_r, ddot_θ_l_l, ddot_θ_l_r, ddot_θ_b`

**s̈ 和 φ̈ 不在其中**。它们是运动学导出量：
```matlab
ddot_s   = 0.5*R_w * (ddot_θ_w_l + ddot_θ_w_r);
ddot_phi = 0.5*R_w/R_l * (-ddot_θ_w_l + ddot_θ_w_r) - 0.5*l_l/R_l*ddot_θ_l_l + 0.5*l_r/R_l*ddot_θ_l_r;
```

### 做法

solve 只解 5 个角加速度，然后**用 solve 输出手动展开 s̈、φ̈**，填入 dot_x 的第 2、4 行。

---

## 3. jacobian 第二参数必须为纯独立符号

### 问题

`jacobian(dot_x, x)` 的第二个参数 `x` 中每个元素必须是**独立符号**，不能是表达式。

如果把 `x` 中的 `s` 替换为运动学表达式 `R_w*θ_w`，则 jacobian 求 ∂/∂(R_w*θ_w) 时会把整个表达式当成一个变量，导致偏导结果错误。

### 正确写法

```matlab
% 第二参数 x：全部是纯独立符号
x = [s, dot_s, phi, dot_phi, theta_l_l, dot_theta_l_l, theta_l_r, dot_theta_l_r, theta_b, dot_theta_b];

% 第一参数 dot_x：第 2、4 行展开为 solve 输出的合成量，其余行保持恒等
dot_x = [dot_s, ddot_s, dot_phi, ddot_phi, dot_theta_l_l, ddot_theta_l_l, dot_theta_l_r, ddot_theta_l_r, dot_theta_b, ddot_theta_b];
```

**s 和 dots 作为独立符号，ddot_s 中不显含 s → A 的 1、2 行为 0，符合物理事实**（s 不出现在动力学方程中）。

---

## 4. φ 的处理和 s 完全一样

### 问题

φ 在 equ1~equ5 中也没有显式出现，和 s 一样是"外部"状态量。

### 做法

φ̈ 和 s̈ 一样是合成量，同样**先 solve 再手动展开**。3.15 方程中 φ̈ 通过 `0.5*I_z*R_w/R_l*(ddot_θ_w_l - ddot_θ_w_r)` 的形式参与，已在 solve 中消化，φ̈ 本身由 solve 输出组装即可。

---

## 5. 方程已线性化，jacobian 后无需 subs(状态=0)

### 现象 vs 北科脚本

北科脚本的方程含 `sinθ, cosθ, θ̇²` → solve 后 θ̈ 仍含三角函数 → jacobian 偏导残存 sin/cos → **必须 subs(θ=0, θ̇=0)** 抹掉。

上交 equ1~equ5 已在 §3.2 做小角度近似（`sinθ→θ, cosθ→1`，舍 θ̇²）：
```
equ = (参数)·ddot_θ + (参数·g)·θ + T = 0
                        ↑ 已是 θ 的线性项
```

**对状态变量 subs(0) 无效果**（偏导已是常数）。但对腿长等物理参数仍需 subs 代入数值。

---

## 6. 符号赋值顺序：数值放 jacobian 之后，否则分数爆炸

### 现象

Ja 矩阵中出现几十位整数：
```
51933552740287646106775394497594591703437684145797616278840080 * l_l * l_w_l ...
```

### 原因

`syms g; g = 9.81;` 后 `sym(8.4) → 42/5`。solve 做 5×5 矩阵求逆 = adj(A)/det(A)，大量 `42/5, 981/100` 相乘通分，分子分母同时暴涨。

### 修复

不提前赋值，solve 和 jacobian 时全部参数保持纯字母：

```matlab
% 符号声明区 —— 全部 syms，一行赋值都不写
syms g R_w R_l l_l l_r l_w_l l_w_r l_b_l l_b_r l_c ...
syms m_b m_w m_l I_w I_z I_b I_l_l I_l_r ...

% solve + jacobian（全字母，无分数）
equ1 = ...; equ2 = ...;
solve → Ja, Jb

% ← 到这里才代入数值
Ja_num = subs(Ja, [g, m_b, m_w, R_w, ...], [9.81, 8.4, 0.322, 0.077, ...]);
```

**结论：全符号策略彻底避免分数通分爆炸。**

---

## 7. 三条语句的顺序导致 Jb 第 2、4 行全零

### 关键概念

在 MATLAB 中，符号变量的赋值是**快照（snapshot）**，不是引用。当你写：

```matlab
a = x;        % a 拍下 x 当前的值
x = x + 1;    % x 变了
               % a 不会跟着变！
```

`a` 存的是赋值那一刻 `x` 的拷贝，之后 `x` 怎么变都与 `a` 无关。同样地，`dot_x = [..., ddot_s, ...]` 拍下的是此刻 `ddot_s` 的值，后面重定义 `ddot_s` 不会触发 `dot_x` 自动更新。

---

### 现象

`Jb = jacobian(dot_x, u)` 中，第 2 行（对应 s̈）和第 4 行（对应 φ̈）对 `T_w_l, T_w_r, T_b_l, T_b_r` 的偏导全是 0。但上交文档中 `b_{2,j}` 和 `b_{4,j}` 明确非零。

---

### 根因

原始代码中有两次"拍照过早"：

```matlab
% ═══ 第 1 次拍早：ddot_s / ddot_phi 定义在 solve 之前 ═══
% L16: syms ddot_s;        → ddot_s = 裸符号, 不含任何变量关系
% L17: syms ddot_phi;      → ddot_phi = 裸符号
%
% L79: [ddot_θ_w_l, ...] = solve(equ1~equ5, ...);
%       → solve 覆盖了 ddot_θ_w_l 等，把它们变成了含 T_w, T_b, θ 的表达式
%       → 但 ddot_s 还是裸符号！
%
% ═══ 第 2 次拍早：dot_x 拼装 —— 此时 ddot_s 还是裸符号 ═══
% L80: dot_x = [dot_s, ddot_s, dot_phi, ddot_phi, ...];
%       → dot_x(2) 拍下的是裸符号 ddot_s，而不是 solve 相关表达式
%
% L82: ddot_s   = 0.5*R_w*(ddot_θ_w_l + ddot_θ_w_r);  ← 晚了！dot_x 不会更新
% L83: ddot_phi = ...;                                 ← 晚了！dot_x 不会更新
```

代入数字来理解：

| 时刻 | `ddot_s` 的真实值 | `dot_x(2)` 拍下的值 |
|------|-------------------|---------------------|
| syms 声明后 | `ddot_s`（裸符号） | — |
| solve 后 | `ddot_s`（裸符号，没变） | — |
| `ddot_s = f(ddot_θ)` | `0.5*R_w*(solve输出₁ + solve输出₂)` ← 含 T_w | `ddot_s`（裸符号，没更新）|

jacobian 对 `dot_x(2)`（裸符号 ddot_s）求 ∂/∂T_w = **0**，因为裸符号跟 T_w 没有任何关系。

---

### 修复：三步换序

```matlab
% ① 先 solve —— 此时 ddot_θ 变成含 T_w, T_b, θ 的表达式
[ddot_θ_w_l, ddot_θ_w_r, ddot_θ_l_l, ddot_θ_l_r, ddot_θ_b] = solve([equ1~equ5], [...]);

% ② 再定义 s̈ 和 φ̈ —— 此时 ddot_θ 已是 solve 输出，表达式含 T_w, T_b
ddot_s   = 0.5*R_w * (ddot_θ_w_l + ddot_θ_w_r);
ddot_phi = 0.5*R_w/R_l * (-ddot_θ_w_l + ddot_θ_w_r) - ...;

% ③ 最后拼装 dot_x —— 拍下的是"含 T_w/T_b"的新 ddot_s
dot_x = [dot_s, ddot_s, dot_phi, ddot_phi, dot_θ_l_l, ddot_θ_l_l, dot_θ_l_r, ddot_θ_l_r, dot_θ_b, ddot_θ_b];

% ④ jacobian
Ja = jacobian(dot_x, x);
Jb = jacobian(dot_x, u);    % ← 现在 ∂ddot_s/∂T_w ≠ 0 ✅
```

修复后，`dot_x(2)` 存的是 `0.5*R_w*(solve输出₁ + solve输出₂)`，这个表达式显含 T_w、T_b，j jacobian 求偏导就能得到正确非零值。

**一句话：solve → ddot_s/ddot_phi → dot_x → jacobian，四步必须严格按这个顺序。**

---

## 8. 上交给 A/B 表达式但不给 K —— 为什么要自己算

- **上交给的是推导结果**（A、B 的符号表达式），用于验证建模正确性
- **K 由具体机械参数决定**，参数不同 K 完全不同
- 必须代入自己模型的参数（质量、惯量、杆长）重新跑 jacobian → lqrd，无法直接用上交的 K

---

## 9. 腿长自适应：fit 多项式拟合

### 问题

LQR 只能对**固定腿长**的线性系统计算 K。实际运行时腿长是变化的。

### 做法

```
多个离散腿长点 (l₁, l₂, ..., lₙ)
  → 每个点: subs(腿长参数) → double → lqrd → 得到一个 4×10 的 K
  → fit(leglen, K(某元素), 'poly2') → K = a·l² + b·l + c
```

用二次多项式拟合 K 的每个元素关于腿长 l 的连续函数，运行时根据当前腿长插值得到实时 K。

**K_A / K_B / K_K 就是 fit 返回的拟合系数**（二次项、一次项、常数项）。

---

## 10. lqrd vs lqr

| 函数 | 类型 | 参数 | 用途 |
|------|------|------|------|
| `lqr(A, B, Q, R)` | 连续 LQR | 无 Ts | 解连续 Riccati 方程 |
| `lqrd(A, B, Q, R, Ts)` | 离散 LQR | 含采样时间 Ts | 先离散化再解离散 Riccati |

仿真中控制是离散的（每个时间步算一次 u），所以用 `lqrd`。

---

## 11. double() —— 符号转数值的"过门"

符号工具箱的数即使看起来像数字，`lqr`、`lqrd` 也不认。必须用 `double()` 转换：

```matlab
A_num = double(subs(A_sym, [l_l, l_w_l, ...], [0.25, 0.12, ...]));  % sym → double
K     = lqrd(A_num, B_num, Q, R, 0.001);                              % lqrd 只接受 double
```

---

## 12. subs / vpa / simplify 执行顺序

```matlab
A = simplify( vpa( subs(Ja, [theta, dot_theta, ...], [0, 0, ...]) ) );
           └─────────────────── 从内往外 ──────────────────┘
           ① subs: 代入平衡点数值
           ② vpa:  符号精确分数 → 小数
           ③ simplify: 代数化简合并同类项
```

**这个顺序对本模型不重要**（方程已线性），但对北科那种含 sin/cos 的非线性方程是必要的。

---

## 13. LQR 输出 → VMC 输入的对应关系

```
LQR 输出 u = [T_w_l, T_w_r, T_b_l, T_b_r]
              │        │        │        │
              轮力矩    轮力矩    髋力矩    髋力矩
              │        │        │        │
              └────────┴────────┘        │
              直接驱动轮电机              │
                                         ↓
                              VMC: Tp(切向力矩) ← 直接对应
                                   F0(径向力)   → 需额外高度控制器
```

LQR 输出的 `T_b_l`/`T_b_r` 对应 VMC 的切向力矩 Tp。腿长方向的控制力 F0 需要另外设计高度控制器。

---

## 14. 状态量与控制量定义

```
x = [s,    dot_s,    phi,    dot_phi,    θ_l_l,    dot_θ_l_l,    θ_l_r,    dot_θ_l_r,    θ_b,    dot_θ_b]
     位移   位移速度   yaw角   yaw角速度   左腿倾角  左腿倾角速度   右腿倾角  右腿倾角速度   机体俯仰 机体俯仰速度

u = [T_w_l,    T_w_r,    T_b_l,    T_b_r]
     左轮力矩   右轮力矩   左髋力矩   右髋力矩
```