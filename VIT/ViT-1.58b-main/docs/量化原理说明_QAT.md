# 权重与激活量化原理说明（QAT）

本文对应实现文件：`quant_linear.py`。

---

## 1. 这是不是 QAT？

**是的，属于 QAT（Quantization-Aware Training，量化感知训练）。**

| 对比项 | PTQ（训练后量化） | 本项目（QAT） |
|--------|-------------------|---------------|
| 何时量化 | 模型训完后再压精度 | **训练过程中**前向就做量化 |
| 权重存储 | 通常先训 fp32，再校准 | 内存里仍存 fp32「影子权重」，前向用量化后的值 |
| 能否反传更新 | 一般不再端到端训练量化误差 | 用 **STE** 把梯度回传到影子权重，继续学习 |
| 目的 | 快速部署 | 让网络适应量化噪声，低比特精度更稳 |

简要流程：

```text
训练时每个 forward：
  浮点权重 w  ──量化──►  w_q  ──参与计算──►  loss
       ▲                    │
       └──── STE 梯度近似 ───┘
推理/对比时同样走量化前向（本实现训练与评估用同一套 QuantLinear）
```

注意：这里是 **仿真量化（fake quantization）**——整数码点算完后又乘回 `scale`，用浮点模拟低比特效果，并没有真的换成 int8 TensorCore 内核。对比的是「精度掉多少」，不是「推理吞吐提多少」。

---

## 2. 前向整体流程（非 fp32）

`QuantLinear.forward` 大致是：

```text
输入 x
  │
  ▼
LayerNorm(x)          # 先归一化，稳定量化范围
  │
  ▼
激活量化（int8）       # 可选，默认开启
  │  STE: x_use = x + (x_q - x).detach()
  ▼
权重量化（按 precision）
  │  STE: w_use = w + (w_q - w).detach()
  ▼
y = x_use @ w_use^T + bias
```

`fp32` 模式跳过以上全部，直接普通 `nn.Linear`。

分类头 `head` 在 `vit_model.py` 里固定为 fp32，不参与主干量化对比。

---

## 3. 激活如何量化？

实现：`_activation_quant_int8`

### 策略

- **精度**：固定 **int8**
- **粒度**：**per-token**（对最后一维 `dim=-1` 各自算 scale）
- **方式**：对称 absmax（按该 token 绝对值最大值标定）

### 公式

对每个 token 向量 \(x\)：

\[
s_x = \frac{127}{\max(|x|) + \varepsilon},\quad
x_q = \mathrm{clip}\!\left(\mathrm{round}(x \cdot s_x),\,-128,\,127\right) / s_x
\]

代码等价写法：

```python
scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
y = (x * scale).round().clamp_(-128, 127) / scale
```

### 码点范围

| 项目 | 值 |
|------|-----|
| 整数码点 | **[-128, 127]** |
| 档位数 | 256 |
| 返回值形态 | 已反量化回浮点（fake quant） |

在 `int8 / int4 / int2 / bit158` 四种模式下，**激活都用这套 int8**；只有权重比特数在变。

---

## 4. 权重如何量化？

### 4.1 int8 / int4 / int2：对称均匀量化

实现：`_weight_quant_uniform(w, bits)`

- **粒度**：整层权重一个 scale（tensor-wise absmax）
- **对称**：正负共用同一 `scale`，零点为 0

公式：

\[
q_{\max} = 2^{bits-1} - 1,\quad
s_w = \frac{\max(|w|)}{q_{\max}},\quad
q = \mathrm{clip}\!\left(\mathrm{round}(w / s_w),\,-q_{\max},\,q_{\max}\right),\quad
w_q = q \cdot s_w
\]

| 精度 | bits | \(q_{\max}\) | 整数码点范围 | 档位数 |
|------|------|--------------|--------------|--------|
| int8 | 8 | 127 | **[-127, 127]** | 255 |
| int4 | 4 | 7 | **[-7, 7]** | 15 |
| int2 | 2 | 1 | **[-1, 1]** | 3 |

说明：对称 int8 常用 `[-127,127]`，故意不用 `-128`，保证正负档位对称。

### 4.2 bit158：三值量化（≈1.58 bit）

实现：`_weight_quant_bit158`

灵感来自 BitNet b1.58：每个权重只取三个离散水平。

\[
s = \mathrm{mean}(|w|),\quad
q = \mathrm{clip}\!\left(\mathrm{round}(w / s),\,-1,\,1\right),\quad
w_q = q \cdot s
\]

| 项目 | 值 |
|------|-----|
| 整数码点 | **{-1, 0, 1}** |
| 实际权重水平 | **{-s, 0, +s}** |
| 有效比特 | \(\log_2 3 \approx 1.585\) |

### 4.3 int2 与 bit158 的区别

两者码点集合都是 `{-1,0,1}`，但 **scale 定义不同**：

| | int2 | bit158 |
|--|------|--------|
| scale | \(\max(|w|) / 1\) | \(\mathrm{mean}(|w|)\) |
| 含义 | 均匀 2bit 对称量化 | BitNet 风格三值量化 |
| 典型效果 | 更「贴」极值 | 更偏平均幅度，常与 absmean 训练搭配 |

---

## 5. STE（直通估计）是怎么工作的？

`round` / `clamp` 导数几乎处处为 0，若直接反传，权重学不动。因此用 STE：

```python
x_use = x_norm + (x_q - x_norm).detach()
w_use = self.weight + (w_q - self.weight).detach()
```

含义：

- **前向**：用量化后的 `x_q`、`w_q`
- **反向**：把对 `x_use` / `w_use` 的梯度，近似当成对 `x_norm` / `self.weight` 的梯度（跳过量化非线性）

因此：

- 参数里保存的仍是 **fp32 影子权重** `self.weight`
- 每次 forward 现算量化版本参与 matmul
- 优化器更新的是影子权重 → 这就是典型的 QAT / fake-quant 训练

---

## 6. 一张图串起来

```text
                    ┌─────────────────────────────┐
                    │  self.weight (fp32 影子权重) │
                    └──────────────┬──────────────┘
                                   │
                    quantize (int8/4/2 或 bit158)
                                   │
                                   ▼
                              w_q (fake)
                                   │
x ──► LayerNorm ──► act int8 ──► x_q ──► Linear(x_q, w_q) ──► y
              │                    ▲
              └──── STE 回传 ──────┘
```

---

## 7. 和「真·部署量化」的差异（阅读时注意）

1. **仿真 vs 真整数算子**：这里全程仍是浮点 matmul，只是数值被限制在离散格子上。
2. **未做校准集 PTQ**：没有单独的 `calibrate` 阶段；scale 每个 forward 现算。
3. **未量化全部模块**：PatchEmbed 卷积、LayerNorm、分类头等仍是 fp32。
4. **对比目标**：看不同权重比特下，同一数据与训练设定下的 **验证准确率** 差异。

---

## 8. 相关代码索引

| 内容 | 位置 |
|------|------|
| 激活 int8 | `quant_linear.py` → `_activation_quant_int8` |
| 权重均匀量化 | `quant_linear.py` → `_weight_quant_uniform` |
| 1.58bit 三值 | `quant_linear.py` → `_weight_quant_bit158` |
| STE + 前向 | `quant_linear.py` → `QuantLinear.forward` |
| 哪些层用量化 | `vit_model.py` → `Attention` / `Mlp` 中的 `QuantLinear` |
