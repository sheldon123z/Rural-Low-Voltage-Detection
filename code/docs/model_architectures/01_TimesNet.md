# TimesNet 模型架构文档

> **模型名称**: TimesNet
> **论文**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
> **发表**: ICLR 2023
> **参数量**: ~4.7M

---

## 1. 模型概述

TimesNet 是一种创新的时间序列分析模型，核心思想是将 **1D 时序信号转换为 2D 张量**，利用 2D 卷积捕获 **周期内变化 (intra-period)** 和 **周期间变化 (inter-period)** 两种时序模式。

### 核心创新

1. **FFT 周期发现**: 使用快速傅里叶变换自动发现时序数据中的主要周期
2. **1D→2D 时序转换**: 根据发现的周期将 1D 序列重塑为 2D 张量
3. **Inception 2D 卷积**: 使用多尺度 2D 卷积同时捕获周期内和周期间的模式
4. **自适应周期融合**: 使用 softmax 权重融合多个周期分支的结果

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TimesNet Model                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, T, C]                                                           │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │          Instance Normalization          │                               │
│  │     x = (x - mean) / std                 │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │           Data Embedding                 │                               │
│  │   [B, T, C] → [B, T, d_model]           │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ╔═════════════════════════════════════════╗                               │
│  ║         TimesBlock × e_layers           ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │      FFT Period Discovery          │  ║                               │
│  ║  │         ↓                          │  ║                               │
│  ║  │  ┌──────────────────────────────┐ │  ║                               │
│  ║  │  │ Period 1  Period 2  Period k │ │  ║                               │
│  ║  │  │    ↓         ↓         ↓     │ │  ║                               │
│  ║  │  │ 1D→2D    1D→2D     1D→2D     │ │  ║                               │
│  ║  │  │    ↓         ↓         ↓     │ │  ║                               │
│  ║  │  │ Conv2D   Conv2D    Conv2D    │ │  ║                               │
│  ║  │  │    ↓         ↓         ↓     │ │  ║                               │
│  ║  │  │ 2D→1D    2D→1D     2D→1D     │ │  ║                               │
│  ║  │  └──────────────────────────────┘ │  ║                               │
│  ║  │         ↓                          │  ║                               │
│  ║  │  Adaptive Aggregation (softmax)    │  ║                               │
│  ║  │         ↓                          │  ║                               │
│  ║  │    Residual Connection             │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ╚═════════════════════════════════════════╝                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │           Layer Normalization            │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │         Output Projection                │                               │
│  │   [B, T, d_model] → [B, T, c_out]       │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │         De-Normalization                 │                               │
│  │     output = output * std + mean         │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  Output: [B, T, c_out]                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. XML 结构表示

```xml
<?xml version="1.0" encoding="UTF-8"?>
<model name="TimesNet" type="time_series_anomaly_detection">
    <metadata>
        <paper>TimesNet: Temporal 2D-Variation Modeling</paper>
        <venue>ICLR 2023</venue>
        <parameters>~4.7M</parameters>
    </metadata>

    <architecture>
        <input shape="[B, T, C]" description="Batch, Sequence Length, Channels"/>

        <preprocessing name="InstanceNormalization">
            <operation>mean = x.mean(dim=1, keepdim=True)</operation>
            <operation>std = sqrt(var(x) + 1e-5)</operation>
            <operation>x_norm = (x - mean) / std</operation>
        </preprocessing>

        <layer name="DataEmbedding" type="embedding">
            <input_dim>enc_in</input_dim>
            <output_dim>d_model</output_dim>
            <components>
                <component name="ValueEmbedding" type="Conv1D"/>
                <component name="PositionalEmbedding" type="Sinusoidal"/>
                <component name="TemporalEmbedding" type="Learned" optional="true"/>
            </components>
        </layer>

        <block name="TimesBlock" repeat="e_layers">
            <module name="FFT_for_Period">
                <description>Discover top-k periods using FFT</description>
                <input>x [B, T, d_model]</input>
                <output>period_list [k], period_weight [B, k]</output>
                <algorithm>
                    <step>xf = rfft(x, dim=1)</step>
                    <step>amplitude = abs(xf).mean(0).mean(-1)</step>
                    <step>top_k_frequencies = topk(amplitude, k)</step>
                    <step>periods = T / frequencies</step>
                </algorithm>
            </module>

            <parallel_branches count="k" name="PeriodBranches">
                <branch id="i">
                    <reshape name="1D_to_2D">
                        <input>[B, T, d_model]</input>
                        <output>[B, d_model, T/period_i, period_i]</output>
                    </reshape>

                    <convolution name="Inception_Block_V1">
                        <kernels>6</kernels>
                        <in_channels>d_model</in_channels>
                        <out_channels>d_ff</out_channels>
                    </convolution>

                    <activation type="GELU"/>

                    <convolution name="Inception_Block_V1">
                        <in_channels>d_ff</in_channels>
                        <out_channels>d_model</out_channels>
                    </convolution>

                    <reshape name="2D_to_1D">
                        <input>[B, d_model, T/period_i, period_i]</input>
                        <output>[B, T, d_model]</output>
                    </reshape>
                </branch>
            </parallel_branches>

            <aggregation name="AdaptiveAggregation">
                <weights>softmax(period_weight)</weights>
                <operation>output = sum(branches * weights)</operation>
            </aggregation>

            <residual>output = output + x</residual>
        </block>

        <layer name="LayerNorm" type="normalization">
            <normalized_shape>d_model</normalized_shape>
        </layer>

        <layer name="Projection" type="linear">
            <input_dim>d_model</input_dim>
            <output_dim>c_out</output_dim>
        </layer>

        <postprocessing name="DeNormalization">
            <operation>output = output * std + mean</operation>
        </postprocessing>

        <output shape="[B, T, c_out]" description="Reconstructed sequence"/>
    </architecture>

    <hyperparameters>
        <param name="seq_len" default="100" description="Input sequence length"/>
        <param name="d_model" default="64" description="Model dimension"/>
        <param name="d_ff" default="128" description="Feed-forward dimension"/>
        <param name="e_layers" default="2" description="Number of TimesBlock layers"/>
        <param name="top_k" default="5" description="Number of periods to discover"/>
        <param name="num_kernels" default="6" description="Inception kernels"/>
        <param name="dropout" default="0.1" description="Dropout rate"/>
    </hyperparameters>
</model>
```

---

## 4. 核心组件详解

### 4.1 FFT 周期发现 (FFT_for_Period)

```
输入序列 x: [B, T, C]
     │
     ▼
┌─────────────────────┐
│   FFT 变换          │  xf = rfft(x, dim=1)
│   [B, T, C] →       │
│   [B, T//2+1, C]    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   幅度谱计算        │  amplitude = |xf|.mean(0).mean(-1)
│   [T//2+1]          │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Top-K 频率选择    │  top_k = topk(amplitude, k)
│   [k]               │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   周期计算          │  period = T / frequency
│   [k]               │
└─────────────────────┘
```

### 4.2 1D→2D→1D 时序变换

```
1D 输入: [B, T, d_model]     例: [32, 100, 64]
     │
     │  period = 20
     ▼
2D 重塑: [B, d_model, T/p, p]  →  [32, 64, 5, 20]
     │
     │  行: 周期间变化 (inter-period)
     │  列: 周期内变化 (intra-period)
     ▼
┌─────────────────────────────────────┐
│         2D Inception 卷积           │
│                                     │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│   │ 1×1 │ │ 3×3 │ │ 5×5 │ │ 7×7 │  │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
│      │       │       │       │      │
│      └───────┴───────┴───────┘      │
│              │                       │
│          Concatenate                 │
└─────────────────────────────────────┘
     │
     ▼
1D 输出: [B, T, d_model]
```

### 4.3 自适应周期聚合

```
Period 1: ────────►┐
Period 2: ────────►├───► Softmax(weights) ───► Weighted Sum ───► Output
Period k: ────────►┘
                    ↑
              FFT Amplitudes
```

---

## 5. 数据流示例

假设参数: `B=32, T=100, C=16, d_model=64, top_k=3`

| 步骤 | 操作 | 输入形状 | 输出形状 |
|:----:|------|:--------:|:--------:|
| 1 | Input | - | [32, 100, 16] |
| 2 | Normalization | [32, 100, 16] | [32, 100, 16] |
| 3 | Embedding | [32, 100, 16] | [32, 100, 64] |
| 4 | FFT Period | [32, 100, 64] | periods=[20,25,50], weights=[32,3] |
| 5a | Reshape (p=20) | [32, 100, 64] | [32, 64, 5, 20] |
| 5b | Conv2D | [32, 64, 5, 20] | [32, 64, 5, 20] |
| 5c | Reshape back | [32, 64, 5, 20] | [32, 100, 64] |
| 6 | Aggregate | 3×[32, 100, 64] | [32, 100, 64] |
| 7 | Residual | [32, 100, 64] | [32, 100, 64] |
| 8 | Projection | [32, 100, 64] | [32, 100, 16] |
| 9 | De-Norm | [32, 100, 16] | [32, 100, 16] |

---

## 6. 异常检测原理

TimesNet 采用 **重构误差** 方法进行异常检测：

```
┌────────────────┐      ┌─────────────┐      ┌────────────────┐
│  输入序列 x    │ ───► │  TimesNet   │ ───► │  重构序列 x'   │
└────────────────┘      └─────────────┘      └────────────────┘
        │                                            │
        │                                            │
        └───────────────┬───────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ 重构误差计算   │
                │ err = |x - x'| │
                └───────────────┘
                        │
                        ▼
                ┌───────────────┐
                │   阈值判定     │  if err > threshold: anomaly
                └───────────────┘
```

**阈值计算**:
```python
threshold = np.percentile(train_errors, 100 - anomaly_ratio)
```

---

## 7. 关键代码结构

```python
class Model(nn.Module):
    def __init__(self, configs):
        # 数据嵌入层
        self.enc_embedding = DataEmbedding(...)

        # TimesBlock 堆叠
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # 输出投影
        self.projection = nn.Linear(d_model, c_out)

    def anomaly_detection(self, x_enc):
        # 1. 归一化
        # 2. 嵌入
        # 3. TimesBlock × e_layers
        # 4. 投影
        # 5. 反归一化
        return reconstructed
```

---

## 8. 性能指标

| 数据集 | Accuracy | Precision | Recall | F1 |
|:------:|:--------:|:---------:|:------:|:--:|
| PSM | 0.9855 | 0.9848 | 0.9625 | **0.9735** |
| RuralVoltage | 0.8259 | 0.8939 | 0.4582 | 0.6059 |
| SMD | 0.9846 | 0.7821 | 0.8719 | 0.8246 |

---

*文档生成时间: 2026-02-02*
