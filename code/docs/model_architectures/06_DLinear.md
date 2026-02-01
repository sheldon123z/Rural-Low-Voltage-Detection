# DLinear 模型架构文档

> **模型名称**: DLinear (Decomposition Linear)
> **论文**: Are Transformers Effective for Time Series Forecasting?
> **发表**: AAAI 2023
> **参数量**: ~20K (极轻量级)

---

## 1. 模型概述

DLinear 是一个 **极简但有效** 的时间序列模型。核心思想是将时间序列分解为 **趋势 (Trend)** 和 **季节性 (Seasonal)** 两个部分，分别用简单的线性层处理，然后合并。

### 核心创新

1. **序列分解**: 使用移动平均将序列分解为趋势和季节性
2. **双线性分支**: 独立的线性层分别处理两个分量
3. **极简架构**: 无注意力、无卷积、无循环
4. **高效推理**: 参数量仅 ~20K，推理速度极快

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DLinear Model                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, T, C]                                                           │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │          Instance Normalization          │                               │
│  │     x = (x - mean) / std                 │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ╔═════════════════════════════════════════╗                               │
│  ║         ★ Series Decomposition          ║                               │
│  ║                                          ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │       Moving Average Filter        │  ║                               │
│  ║  │       (kernel = 25, pad = 12)     │  ║                               │
│  ║  │                                    │  ║                               │
│  ║  │  trend = AvgPool1d(x, kernel=25)  │  ║                               │
│  ║  │  seasonal = x - trend             │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ╚═════════════════════════════════════════╝                               │
│         │                                                                   │
│         ├─────────────────────────────┐                                    │
│         ↓                             ↓                                    │
│  ┌─────────────────┐           ┌─────────────────┐                        │
│  │   Trend Branch   │           │ Seasonal Branch │                        │
│  │                  │           │                  │                        │
│  │  [B, T, C]       │           │  [B, T, C]       │                        │
│  │       ↓          │           │       ↓          │                        │
│  │  Permute         │           │  Permute         │                        │
│  │  [B, C, T]       │           │  [B, C, T]       │                        │
│  │       ↓          │           │       ↓          │                        │
│  │  ┌──────────┐   │           │  ┌──────────┐   │                        │
│  │  │ Linear   │   │           │  │ Linear   │   │                        │
│  │  │ (T → T)  │   │           │  │ (T → T)  │   │                        │
│  │  │ per      │   │           │  │ per      │   │                        │
│  │  │ channel  │   │           │  │ channel  │   │                        │
│  │  └──────────┘   │           │  └──────────┘   │                        │
│  │       ↓          │           │       ↓          │                        │
│  │  [B, C, T]       │           │  [B, C, T]       │                        │
│  │       ↓          │           │       ↓          │                        │
│  │  Permute         │           │  Permute         │                        │
│  │  [B, T, C]       │           │  [B, T, C]       │                        │
│  └────────┬─────────┘           └────────┬─────────┘                        │
│           │                               │                                 │
│           └───────────┬───────────────────┘                                │
│                       ↓                                                     │
│              ┌─────────────────┐                                           │
│              │       Add       │                                           │
│              │ trend + seasonal│                                           │
│              └─────────────────┘                                           │
│                       ↓                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │         De-Normalization                 │                               │
│  │     output = output * std + mean         │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  Output: [B, T, C]                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. XML 结构表示

```xml
<?xml version="1.0" encoding="UTF-8"?>
<model name="DLinear" type="time_series_anomaly_detection">
    <metadata>
        <paper>Are Transformers Effective for Time Series Forecasting?</paper>
        <venue>AAAI 2023</venue>
        <parameters>~20K</parameters>
        <philosophy>Simplicity beats complexity for many time series tasks</philosophy>
    </metadata>

    <architecture>
        <input shape="[B, T, C]" description="Batch, Sequence Length, Channels"/>

        <preprocessing name="InstanceNormalization">
            <operation>mean = x.mean(dim=1, keepdim=True)</operation>
            <operation>std = sqrt(var(x) + 1e-5)</operation>
            <operation>x_norm = (x - mean) / std</operation>
        </preprocessing>

        <module name="series_decomp" type="Decomposition">
            <description>Separate trend and seasonal components</description>

            <moving_average_filter>
                <kernel_size default="25" description="Window size for smoothing"/>
                <padding>kernel_size // 2</padding>
                <type>AvgPool1d</type>
            </moving_average_filter>

            <decomposition>
                <trend>trend = moving_avg(x)</trend>
                <seasonal>seasonal = x - trend</seasonal>
            </decomposition>

            <physical_meaning>
                <trend>Low-frequency, slowly varying component</trend>
                <seasonal>High-frequency, periodic fluctuations</seasonal>
            </physical_meaning>
        </module>

        <parallel_branches>
            <branch name="Linear_Trend">
                <description>Process trend component</description>
                <operations>
                    <step>x = x.permute(0, 2, 1)  # [B, C, T]</step>
                    <step>x = Linear(x)  # Channel-wise linear</step>
                    <step>x = x.permute(0, 2, 1)  # [B, T, C]</step>
                </operations>
                <linear_layer>
                    <input_dim>seq_len</input_dim>
                    <output_dim>seq_len</output_dim>
                    <weight_shape>[C, seq_len, seq_len]</weight_shape>
                </linear_layer>
            </branch>

            <branch name="Linear_Seasonal">
                <description>Process seasonal component</description>
                <operations>
                    <step>x = x.permute(0, 2, 1)  # [B, C, T]</step>
                    <step>x = Linear(x)  # Channel-wise linear</step>
                    <step>x = x.permute(0, 2, 1)  # [B, T, C]</step>
                </operations>
                <linear_layer>
                    <input_dim>seq_len</input_dim>
                    <output_dim>seq_len</output_dim>
                    <weight_shape>[C, seq_len, seq_len]</weight_shape>
                </linear_layer>
            </branch>
        </parallel_branches>

        <aggregation>
            <operation>output = trend_out + seasonal_out</operation>
        </aggregation>

        <postprocessing name="DeNormalization">
            <operation>output = output * std + mean</operation>
        </postprocessing>

        <output shape="[B, T, C]" description="Reconstructed sequence"/>
    </architecture>

    <hyperparameters>
        <param name="seq_len" default="100" description="Input/output sequence length"/>
        <param name="enc_in" default="25" description="Number of input channels"/>
        <param name="kernel_size" default="25" description="Moving average kernel"/>
        <param name="individual" default="false" description="Individual weights per channel"/>
    </hyperparameters>

    <complexity_analysis>
        <parameters>O(C × T²)</parameters>
        <time_complexity>O(B × C × T)</time_complexity>
        <space_complexity>O(C × T²)</space_complexity>
        <comparison_with_transformer>
            <transformer_params>O(T² + d_model²)</transformer_params>
            <dlinear_advantage>No attention, no positional encoding</dlinear_advantage>
        </comparison_with_transformer>
    </complexity_analysis>
</model>
```

---

## 4. 核心组件详解

### 4.1 序列分解 (Series Decomposition)

```
原始信号 x:
────∧───∨───∧───∨───∧───∨────  (包含趋势 + 季节性)
     \   /     \   /     \
      \_/       \_/       \_/
              │
              ↓
     Moving Average (kernel=25)
              │
              ↓
趋势分量 trend:
────────────────────────────────  (平滑的低频部分)
        ╱                  ╲
       ╱                    ╲

季节性分量 seasonal = x - trend:
────∧───∨───∧───∨───∧───∨────  (去除趋势后的高频部分)
   ╱│╲ ╱│╲ ╱│╲ ╱│╲ ╱│╲ ╱│╲
  ╱ │ ╲│ ╲│ ╲│ ╲│ ╲│ ╲│ ╲
```

### 4.2 移动平均实现

```python
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        # 前后填充以保持长度
        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # 应用平均池化
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
```

### 4.3 通道独立线性层

```
输入: [B, T, C]
      │
      ↓
Permute: [B, C, T]
      │
      ↓
Linear: 对每个通道独立应用 T→T 的线性变换
      │
      │   Channel 0: [T] → Linear → [T]
      │   Channel 1: [T] → Linear → [T]
      │   ...
      │   Channel C-1: [T] → Linear → [T]
      │
      ↓
Permute: [B, T, C]

参数量: C × T × T (约 16 × 100 × 100 = 160K，但使用共享权重时更少)
```

---

## 5. 数据流示例

假设参数: `B=32, T=100, C=16`

| 步骤 | 操作 | 输入形状 | 输出形状 |
|:----:|------|:--------:|:--------:|
| 1 | Input | - | [32, 100, 16] |
| 2 | Normalization | [32, 100, 16] | [32, 100, 16] |
| 3 | Series Decomp | [32, 100, 16] | trend: [32, 100, 16], seasonal: [32, 100, 16] |
| 4a | Linear_Trend | [32, 100, 16] | [32, 100, 16] |
| 4b | Linear_Seasonal | [32, 100, 16] | [32, 100, 16] |
| 5 | Add | 2×[32, 100, 16] | [32, 100, 16] |
| 6 | De-Norm | [32, 100, 16] | [32, 100, 16] |

---

## 6. 与 TimesNet 的对比

| 特性 | DLinear | TimesNet |
|------|---------|----------|
| **参数量** | ~20K | ~4.7M |
| **核心操作** | 线性层 | FFT + 2D卷积 |
| **时序建模** | 分解 + 线性 | 周期发现 + 多尺度 |
| **计算复杂度** | O(T) | O(T log T + T²/p) |
| **训练速度** | ★★★★★ | ★★★☆☆ |
| **表达能力** | ★★★☆☆ | ★★★★★ |
| **适用场景** | 简单周期 | 复杂多周期 |

---

## 7. 优缺点分析

### 优点

```
✅ 极少参数量 (~20K vs TimesNet ~4.7M)
✅ 训练速度快 (无复杂操作)
✅ 推理延迟低 (适合实时应用)
✅ 不易过拟合 (简单架构)
✅ 易于理解和调试
```

### 缺点

```
❌ 表达能力有限 (仅线性变换)
❌ 无法捕获复杂的非线性模式
❌ 对多周期信号处理能力弱
❌ 缺乏显式的周期建模
```

---

## 8. 参数配置

### 8.1 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|:------:|------|---------:|
| `kernel_size` | 25 | 移动平均窗口 | 根据数据周期调整 |
| `individual` | False | 是否每通道独立权重 | True 增加参数但更灵活 |

### 8.2 移动平均窗口选择

```
kernel_size 选择原则:
- 太小: 无法有效分离趋势
- 太大: 过度平滑，丢失信息

经验值:
- 高频数据 (>1Hz): kernel_size = 51-101
- 中频数据 (0.1-1Hz): kernel_size = 25-51
- 低频数据 (<0.1Hz): kernel_size = 11-25
```

---

## 9. 异常检测性能

| 数据集 | DLinear F1 | TimesNet F1 | 差距 |
|:------:|:----------:|:-----------:|:----:|
| PSM | ~0.85 | 0.9735 | -12% |
| RuralVoltage | ~0.45 | 0.6059 | -26% |

**结论**: DLinear 作为轻量级基线有价值，但复杂异常检测推荐使用 TimesNet 系列

---

## 10. 使用示例

```bash
# 基础训练
python run.py --is_training 1 \
    --model DLinear \
    --data PSM \
    --root_path ./dataset/PSM \
    --seq_len 100 \
    --enc_in 25 --c_out 25 \
    --train_epochs 10 \
    --batch_size 32

# 农村电压数据
python run.py --is_training 1 \
    --model DLinear \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 100 \
    --train_epochs 10
```

---

## 11. 适用场景总结

| 场景 | 推荐度 | 说明 |
|------|:------:|------|
| 快速基线验证 | ★★★★★ | 快速建立性能基准 |
| 资源受限环境 | ★★★★★ | 极低计算需求 |
| 简单周期数据 | ★★★★☆ | 有效捕获简单模式 |
| 复杂异常检测 | ★★☆☆☆ | 建议使用 TimesNet |
| 实时推理 | ★★★★★ | 延迟极低 |

---

*文档生成时间: 2026-02-02*
