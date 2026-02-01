# MTSTimesNet 模型架构文档

> **模型名称**: MTSTimesNet (Multi-Scale TimesNet)
> **基础**: TimesNet (ICLR 2023)
> **创新**: 多尺度时序并行建模
> **参数量**: ~6M

---

## 1. 模型概述

MTSTimesNet 是针对 **多尺度时序模式** 优化的 TimesNet 变体。核心创新是通过 **并行多尺度分支** 同时捕获短期、中期、长期的时序变化，并使用 **自适应融合门** 动态组合不同尺度的特征。

### 核心创新

1. **多尺度并行分支**: 短期 (2-20)、中期 (20-60)、长期 (60-200) 三个尺度
2. **尺度专属 TimesBlock**: 每个尺度有独立的周期发现和卷积处理
3. **自适应融合门**: 根据输入动态调整尺度权重
4. **跨尺度连接**: 促进不同尺度之间的信息流动

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MTSTimesNet Model                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, T, C]                                                           │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │          Instance Normalization          │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │           Data Embedding                 │                               │
│  │   [B, T, C] → [B, T, d_model]           │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ╔═════════════════════════════════════════════════════════════════════╗   │
│  ║              ★ Multi-Scale Parallel Branches                        ║   │
│  ║                                                                      ║   │
│  ║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       ║   │
│  ║  │   Short Scale   │ │  Medium Scale   │ │   Long Scale    │       ║   │
│  ║  │   (2-20 周期)   │ │  (20-60 周期)   │ │  (60-200 周期)  │       ║   │
│  ║  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤       ║   │
│  ║  │                 │ │                 │ │                 │       ║   │
│  ║  │ FFT Discovery   │ │ FFT Discovery   │ │ FFT Discovery   │       ║   │
│  ║  │ (min=2, max=20) │ │ (min=20,max=60) │ │ (min=60,max=200)│       ║   │
│  ║  │       ↓         │ │       ↓         │ │       ↓         │       ║   │
│  ║  │ ScaleSpecific   │ │ ScaleSpecific   │ │ ScaleSpecific   │       ║   │
│  ║  │  TimesBlock     │ │  TimesBlock     │ │  TimesBlock     │       ║   │
│  ║  │ × e_layers      │ │ × e_layers      │ │ × e_layers      │       ║   │
│  ║  │       ↓         │ │       ↓         │ │       ↓         │       ║   │
│  ║  │   short_out     │ │  medium_out     │ │   long_out      │       ║   │
│  ║  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘       ║   │
│  ║           │                   │                   │                 ║   │
│  ║           └───────────────────┼───────────────────┘                 ║   │
│  ║                               ↓                                      ║   │
│  ║  ┌───────────────────────────────────────────────────────────────┐  ║   │
│  ║  │              ★ Cross-Scale Connection                         │  ║   │
│  ║  │                                                                │  ║   │
│  ║  │  cross_scale_conv: Conv1d(d_model*3, d_model, kernel=1)       │  ║   │
│  ║  │                                                                │  ║   │
│  ║  │  cross_features = Concat([short, medium, long])               │  ║   │
│  ║  │  cross_out = conv(cross_features) + short + medium + long     │  ║   │
│  ║  └───────────────────────────────────────────────────────────────┘  ║   │
│  ║                               ↓                                      ║   │
│  ║  ┌───────────────────────────────────────────────────────────────┐  ║   │
│  ║  │              ★ Adaptive Fusion Gate                           │  ║   │
│  ║  │                                                                │  ║   │
│  ║  │  concat_features = [short_out, medium_out, long_out]          │  ║   │
│  ║  │                               ↓                                │  ║   │
│  ║  │  ┌─────────────────────────────────────────────────────┐      │  ║   │
│  ║  │  │  gate_weights = Softmax(Linear(mean(concat)))       │      │  ║   │
│  ║  │  │                                                      │      │  ║   │
│  ║  │  │  w_short, w_medium, w_long = gate_weights           │      │  ║   │
│  ║  │  └─────────────────────────────────────────────────────┘      │  ║   │
│  ║  │                               ↓                                │  ║   │
│  ║  │  fused = w_short * short + w_medium * medium + w_long * long  │  ║   │
│  ║  └───────────────────────────────────────────────────────────────┘  ║   │
│  ╚═════════════════════════════════════════════════════════════════════╝   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │           Layer Normalization            │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │         Output Projection                │                               │
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
<model name="MTSTimesNet" type="multi_scale_time_series_analysis" base="TimesNet">
    <metadata>
        <innovation>Multi-Scale Parallel Temporal Modeling</innovation>
        <target_domain>Complex Multi-Scale Anomaly Detection</target_domain>
        <parameters>~6M</parameters>
    </metadata>

    <architecture>
        <input shape="[B, T, C]" description="Time series data"/>

        <preprocessing name="InstanceNormalization" inherited="true"/>

        <layer name="DataEmbedding" type="embedding" inherited="true"/>

        <multi_scale_branches>
            <scale_configs>
                <scale name="short" min_period="2" max_period="20">
                    <description>Rapid fluctuations, transient events</description>
                    <temporal_scope>Seconds to minutes</temporal_scope>
                </scale>
                <scale name="medium" min_period="20" max_period="60">
                    <description>Medium-term patterns, load variations</description>
                    <temporal_scope>Minutes to hours</temporal_scope>
                </scale>
                <scale name="long" min_period="60" max_period="200">
                    <description>Long-term trends, daily patterns</description>
                    <temporal_scope>Hours to days</temporal_scope>
                </scale>
            </scale_configs>

            <branch name="short_scale">
                <module name="ScaleSpecificTimesBlock" repeat="e_layers">
                    <fft_period_discovery>
                        <min_period>2</min_period>
                        <max_period>20</max_period>
                        <top_k>5</top_k>
                    </fft_period_discovery>

                    <period_branches count="top_k">
                        <reshape>1D → 2D</reshape>
                        <convolution type="Inception_Block_V1">
                            <kernels>6</kernels>
                        </convolution>
                        <reshape>2D → 1D</reshape>
                    </period_branches>

                    <aggregation>Softmax weighted sum</aggregation>
                    <residual>output = output + x</residual>
                </module>
            </branch>

            <branch name="medium_scale">
                <module name="ScaleSpecificTimesBlock" repeat="e_layers">
                    <fft_period_discovery>
                        <min_period>20</min_period>
                        <max_period>60</max_period>
                        <top_k>5</top_k>
                    </fft_period_discovery>
                    <!-- Same structure as short_scale -->
                </module>
            </branch>

            <branch name="long_scale">
                <module name="ScaleSpecificTimesBlock" repeat="e_layers">
                    <fft_period_discovery>
                        <min_period>60</min_period>
                        <max_period>200</max_period>
                        <top_k>5</top_k>
                    </fft_period_discovery>
                    <!-- Same structure as short_scale -->
                </module>
            </branch>
        </multi_scale_branches>

        <module name="CrossScaleConnection" new="true">
            <description>Information flow between scales</description>
            <convolution type="Conv1d">
                <in_channels>d_model * 3</in_channels>
                <out_channels>d_model</out_channels>
                <kernel_size>1</kernel_size>
            </convolution>
            <operation>
                <step>concat = torch.cat([short, medium, long], dim=-1)</step>
                <step>cross = conv(concat.transpose(1,2)).transpose(1,2)</step>
                <step>output = cross + short + medium + long</step>
            </operation>
        </module>

        <module name="AdaptiveFusionGate" new="true">
            <description>Dynamic scale weighting based on input</description>

            <gate_network type="MLP">
                <input_dim>d_model * 3</input_dim>
                <hidden_dim>d_model</hidden_dim>
                <output_dim>3</output_dim>
                <activation>ReLU (hidden), Softmax (output)</activation>
            </gate_network>

            <algorithm>
                <step>concat = torch.cat([short, medium, long], dim=-1)</step>
                <step>pooled = concat.mean(dim=1)</step>
                <step>gate_weights = softmax(gate_net(pooled))</step>
                <step>w_short, w_medium, w_long = gate_weights.split(1, dim=-1)</step>
                <step>fused = w_short * short + w_medium * medium + w_long * long</step>
            </algorithm>

            <advantage>
                <point>Automatically emphasizes relevant scales</point>
                <point>Adapts to different input characteristics</point>
                <point>Handles varying anomaly temporal spans</point>
            </advantage>
        </module>

        <layer name="LayerNorm" inherited="true"/>
        <layer name="Projection" inherited="true"/>
        <postprocessing name="DeNormalization" inherited="true"/>

        <output shape="[B, T, c_out]"/>
    </architecture>

    <hyperparameters inherited="TimesNet">
        <param name="scale_configs" description="Scale-specific period ranges">
            <short min="2" max="20"/>
            <medium min="20" max="60"/>
            <long min="60" max="200"/>
        </param>
        <param name="fusion_type" default="adaptive" options="adaptive|fixed|learned"/>
    </hyperparameters>
</model>
```

---

## 4. 核心创新详解

### 4.1 多尺度并行分支

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Scale Parallel Branches                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: x [B, T, d_model]                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Short Scale                         │   │
│  │                      (2-20 周期)                         │   │
│  │                                                          │   │
│  │  ████████████████████  快速波动                          │   │
│  │  │││││││││││││││││││   瞬时事件                          │   │
│  │  ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼   开关操作                          │   │
│  │                                                          │   │
│  │  适用: 瞬态异常、快速负荷变化                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Medium Scale                         │   │
│  │                     (20-60 周期)                         │   │
│  │                                                          │   │
│  │  ████████████████████  中期模式                          │   │
│  │       ∧       ∧        负荷变化                          │   │
│  │      / \     / \       设备周期                          │   │
│  │                                                          │   │
│  │  适用: 设备运行周期、中期负荷波动                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Long Scale                          │   │
│  │                     (60-200 周期)                        │   │
│  │                                                          │   │
│  │  ████████████████████  长期趋势                          │   │
│  │            ∧           日周期模式                        │   │
│  │           / \          季节变化                          │   │
│  │          /   \                                           │   │
│  │                                                          │   │
│  │  适用: 日负荷曲线、长期趋势变化                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 尺度范围约束的 FFT

```python
# 标准 TimesNet FFT
periods = T // frequencies  # 无约束

# MTSTimesNet 尺度约束 FFT
def FFT_for_Period_with_scale(x, k, min_period, max_period):
    xf = torch.fft.rfft(x, dim=1)
    freq_list = torch.fft.rfftfreq(T)

    # 计算有效频率范围
    min_freq = 1 / max_period
    max_freq = 1 / min_period

    # 过滤频率
    valid_mask = (freq_list >= min_freq) & (freq_list <= max_freq)
    valid_amp = amplitude[valid_mask]

    # 在有效范围内选择 Top-K
    top_k_periods = ...

    return periods, weights
```

### 4.3 自适应融合门

```
┌─────────────────────────────────────────────────────────────────┐
│                    Adaptive Fusion Gate                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Short Output ───┐                                              │
│                  │                                              │
│  Medium Output ──┼───► Concat ───► Mean Pool ───► MLP ───►     │
│                  │                                              │
│  Long Output ────┘                                              │
│                                                                 │
│                                         ↓                       │
│                                  ┌──────────────┐               │
│                                  │   Softmax    │               │
│                                  │              │               │
│                                  │ [w_s, w_m, w_l]              │
│                                  └──────────────┘               │
│                                         │                       │
│                                         ↓                       │
│                                                                 │
│  Fused = w_s × Short + w_m × Medium + w_l × Long               │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│  示例权重分布:                                                  │
│                                                                 │
│  瞬态异常时:    w_s=0.7  w_m=0.2  w_l=0.1  (短尺度主导)        │
│  周期负荷时:    w_s=0.2  w_m=0.5  w_l=0.3  (中尺度主导)        │
│  长期趋势时:    w_s=0.1  w_m=0.3  w_l=0.6  (长尺度主导)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 跨尺度连接

```
Short  ────┐
           │
Medium ────┼───► Concat ───► Conv1d(3d, d) ───► cross_out
           │                      │
Long   ────┘                      │
                                  ↓
Final = cross_out + short + medium + long

作用:
- 促进不同尺度之间的信息交流
- 学习跨尺度的特征组合
- 增强模型对复杂异常的捕获能力
```

---

## 5. 数据流示例

假设参数: `B=32, T=100, C=16, d_model=64`

| 步骤 | 操作 | 输出形状 |
|:----:|------|:--------:|
| 1 | Input | [32, 100, 16] |
| 2 | Embedding | [32, 100, 64] |
| 3a | Short Scale (top_k=5) | [32, 100, 64] |
| 3b | Medium Scale (top_k=5) | [32, 100, 64] |
| 3c | Long Scale (top_k=5) | [32, 100, 64] |
| 4 | Cross-Scale Connection | [32, 100, 64] |
| 5 | Adaptive Fusion | [32, 100, 64] |
| 6 | Projection | [32, 100, 16] |

---

## 6. 参数配置

### 6.1 尺度配置

| 尺度 | 最小周期 | 最大周期 | 适用异常类型 |
|------|:--------:|:--------:|-------------|
| Short | 2 | 20 | 瞬态、快速波动 |
| Medium | 20 | 60 | 设备周期、负荷变化 |
| Long | 60 | 200 | 日周期、长期趋势 |

### 6.2 调优建议

```python
# 根据数据特性调整尺度范围
SCALE_CONFIGS = {
    # 高频数据 (采样率 > 1Hz)
    'high_freq': {
        'short': (2, 50),
        'medium': (50, 200),
        'long': (200, 500)
    },
    # 低频数据 (采样率 < 0.1Hz)
    'low_freq': {
        'short': (2, 10),
        'medium': (10, 30),
        'long': (30, 100)
    }
}
```

---

## 7. 性能对比

| 数据集 | TimesNet F1 | MTSTimesNet F1 | 说明 |
|:------:|:-----------:|:--------------:|:----:|
| PSM | 0.9735 | ~0.97 | 多尺度无显著优势 |
| RuralVoltage | 0.6059 | ~0.62 | 多尺度略有提升 |

**适用场景**: 当异常在多个时间尺度上表现时效果最佳

---

## 8. 使用示例

```bash
# 基础训练
python run.py --is_training 1 \
    --model MTSTimesNet \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 200 \
    --d_model 64 \
    --e_layers 2

# 注意: seq_len 需要足够长以支持长尺度
# 建议 seq_len >= max(long_scale.max_period) = 200
```

---

*文档生成时间: 2026-02-02*
