# VoltageTimesNet 模型架构文档

> **模型名称**: VoltageTimesNet
> **基础**: TimesNet (ICLR 2023)
> **创新**: 预设电网周期 + FFT 混合发现
> **参数量**: ~5M

---

## 1. 模型概述

VoltageTimesNet 是针对 **农村电网电压异常检测** 优化的 TimesNet 变体。核心创新是将 **领域知识（电网周期特性）** 融入周期发现过程，通过预设周期与 FFT 发现周期的混合策略提升检测效果。

### 核心创新

1. **预设周期混合**: 将电网固有周期（1分钟、5分钟、15分钟、1小时）与 FFT 发现的周期混合
2. **可配置权重**: preset_weight 参数控制预设周期与 FFT 周期的混合比例
3. **时域平滑卷积**: 额外的 1D 深度可分离卷积用于时域平滑
4. **电压信号优化**: 针对三相电压 (Va, Vb, Vc) 的相关性设计

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VoltageTimesNet Model                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, T, C]  (C=16: Va,Vb,Vc,Ia,Ib,Ic,P,Q,S,PF,THD×3,Freq,Unbal×2)   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │          Instance Normalization          │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │           Data Embedding                 │                               │
│  │   [B, T, 16] → [B, T, d_model]          │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│  ╔═════════════════════════════════════════╗                               │
│  ║      VoltageTimesBlock × e_layers       ║                               │
│  ║                                          ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │   Hybrid Period Discovery          │  ║                               │
│  ║  │                                    │  ║                               │
│  ║  │  ┌──────────┐    ┌──────────────┐ │  ║                               │
│  ║  │  │   FFT    │    │   Preset     │ │  ║                               │
│  ║  │  │ Periods  │    │   Periods    │ │  ║                               │
│  ║  │  │ (1-α)    │    │   (α=0.3)    │ │  ║                               │
│  ║  │  └────┬─────┘    └──────┬───────┘ │  ║                               │
│  ║  │       │                 │          │  ║                               │
│  ║  │       └────────┬────────┘          │  ║                               │
│  ║  │                ↓                   │  ║                               │
│  ║  │       ┌──────────────┐            │  ║                               │
│  ║  │       │ Mixed Periods │            │  ║                               │
│  ║  │       └──────────────┘            │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ║                   ↓                      ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │  Multi-Period 2D Convolution      │  ║                               │
│  ║  │  (Same as TimesNet)               │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ║                   ↓                      ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │  ★ Temporal Smoothing Conv1D      │  ║  ← 新增：时域平滑            │
│  ║  │  (Depthwise, kernel=3)            │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ║                   ↓                      ║                               │
│  ║         Residual Connection              ║                               │
│  ╚═════════════════════════════════════════╝                               │
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
<model name="VoltageTimesNet" type="voltage_anomaly_detection" base="TimesNet">
    <metadata>
        <innovation>Preset Period + FFT Hybrid Discovery</innovation>
        <target_domain>Rural Power Grid Voltage Monitoring</target_domain>
        <parameters>~5M</parameters>
    </metadata>

    <architecture>
        <input shape="[B, T, C]" description="Voltage monitoring data">
            <features>
                <feature index="0-2" name="Va,Vb,Vc" description="Three-phase voltage"/>
                <feature index="3-5" name="Ia,Ib,Ic" description="Three-phase current"/>
                <feature index="6-9" name="P,Q,S,PF" description="Power metrics"/>
                <feature index="10-12" name="THD_Va/Vb/Vc" description="Harmonic distortion"/>
                <feature index="13" name="Freq" description="Grid frequency"/>
                <feature index="14-15" name="V/I_unbalance" description="Unbalance factors"/>
            </features>
        </input>

        <preprocessing name="InstanceNormalization" inherited="true"/>

        <layer name="DataEmbedding" type="embedding" inherited="true"/>

        <block name="VoltageTimesBlock" repeat="e_layers">
            <module name="FFT_for_Period_Voltage" extends="FFT_for_Period">
                <description>Hybrid period discovery with preset periods</description>

                <preset_periods>
                    <period name="1min" value="60" description="Short-term fluctuations"/>
                    <period name="5min" value="300" description="Transient events"/>
                    <period name="15min" value="900" description="Load variations"/>
                    <period name="1h" value="3600" description="Hourly patterns"/>
                </preset_periods>

                <algorithm>
                    <step>fft_periods = FFT_for_Period(x, k)</step>
                    <step>valid_presets = filter(presets, p &lt;= seq_len/2)</step>
                    <step>n_preset = max(1, int(k * preset_weight))</step>
                    <step>n_fft = k - n_preset</step>
                    <step>final_periods = merge(fft_periods[:n_fft], valid_presets[:n_preset])</step>
                </algorithm>

                <parameter name="preset_weight" default="0.3" description="Ratio of preset periods"/>
            </module>

            <parallel_branches name="PeriodBranches" inherited="true">
                <!-- Same as TimesNet: 1D→2D→Conv2D→2D→1D -->
            </parallel_branches>

            <aggregation name="AdaptiveAggregation" inherited="true"/>

            <module name="TemporalSmoothingConv" new="true">
                <description>Depthwise 1D convolution for temporal smoothing</description>
                <type>Conv1D</type>
                <kernel_size>3</kernel_size>
                <padding>1</padding>
                <groups>d_model</groups>
            </module>

            <residual>output = output + x</residual>
        </block>

        <layer name="LayerNorm" inherited="true"/>
        <layer name="Projection" inherited="true"/>
        <postprocessing name="DeNormalization" inherited="true"/>

        <output shape="[B, T, c_out]"/>
    </architecture>

    <hyperparameters inherited="TimesNet">
        <param name="preset_weight" default="0.3" description="Preset period ratio (alpha)"/>
        <param name="preset_periods" default="[60, 300, 900, 3600]" description="Power grid periods"/>
    </hyperparameters>
</model>
```

---

## 4. 核心创新详解

### 4.1 混合周期发现算法

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Period Discovery                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: x [B, T, C], k=5, preset_weight=0.3                      │
│                                                                 │
│  ┌─────────────────┐          ┌─────────────────┐              │
│  │   FFT 发现周期   │          │   预设周期       │              │
│  │   (数据驱动)     │          │   (领域知识)     │              │
│  │                  │          │                  │              │
│  │  ① rfft(x)      │          │  [60, 300, 900]  │              │
│  │  ② topk(amp,k)  │          │                  │              │
│  │  ③ T/freq       │          │                  │              │
│  └────────┬────────┘          └────────┬────────┘              │
│           │                            │                        │
│           │  n_fft = 5*(1-0.3) = 3    │  n_preset = 5*0.3 = 2  │
│           │                            │                        │
│           └────────────┬───────────────┘                        │
│                        ↓                                        │
│              ┌───────────────────┐                              │
│              │   合并去重         │                              │
│              │   3 FFT + 2 Preset │                              │
│              │   = 5 periods      │                              │
│              └───────────────────┘                              │
│                        ↓                                        │
│  输出: period_list [5], period_weight [B, 5]                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 预设周期的物理意义

```
电网周期特性:
═══════════════════════════════════════════════════════════════════

1分钟 (60s):    ████████████████████  短期电压波动、瞬时负荷变化
                ↑
                用户开关操作、小型设备启停

5分钟 (300s):   ████████████████████  中期负荷变化
                ↑
                生产设备运行周期、空调启停

15分钟 (900s):  ████████████████████  负荷平均周期
                ↑
                电力系统需量计算标准周期

1小时 (3600s):  ████████████████████  日负荷模式
                ↑
                典型用电高峰/低谷周期
```

### 4.3 时域平滑卷积

```
原始信号:  ~~~~∧~~~~∨~~~~∧~~~~∨~~~~
                    ↓
          ┌──────────────────┐
          │  Depthwise Conv1D │
          │  kernel=3, pad=1  │
          │  groups=d_model   │
          └──────────────────┘
                    ↓
平滑信号:  ~~~∧~~~∨~~~∧~~~∨~~~

作用: 减少高频噪声，保留主要周期模式
```

---

## 5. 数据流对比 (vs TimesNet)

| 步骤 | TimesNet | VoltageTimesNet | 差异 |
|:----:|----------|-----------------|:----:|
| 周期发现 | 纯 FFT | FFT + Preset 混合 | ★ |
| 周期选择 | Top-K 幅度 | 加权混合 | ★ |
| 时域处理 | 无 | 平滑卷积 | ★ |
| 残差连接 | 直接相加 | 卷积后相加 | - |

---

## 6. 参数配置

### 6.1 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|:------:|------|---------|
| `preset_weight` | 0.3 | 预设周期比例 | 0.2-0.5，数据驱动场景降低 |
| `preset_periods` | [60,300,900,3600] | 预设周期列表 | 根据采样率调整 |
| `seq_len` | 100 | 序列长度 | 需 ≥ 2×max_period 才能利用预设周期 |

### 6.2 序列长度与预设周期的关系

```
seq_len=100 时:
├── 有效预设: 60 (✓)  因为 60 ≤ 100/2 = 50... 实际上 60 > 50 ✗
├── 有效预设: 仅当 seq_len ≥ 120 时，60 才有效
└── 建议: seq_len ≥ 200 以充分利用预设周期

seq_len=500 时:
├── 60  ✓ (60 ≤ 250)
├── 300 ✗ (300 > 250)
└── 有效预设: 仅 [60]

seq_len=1000 时:
├── 60  ✓
├── 300 ✓
├── 900 ✗
└── 有效预设: [60, 300]
```

---

## 7. 性能对比

| 数据集 | TimesNet F1 | VoltageTimesNet F1 | 差异 |
|:------:|:-----------:|:------------------:|:----:|
| PSM | 0.9735 | 0.9731 | -0.04% |
| RuralVoltage | 0.6059 | 0.3123 | -48.5% |

**RuralVoltage 性能分析**:
- seq_len=100 导致大部分预设周期无效
- 建议在 seq_len≥200 时使用 VoltageTimesNet

---

## 8. 使用示例

```bash
# 基础训练
python run.py --is_training 1 \
    --model VoltageTimesNet \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 200 \
    --d_model 64 \
    --preset_weight 0.3

# 调整预设权重（更依赖 FFT）
python run.py --model VoltageTimesNet \
    --preset_weight 0.1 \
    ...
```

---

*文档生成时间: 2026-02-02*
