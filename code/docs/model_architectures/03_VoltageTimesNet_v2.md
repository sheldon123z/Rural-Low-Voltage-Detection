# VoltageTimesNet_v2 模型架构文档

> **模型名称**: VoltageTimesNet_v2
> **基础**: VoltageTimesNet
> **核心创新**: 可学习权重 + 召回率优化
> **参数量**: ~9.5M
> **目标**: 召回率 >70% (已达成: 71.13%)

---

## 1. 模型概述

VoltageTimesNet_v2 是针对 **高召回率场景** 优化的电压异常检测模型。通过引入可学习的预设权重、异常敏感度增强模块和多尺度时域卷积，在保持合理精确率的同时显著提升召回率。

### 五大核心改进

| # | 改进点 | 原版 | v2 版 | 效果 |
|:-:|--------|------|-------|------|
| 1 | 预设权重 | 固定 0.3 | **可学习参数** | 自适应调整 |
| 2 | 异常检测 | 单一重构误差 | **敏感度增强模块** | 召回率↑ |
| 3 | 时域建模 | 单尺度卷积 | **多尺度 (3/5/7)** | 多粒度特征 |
| 4 | 特征编码 | 通用嵌入 | **电能质量编码器** | 领域特征 |
| 5 | 相位约束 | 无 | **三相约束模块** | 相位关联 |

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       VoltageTimesNet_v2 Model                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, T, C=16]                                                        │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────┐                               │
│  │          Instance Normalization          │                               │
│  └─────────────────────────────────────────┘                               │
│         ↓                                                                   │
│         ├────────────────┬────────────────┐                                │
│         ↓                ↓                ↓                                │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                         │
│  │   Data     │   │  ★ Power   │   │  ★ Phase   │                         │
│  │ Embedding  │   │  Quality   │   │ Constraint │                         │
│  │            │   │  Encoder   │   │  Module    │                         │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                         │
│        │                │                │                                 │
│        └────────────────┴────────────────┘                                │
│                         ↓                                                  │
│               ┌──────────────────┐                                         │
│               │  Feature Fusion   │   Concat + Linear                      │
│               │  [2*d_model] → d  │                                        │
│               └────────┬─────────┘                                         │
│                        ↓                                                   │
│               ┌──────────────────┐                                         │
│               │ ★ Anomaly       │   放大异常特征，提升召回率              │
│               │   Sensitivity    │                                        │
│               │   Amplifier      │                                        │
│               └────────┬─────────┘                                         │
│                        ↓                                                   │
│  ╔═════════════════════════════════════════════════════════════════════╗  │
│  ║              VoltageTimesBlock_v2 × e_layers                        ║  │
│  ║                                                                      ║  │
│  ║  ┌─────────────────────────────────────────────────────────────┐   ║  │
│  ║  │         Hybrid Period Discovery v2                          │   ║  │
│  ║  │                                                              │   ║  │
│  ║  │    FFT Periods ──┬── ★ Learnable preset_weight ──┬── Preset │   ║  │
│  ║  │                  │    sigmoid(θ) ∈ [0,1]          │          │   ║  │
│  ║  │                  └──────────────┬─────────────────┘          │   ║  │
│  ║  │                                 ↓                            │   ║  │
│  ║  │                         Mixed Periods                        │   ║  │
│  ║  └─────────────────────────────────────────────────────────────┘   ║  │
│  ║                                 ↓                                   ║  │
│  ║  ┌─────────────────────────────────────────────────────────────┐   ║  │
│  ║  │            Multi-Period 2D Convolution                      │   ║  │
│  ║  │            (Inception Block V1)                             │   ║  │
│  ║  └─────────────────────────────────────────────────────────────┘   ║  │
│  ║                                 ↓                                   ║  │
│  ║  ┌─────────────────────────────────────────────────────────────┐   ║  │
│  ║  │         ★ Multi-Scale Temporal Convolution                  │   ║  │
│  ║  │                                                              │   ║  │
│  ║  │    ┌─────────┐  ┌─────────┐  ┌─────────┐                   │   ║  │
│  ║  │    │ Conv1D  │  │ Conv1D  │  │ Conv1D  │                   │   ║  │
│  ║  │    │ k=3     │  │ k=5     │  │ k=7     │                   │   ║  │
│  ║  │    └────┬────┘  └────┬────┘  └────┬────┘                   │   ║  │
│  ║  │         └────────────┼────────────┘                         │   ║  │
│  ║  │                      ↓                                      │   ║  │
│  ║  │                 Concatenate                                 │   ║  │
│  ║  └─────────────────────────────────────────────────────────────┘   ║  │
│  ║                                 ↓                                   ║  │
│  ║                       Residual Connection                           ║  │
│  ╚═════════════════════════════════════════════════════════════════════╝  │
│                        ↓                                                  │
│               ┌──────────────────┐                                        │
│               │     Projection    │                                        │
│               └────────┬─────────┘                                        │
│                        ↓                                                  │
│  Output: [B, T, c_out]                                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. XML 结构表示

```xml
<?xml version="1.0" encoding="UTF-8"?>
<model name="VoltageTimesNet_v2" type="recall_optimized_anomaly_detection" base="VoltageTimesNet">
    <metadata>
        <innovation>Learnable Preset Weight + Recall Optimization</innovation>
        <target>Recall >70% while maintaining reasonable Precision</target>
        <parameters>~9.5M</parameters>
        <achievement>Recall: 71.13% (target achieved)</achievement>
    </metadata>

    <architecture>
        <input shape="[B, T, C]"/>

        <preprocessing name="InstanceNormalization"/>

        <!-- New Module: Power Quality Encoder -->
        <module name="PowerQualityEncoder" new="true">
            <description>Domain-specific encoder for power quality features</description>
            <submodule name="VoltageEncoder">
                <input_indices>[0, 1, 2]</input_indices>
                <comment>Va, Vb, Vc</comment>
                <output_dim>d_model/2</output_dim>
            </submodule>
            <submodule name="THDEncoder">
                <input_indices>[10, 11, 12]</input_indices>
                <comment>THD_Va, THD_Vb, THD_Vc</comment>
                <output_dim>d_model/4</output_dim>
            </submodule>
            <submodule name="UnbalanceEncoder">
                <input_indices>[14, 15]</input_indices>
                <comment>V_unbalance, I_unbalance</comment>
                <output_dim>d_model/4</output_dim>
            </submodule>
            <fusion>Linear(d_model, d_model) + LayerNorm</fusion>
        </module>

        <!-- New Module: Anomaly Sensitivity Amplifier -->
        <module name="AnomalySensitivityAmplifier" new="true">
            <description>Amplify anomaly signals for better recall</description>
            <attention>
                <layer>Linear(d_model, d_model*2) → ReLU → Linear(d_model*2, d_model) → Sigmoid</layer>
                <output>attention_weights ∈ [0, 1]</output>
            </attention>
            <amplification>
                <learnable_scale>nn.Parameter(amplify_factor=2.0)</learnable_scale>
                <formula>amplified = x * (1 + attn_weights * scale)</formula>
            </amplification>
        </module>

        <!-- New Module: Phase Constraint -->
        <module name="PhaseConstraintModule" new="true" conditional="enc_in >= 3">
            <description>Learn three-phase voltage correlations</description>
            <learnable_correlation_matrix shape="[3, 3]">
                <initialization>eye(3) + 0.5 * ones(3,3)</initialization>
            </learnable_correlation_matrix>
            <regularization>
                <loss>diag > off_diag (encourage self-correlation)</loss>
            </regularization>
        </module>

        <layer name="FeatureFusion">
            <input>concat(enc_embedding, pq_features + phase_features)</input>
            <operation>Linear(d_model*2, d_model)</operation>
        </layer>

        <block name="VoltageTimesBlock_v2" repeat="e_layers">
            <!-- Learnable preset weight -->
            <parameter name="preset_weight" type="nn.Parameter">
                <initialization>inverse_sigmoid(0.3) ≈ -0.847</initialization>
                <usage>sigmoid(preset_weight) → [0, 1]</usage>
            </parameter>

            <module name="FFT_for_Period_Voltage_v2">
                <algorithm>Same as v1 but with learnable preset_weight</algorithm>
            </module>

            <module name="MultiPeriod2DConv" inherited="true"/>

            <!-- New: Multi-scale temporal convolution -->
            <module name="MultiScaleTemporalConv" new="true">
                <description>Capture patterns at different time scales</description>
                <branch kernel="3" description="Short-term fluctuations"/>
                <branch kernel="5" description="Medium-term patterns"/>
                <branch kernel="7" description="Longer-term trends"/>
                <fusion>Concatenate → Project(if needed) → LayerNorm + Residual</fusion>
            </module>

            <residual>output = output + x</residual>
        </block>

        <layer name="Projection" inherited="true"/>
        <postprocessing name="DeNormalization" inherited="true"/>

        <output shape="[B, T, c_out]"/>
    </architecture>

    <hyperparameters>
        <param name="preset_weight" default="0.3" learnable="true"/>
        <param name="anomaly_amplify_factor" default="2.0" learnable="true"/>
        <param name="anomaly_ratio" default="3.0" description="Threshold percentile: 97%"/>
    </hyperparameters>

    <auxiliary_loss>
        <loss name="phase_constraint_loss" weight="0.01">
            <description>Regularize phase correlation matrix</description>
        </loss>
    </auxiliary_loss>
</model>
```

---

## 4. 核心组件详解

### 4.1 可学习预设权重

```
┌─────────────────────────────────────────────────────────────────┐
│             Learnable Preset Weight                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  初始化:                                                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  init_weight = 0.3                                    │      │
│  │  init_logit = log(0.3 / (1 - 0.3)) ≈ -0.847          │      │
│  │  preset_weight = nn.Parameter(tensor(init_logit))     │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  前向传播:                                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  effective_weight = sigmoid(preset_weight)            │      │
│  │  → 约束在 [0, 1] 范围内                               │      │
│  │  → 可通过梯度下降优化                                 │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  vs 固定权重:                                                   │
│  ┌────────────────────┐    ┌────────────────────┐             │
│  │ v1: 固定 0.3       │    │ v2: 可学习         │             │
│  │ n_preset = k * 0.3 │    │ n_preset = k * σ(θ)│             │
│  │ 无法适应数据      │    │ 数据驱动优化       │             │
│  └────────────────────┘    └────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 异常敏感度增强模块

```
                    输入特征 x
                        │
                        ↓
         ┌──────────────────────────┐
         │    Anomaly Attention     │
         │                          │
         │   Linear(d, 2d) → ReLU   │
         │         ↓                │
         │   Linear(2d, d) → Sigmoid│
         │         ↓                │
         │   attn_weights ∈ [0,1]   │
         └────────────┬─────────────┘
                      │
                      ↓
    ┌────────────────────────────────────┐
    │         Amplification              │
    │                                    │
    │   amplified = x * (1 + w * scale)  │
    │                                    │
    │   where:                           │
    │   - w = attn_weights               │
    │   - scale = learnable (init=2.0)   │
    │                                    │
    │   效果:                            │
    │   - 高注意力 → 特征放大           │
    │   - 低注意力 → 特征保持           │
    │   - 异常特征更容易被检测          │
    └────────────────────────────────────┘
                      │
                      ↓
                 输出特征
```

### 4.3 多尺度时域卷积

```
输入: [B, T, d_model]
            │
            ├───────────────┬───────────────┐
            ↓               ↓               ↓
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Conv1D      │ │   Conv1D      │ │   Conv1D      │
    │   kernel=3    │ │   kernel=5    │ │   kernel=7    │
    │   padding=1   │ │   padding=2   │ │   padding=3   │
    │   out=d/3     │ │   out=d/3     │ │   out=d/3     │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            │   短期波动      │   中期模式      │   长期趋势
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ↓
                      ┌───────────────┐
                      │  Concatenate  │
                      │  [B, T, d]    │
                      └───────┬───────┘
                              ↓
                      ┌───────────────┐
                      │   LayerNorm   │
                      │  + Residual   │
                      └───────────────┘
```

### 4.4 电能质量编码器

```
输入特征 x: [B, T, 16]
     │
     ├── x[:,:,0:3]  (Va, Vb, Vc)
     │        ↓
     │   ┌─────────────────┐
     │   │ VoltageEncoder  │  Linear(3, d/2)
     │   └────────┬────────┘
     │            │
     ├── x[:,:,10:13] (THD_Va, THD_Vb, THD_Vc)
     │        ↓
     │   ┌─────────────────┐
     │   │   THDEncoder    │  Linear(3, d/4)
     │   └────────┬────────┘
     │            │
     └── x[:,:,14:16] (V_unbal, I_unbal)
              ↓
         ┌─────────────────┐
         │ UnbalanceEncoder│  Linear(2, d/4)
         └────────┬────────┘
                  │
                  └────────────┬────────────────┘
                               ↓
                    ┌──────────────────┐
                    │     Concat       │
                    │ [d/2 + d/4 + d/4]│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Fusion Linear   │  [d, d]
                    │  + LayerNorm     │
                    └──────────────────┘
```

---

## 5. 阈值敏感性分析

### 5.1 anomaly_ratio 参数

```
anomaly_ratio 控制异常检测阈值的百分位数:
threshold = np.percentile(train_errors, 100 - anomaly_ratio)

┌──────────────────────────────────────────────────────────────────┐
│  anomaly_ratio  │  阈值百分位  │  效果                           │
├──────────────────────────────────────────────────────────────────┤
│      1.0        │    99%       │  高阈值 → 高Precision, 低Recall │
│      2.0        │    98%       │                                 │
│      3.0        │    97%       │  ★ 推荐：平衡点                 │
│      5.0        │    95%       │  低阈值 → 低Precision, 高Recall │
│     10.0        │    90%       │                                 │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 实验结果

| anomaly_ratio | Precision | Recall | F1 | 推荐场景 |
|:-------------:|:---------:|:------:|:--:|---------|
| 1.0 | **0.9939** | 0.6197 | 0.7634 | 误报敏感 |
| 2.0 | 0.7043 | 0.6804 | 0.6921 | 平衡 |
| **3.0** | 0.5463 | **0.7113** | **0.6178** | **召回优先** |
| 5.0 | 0.3584 | 0.7749 | 0.4901 | 漏检敏感 |

---

## 6. 使用示例

### 6.1 高召回率模式

```bash
python run.py --is_training 1 \
    --model VoltageTimesNet_v2 \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 100 \
    --d_model 64 \
    --anomaly_ratio 3.0    # 97% 百分位数阈值
```

### 6.2 高精确率模式

```bash
python run.py --is_training 1 \
    --model VoltageTimesNet_v2 \
    --anomaly_ratio 1.0    # 99% 百分位数阈值
    ...
```

---

## 7. 性能对比

| 指标 | TimesNet | VoltageTimesNet | VoltageTimesNet_v2 |
|:----:|:--------:|:---------------:|:------------------:|
| Recall | 0.4582 | 0.1906 | **0.7113** |
| Precision | 0.8939 | 0.8636 | 0.5463 |
| F1 | 0.6059 | 0.3123 | 0.6178 |
| 参数量 | 4.7M | 5M | 9.5M |

**关键成就**: 召回率从 45.82% 提升至 **71.13%**，提升 **+55%**

---

*文档生成时间: 2026-02-02*
