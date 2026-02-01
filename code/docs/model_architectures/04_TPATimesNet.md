# TPATimesNet 模型架构文档

> **模型名称**: TPATimesNet (Three-Phase Attention TimesNet)
> **基础**: TimesNet (ICLR 2023)
> **创新**: 三相交叉注意力机制
> **参数量**: ~5M

---

## 1. 模型概述

TPATimesNet 是针对 **三相电压异常检测** 优化的 TimesNet 变体。核心创新是引入 **三相交叉注意力机制**，显式建模三相电压 (Va, Vb, Vc) 之间的相关性，提升对三相不平衡等异常的检测能力。

### 核心创新

1. **三相注意力机制**: 专门针对 Va, Vb, Vc 设计的交叉注意力
2. **相位偏置矩阵**: 可学习的相位关系偏置 `phase_bias`
3. **注意力-卷积融合**: 通过 `fusion_gate` 动态融合两种特征
4. **物理约束**: 利用三相电气系统的固有对称性

---

## 2. ASCII 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TPATimesNet Model                                  │
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
│  ║    PhaseAwareTimesBlock × e_layers      ║                               │
│  ║                                          ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │      FFT Period Discovery          │  ║                               │
│  ║  │      (Same as TimesNet)            │  ║                               │
│  ║  └───────────────────────────────────┘  ║                               │
│  ║                   ↓                      ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │  Multi-Period 2D Convolution      │  ║                               │
│  ║  │  (Inception Block)                │  ║                               │
│  ║  └─────────────┬─────────────────────┘  ║                               │
│  ║                │                         ║                               │
│  ║                ↓                         ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │  ★ Three-Phase Attention          │  ║  ← 核心创新                   │
│  ║  │                                    │  ║                               │
│  ║  │  ┌────────────────────────────┐   │  ║                               │
│  ║  │  │   Q = Linear(x)            │   │  ║                               │
│  ║  │  │   K = Linear(x)            │   │  ║                               │
│  ║  │  │   V = Linear(x)            │   │  ║                               │
│  ║  │  └────────────────────────────┘   │  ║                               │
│  ║  │              ↓                     │  ║                               │
│  ║  │  ┌────────────────────────────┐   │  ║                               │
│  ║  │  │   Attention Scores          │   │  ║                               │
│  ║  │  │   scores = Q @ K^T / √d    │   │  ║                               │
│  ║  │  └────────────────────────────┘   │  ║                               │
│  ║  │              ↓                     │  ║                               │
│  ║  │  ┌────────────────────────────┐   │  ║                               │
│  ║  │  │   ★ Phase Bias Addition    │   │  ║                               │
│  ║  │  │   scores[:,:,:3,:3] +=     │   │  ║                               │
│  ║  │  │       phase_bias           │   │  ║                               │
│  ║  │  │                             │   │  ║                               │
│  ║  │  │   ┌─────────────────┐      │   │  ║                               │
│  ║  │  │   │ 1.0  0.5  0.5  │      │   │  ║                               │
│  ║  │  │   │ 0.5  1.0  0.5  │ init │   │  ║                               │
│  ║  │  │   │ 0.5  0.5  1.0  │      │   │  ║                               │
│  ║  │  │   └─────────────────┘      │   │  ║                               │
│  ║  │  └────────────────────────────┘   │  ║                               │
│  ║  │              ↓                     │  ║                               │
│  ║  │  ┌────────────────────────────┐   │  ║                               │
│  ║  │  │   attn_out = softmax @ V   │   │  ║                               │
│  ║  │  └────────────────────────────┘   │  ║                               │
│  ║  └─────────────┬─────────────────────┘  ║                               │
│  ║                │                         ║                               │
│  ║                ↓                         ║                               │
│  ║  ┌───────────────────────────────────┐  ║                               │
│  ║  │  ★ Fusion Gate                    │  ║                               │
│  ║  │                                    │  ║                               │
│  ║  │  gate = σ(Linear([conv; attn]))   │  ║                               │
│  ║  │  output = gate * conv_out +       │  ║                               │
│  ║  │           (1-gate) * attn_out     │  ║                               │
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
<model name="TPATimesNet" type="three_phase_voltage_anomaly_detection" base="TimesNet">
    <metadata>
        <innovation>Three-Phase Cross Attention Mechanism</innovation>
        <target_domain>Three-Phase Power System Monitoring</target_domain>
        <parameters>~5M</parameters>
    </metadata>

    <architecture>
        <input shape="[B, T, C]" description="Three-phase voltage monitoring data">
            <features>
                <feature index="0-2" name="Va,Vb,Vc" description="Three-phase voltage (key features)"/>
                <feature index="3-5" name="Ia,Ib,Ic" description="Three-phase current"/>
                <feature index="6-9" name="P,Q,S,PF" description="Power metrics"/>
                <feature index="10-12" name="THD_Va/Vb/Vc" description="Harmonic distortion"/>
                <feature index="13" name="Freq" description="Grid frequency"/>
                <feature index="14-15" name="V/I_unbalance" description="Unbalance factors"/>
            </features>
        </input>

        <preprocessing name="InstanceNormalization" inherited="true"/>

        <layer name="DataEmbedding" type="embedding" inherited="true"/>

        <block name="PhaseAwareTimesBlock" repeat="e_layers">
            <module name="FFT_for_Period" inherited="true">
                <description>Standard FFT period discovery from TimesNet</description>
            </module>

            <parallel_branches name="PeriodBranches" inherited="true">
                <!-- Same as TimesNet: 1D→2D→Conv2D→2D→1D -->
            </parallel_branches>

            <aggregation name="AdaptiveAggregation" inherited="true"/>

            <module name="ThreePhaseAttention" new="true">
                <description>Cross-attention for three-phase voltage correlations</description>

                <projections>
                    <projection name="query" type="Linear">
                        <input_dim>d_model</input_dim>
                        <output_dim>d_model</output_dim>
                    </projection>
                    <projection name="key" type="Linear">
                        <input_dim>d_model</input_dim>
                        <output_dim>d_model</output_dim>
                    </projection>
                    <projection name="value" type="Linear">
                        <input_dim>d_model</input_dim>
                        <output_dim>d_model</output_dim>
                    </projection>
                </projections>

                <attention_computation>
                    <step>Q = query_proj(x)</step>
                    <step>K = key_proj(x)</step>
                    <step>V = value_proj(x)</step>
                    <step>scores = Q @ K.transpose(-2, -1) / sqrt(d_model)</step>
                    <step>scores[:, :, :3, :3] += phase_bias</step>
                    <step>attn = softmax(scores, dim=-1)</step>
                    <step>attn = dropout(attn)</step>
                    <step>output = attn @ V</step>
                    <step>output = output_proj(output)</step>
                </attention_computation>

                <parameter name="phase_bias" learnable="true">
                    <description>Learnable bias for three-phase relationships</description>
                    <initialization>
                        <matrix rows="3" cols="3">
                            <row>[1.0, 0.5, 0.5]</row>
                            <row>[0.5, 1.0, 0.5]</row>
                            <row>[0.5, 0.5, 1.0]</row>
                        </matrix>
                    </initialization>
                    <physical_meaning>
                        <diagonal>Self-correlation (strongest)</diagonal>
                        <off_diagonal>Cross-phase correlation (weaker)</off_diagonal>
                    </physical_meaning>
                </parameter>

                <output_projection type="Linear">
                    <input_dim>d_model</input_dim>
                    <output_dim>d_model</output_dim>
                </output_projection>
            </module>

            <module name="FusionGate" new="true">
                <description>Dynamic fusion of conv and attention outputs</description>
                <type>Gated Linear Unit</type>
                <formula>gate = sigmoid(Linear(concat(conv_out, attn_out)))</formula>
                <fusion>output = gate * conv_out + (1-gate) * attn_out</fusion>
                <advantage>Adaptive weighting based on input characteristics</advantage>
            </module>

            <residual>output = output + x</residual>
        </block>

        <layer name="LayerNorm" inherited="true"/>
        <layer name="Projection" inherited="true"/>
        <postprocessing name="DeNormalization" inherited="true"/>

        <output shape="[B, T, c_out]"/>
    </architecture>

    <hyperparameters inherited="TimesNet">
        <param name="n_heads" default="1" description="Attention heads (single for simplicity)"/>
        <param name="phase_indices" default="[0,1,2]" description="Indices of Va,Vb,Vc in input"/>
        <param name="dropout" default="0.1" description="Attention dropout rate"/>
    </hyperparameters>
</model>
```

---

## 4. 核心创新详解

### 4.1 三相注意力机制

```
┌─────────────────────────────────────────────────────────────────┐
│                    Three-Phase Attention                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: x [B, T, d_model]                                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Query Q    │    │    Key K     │    │   Value V    │      │
│  │  Linear(x)   │    │  Linear(x)   │    │  Linear(x)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └─────────┬─────────┘                   │               │
│                   ↓                             │               │
│         ┌─────────────────────┐                 │               │
│         │  scores = Q @ K^T   │                 │               │
│         │        / √d         │                 │               │
│         └─────────────────────┘                 │               │
│                   ↓                             │               │
│         ┌─────────────────────┐                 │               │
│         │  ★ Add Phase Bias   │                 │               │
│         │                      │                 │               │
│         │  scores[Va,Va] += 1.0                 │               │
│         │  scores[Va,Vb] += 0.5                 │               │
│         │  scores[Va,Vc] += 0.5                 │               │
│         │  scores[Vb,Va] += 0.5                 │               │
│         │  scores[Vb,Vb] += 1.0                 │               │
│         │  scores[Vb,Vc] += 0.5                 │               │
│         │  scores[Vc,Va] += 0.5                 │               │
│         │  scores[Vc,Vb] += 0.5                 │               │
│         │  scores[Vc,Vc] += 1.0                 │               │
│         └─────────────────────┘                 │               │
│                   ↓                             │               │
│         ┌─────────────────────┐                 │               │
│         │  attn = softmax(scores)               │               │
│         └─────────────────────┘                 │               │
│                   │                             │               │
│                   └──────────────┬──────────────┘               │
│                                  ↓                               │
│                        ┌─────────────────┐                       │
│                        │ output = attn @ V │                     │
│                        └─────────────────┘                       │
│                                  ↓                               │
│  输出: [B, T, d_model]                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 相位偏置矩阵的物理意义

```
三相电压系统特性:
═══════════════════════════════════════════════════════════════════

理想三相系统 (120° 相位差):

     Va ────────∧────────
                 \
     Vb ──────────∧──────
                   \
     Vc ────────────∧────

相位偏置矩阵:
┌─────────────────────────────────────┐
│         Va      Vb      Vc          │
│   Va  [ 1.0    0.5     0.5  ]       │  ← 自相关最强
│   Vb  [ 0.5    1.0     0.5  ]       │  ← 互相关较弱
│   Vc  [ 0.5    0.5     1.0  ]       │  ← 对称结构
└─────────────────────────────────────┘

物理含义:
- 对角线 (1.0): 每相与自身的关联最强
- 非对角线 (0.5): 相间关联较弱但存在
- 对称性: 反映三相系统的物理对称性
- 可学习: 训练过程中适应实际数据分布
```

### 4.3 融合门机制

```
Conv Output ─────────┐
                     ├───► Concat ───► Linear ───► Sigmoid ───► gate
Attention Output ────┘
                                                        │
                                                        ↓
Final Output = gate × Conv_out + (1-gate) × Attn_out

作用:
- 动态平衡卷积和注意力的贡献
- 根据输入特性自适应调整权重
- 当三相关联重要时，注意力权重增加
- 当周期模式重要时，卷积权重增加
```

---

## 5. 数据流对比 (vs TimesNet)

| 步骤 | TimesNet | TPATimesNet | 差异 |
|:----:|----------|-------------|:----:|
| 周期发现 | FFT | FFT (相同) | - |
| 2D 卷积 | Inception | Inception (相同) | - |
| 特征增强 | 无 | 三相注意力 | ★ |
| 输出融合 | 加权求和 | 融合门 | ★ |
| 物理约束 | 无 | 相位偏置矩阵 | ★ |

---

## 6. 参数配置

### 6.1 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|:------:|------|---------:|
| `phase_indices` | [0,1,2] | 三相电压列索引 | 根据数据格式调整 |
| `n_heads` | 1 | 注意力头数 | 三相场景通常 1 即可 |
| `dropout` | 0.1 | 注意力 Dropout | 过拟合时增加 |

### 6.2 相位偏置初始化策略

```python
# 默认初始化 (对称)
phase_bias = [
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.5],
    [0.5, 0.5, 1.0]
]

# 可选: 随机初始化 + 正则化
phase_bias = torch.randn(3, 3) * 0.1
# 正则化损失: 鼓励对角线 > 非对角线
```

---

## 7. 适用场景

| 场景 | 适用性 | 说明 |
|------|:------:|------|
| 三相不平衡检测 | ★★★★★ | 核心优势 |
| 单相故障检测 | ★★★★☆ | 可捕获相间差异 |
| 谐波异常 | ★★★☆☆ | 需配合 THD 特征 |
| 电压骤降/骤升 | ★★★★☆ | 可检测不对称事件 |

---

## 8. 使用示例

```bash
# 基础训练
python run.py --is_training 1 \
    --model TPATimesNet \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 100 \
    --d_model 64 \
    --e_layers 2

# 调整三相索引 (如果数据格式不同)
# 需要修改代码中的 phase_indices
```

---

*文档生成时间: 2026-02-02*
