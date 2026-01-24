# 农网低电压异常检测实验计划

> **注意**: 详细的理论背景和研究方案请参阅 [experiments/README_EXPERIMENT_PLAN.md](experiments/README_EXPERIMENT_PLAN.md)
> 
> 本文档侧重于**如何运行实验**，experiments/ 目录还包含数据生成和结果分析的辅助工具。

## 实验概述

本实验旨在针对农村电网低电压异常检测问题，在 TimesNet 模型基础上进行创新性改进，并通过系统性实验验证改进效果。

## 1. 创新模型

### 1.1 TPATimesNet (Three-Phase Attention TimesNet)

**文件位置**: [models/TPATimesNet.py](models/TPATimesNet.py)

**核心创新**: 三相注意力机制
- 显式建模三相电压 (Va, Vb, Vc) 之间的相位关系
- 引入 `ThreePhaseAttention` 模块捕获跨相位交互
- 使用自适应融合门组合卷积特征和注意力特征

**适用场景**:
- 三相不平衡检测
- 跨相位异常传播分析
- 相位故障定位

**技术细节**:
```
输入特征 → TimesBlock(FFT周期发现+2D卷积) → ThreePhaseAttention → 融合门 → 输出
                                              ↓
                                         Va/Vb/Vc 交互
                                         相位差建模
```

### 1.2 MTSTimesNet (Multi-scale Temporal TimesNet)

**文件位置**: [models/MTSTimesNet.py](models/MTSTimesNet.py)

**核心创新**: 并行多尺度时序分支
- 短期分支: 捕获瞬时波动 (周期 2-20)
- 中期分支: 捕获趋势变化 (周期 20-60)
- 长期分支: 捕获周期模式 (周期 60-200)
- 自适应融合门学习最优尺度组合

**适用场景**:
- 复杂多尺度异常模式识别
- 不同时间粒度的异常检测
- 长期趋势与短期波动联合分析

**技术细节**:
```
输入 ─┬─→ 短期分支 (ScaleSpecificTimesBlock) ─┐
      ├─→ 中期分支 (ScaleSpecificTimesBlock) ─┼─→ CrossScaleConnection → AdaptiveFusion → 输出
      └─→ 长期分支 (ScaleSpecificTimesBlock) ─┘
```

### 1.3 HybridTimesNet (Hybrid Period Discovery TimesNet)

**文件位置**: [models/HybridTimesNet.py](models/HybridTimesNet.py)

**核心创新**: 混合周期发现
- 结合数据驱动的 FFT 周期发现
- 融入领域知识的预设电气周期
- 置信度加权融合机制

**预设周期** (基于电力系统知识):
- 工频周期 (50Hz → 20ms)
- 秒级周期 (1s)
- 分钟级周期 (1min)
- 15分钟结算周期
- 小时级负荷周期

**适用场景**:
- 需要可靠捕获已知电气周期
- 领域知识与数据驱动结合
- 稳健的周期模式识别

## 2. 实验设计

### 2.1 数据集

使用 RuralVoltage 数据集，包含16个特征:
| 特征类别 | 特征名称 |
|---------|---------|
| 电压 | Va, Vb, Vc (三相电压) |
| 电流 | Ia, Ib, Ic (三相电流) |
| 功率 | P (有功), Q (无功), S (视在), PF (功率因数) |
| 电能质量 | THD_Va, THD_Vb, THD_Vc (谐波畸变率) |
| 其他 | Freq (频率), V_unbalance, I_unbalance (不平衡度) |

**异常类型**:
1. **欠压异常** (Undervoltage): V < 198V
2. **过压异常** (Overvoltage): V > 242V
3. **电压骤降** (Voltage Sag): 短期电压跌落
4. **谐波异常** (Harmonic): THD > 5%
5. **三相不平衡** (Unbalance): 不平衡度 > 4%

### 2.2 实验配置

| 参数 | 标准配置 | 深层配置 | 长序列配置 |
|-----|---------|---------|-----------|
| seq_len | 100 | 100 | 200 |
| d_model | 64 | 128 | 64 |
| d_ff | 64 | 128 | 64 |
| e_layers | 2 | 4 | 2 |
| top_k | 5 | 5 | 8 |
| batch_size | 32 | 32 | 16 |
| learning_rate | 1e-4 | 5e-5 | 1e-4 |
| train_epochs | 10 | 15 | 10 |

### 2.3 基线模型

| 模型 | 类型 | 特点 |
|-----|------|-----|
| TimesNet | 时序卷积 | FFT周期发现 + 2D卷积 |
| DLinear | 线性 | 分解 + 线性映射 |
| PatchTST | Transformer | Patch嵌入 + 自注意力 |
| Transformer | Transformer | 标准自注意力 |
| Autoformer | Transformer | 自相关机制 |
| VoltageTimesNet | 时序卷积 | 电压特化TimesNet |

### 2.4 评估指标

- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数
- **Accuracy**: 准确率
- **AUC-ROC**: ROC曲线下面积

## 3. 运行实验

### 3.1 快速开始

```bash
cd code/voltage_anomaly_detection

# 运行所有实验
bash scripts/run_all_experiments.sh

# 仅运行基线
bash scripts/run_all_experiments.sh baseline

# 仅运行创新模型
bash scripts/run_all_experiments.sh innovative

# 运行消融实验
bash scripts/run_all_experiments.sh ablation
```

### 3.2 单独运行

```bash
# TPATimesNet
bash scripts/RuralVoltage/TPATimesNet.sh

# MTSTimesNet
bash scripts/RuralVoltage/MTSTimesNet.sh

# HybridTimesNet
bash scripts/RuralVoltage/HybridTimesNet.sh

# 基线模型
bash scripts/RuralVoltage/run_baselines.sh
```

### 3.3 自定义实验

```bash
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/RuralVoltage \
    --data_path train.csv \
    --model_id custom_experiment \
    --model TPATimesNet \
    --data RuralVoltage \
    --features M \
    --seq_len 100 \
    --enc_in 17 \
    --c_out 17 \
    --d_model 64 \
    --e_layers 2 \
    --batch_size 32 \
    --learning_rate 0.0001
```

## 4. 消融实验

### 4.1 组件贡献验证

| 消融组 | 实验内容 | 目的 |
|-------|---------|-----|
| A1 | TPATimesNet vs TimesNet | 验证三相注意力贡献 |
| A2 | MTSTimesNet vs TimesNet | 验证多尺度分支贡献 |
| A3 | HybridTimesNet vs TimesNet | 验证混合周期发现贡献 |

### 4.2 超参数敏感性

- **top_k**: 周期数量 {3, 5, 8}
- **d_model**: 模型维度 {32, 64, 128}
- **e_layers**: 层数 {1, 2, 3, 4}

## 5. 预期结果

### 5.1 性能提升预期

| 模型 | 相对TimesNet提升 | 主要优势场景 |
|-----|-----------------|-------------|
| TPATimesNet | +2-5% F1 | 三相不平衡检测 |
| MTSTimesNet | +3-6% F1 | 复杂多尺度异常 |
| HybridTimesNet | +2-4% F1 | 周期性异常检测 |

### 5.2 分析维度

1. **整体性能对比**: 所有模型在测试集上的指标对比
2. **异常类型分析**: 各模型在不同异常类型上的表现
3. **消融分析**: 各创新组件的贡献量化
4. **计算效率**: 训练时间、推理时间、参数量

## 6. 文件结构

```
code/voltage_anomaly_detection/
├── models/
│   ├── __init__.py          # 模型注册 (已更新)
│   ├── TimesNet.py          # 原始TimesNet
│   ├── TPATimesNet.py       # 三相注意力模型 [新增]
│   ├── MTSTimesNet.py       # 多尺度时序模型 [新增]
│   └── HybridTimesNet.py    # 混合周期模型 [新增]
├── scripts/
│   ├── run_all_experiments.sh    # 完整实验运行 [新增]
│   └── RuralVoltage/
│       ├── TPATimesNet.sh        # TPATimesNet实验 [新增]
│       ├── MTSTimesNet.sh        # MTSTimesNet实验 [新增]
│       ├── HybridTimesNet.sh     # HybridTimesNet实验 [新增]
│       ├── run_baselines.sh      # 基线实验 [新增]
│       └── run_ablation.sh       # 消融实验 [新增]
├── dataset/RuralVoltage/
│   ├── train.csv
│   ├── test.csv
│   └── test_label.csv
└── test_results/
    └── experiment_summary/       # 实验结果汇总
```

## 7. 注意事项

1. **数据准备**: 确保 `dataset/RuralVoltage/` 目录下有正确格式的数据文件
2. **GPU配置**: 脚本默认使用 GPU 0，可通过修改 `CUDA_VISIBLE_DEVICES` 调整
3. **内存管理**: 长序列配置需要更多显存，可能需要减小 batch_size
4. **结果保存**: 实验结果自动保存在 `test_results/` 和 `checkpoints/` 目录

## 8. 参考文献

1. Wu, H., et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
2. 农村电网电压质量相关国家标准 (GB/T 12325-2008)
3. 时间序列异常检测综述文献
