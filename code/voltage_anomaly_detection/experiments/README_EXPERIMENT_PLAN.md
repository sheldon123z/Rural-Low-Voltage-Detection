# 农村低压配电网电压异常检测实验计划

> **重要说明**: 
> - 本文档提供详细的理论背景和研究方案
> - **模型实现**: 请查看主目录 `../models/` (TPATimesNet.py, MTSTimesNet.py, HybridTimesNet.py)
> - **实验脚本**: 请查看 `../scripts/RuralVoltage/`
> - **快速开始**: 参阅主目录的 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md)
> 
> 本目录 (`experiments/`) 包含辅助工具：
> - `data/` - 数据生成和特征工程工具
> - `analysis/` - 结果分析工具
> - `run_experiment.py` - 独立实验运行器（可选）

---

## 1. 研究背景与目标

### 1.1 研究问题
农村低压配电网存在以下典型异常：
- **欠压异常**: 电压低于 198V（-10%），影响用电设备正常运行
- **过压异常**: 电压高于 242V（+10%），损坏用电设备
- **电压骤降**: 短时电压跌落，影响敏感负荷
- **谐波畸变**: THD > 5%，电能质量问题
- **三相不平衡**: 不平衡度 > 4%，影响系统效率

### 1.2 研究目标
1. 构建针对农村电网特点的高质量数据集
2. 基于 TimesNet 提出创新性的电压异常检测模型
3. 通过对比实验验证模型在农网场景下的优越性

---

## 2. 数据处理实验方案

### 2.1 数据集设计 (RuralVoltageV2)

**17维特征设计**（符合 GB/T 12325-2008）:

| 类别 | 特征 | 单位 | 正常范围 | 异常阈值 |
|------|------|------|----------|----------|
| 三相电压 | Va, Vb, Vc | V | 198-242 | <198 或 >242 |
| 三相电流 | Ia, Ib, Ic | A | 0-50 | >额定150% |
| 有功功率 | P | kW | - | - |
| 无功功率 | Q | kVar | - | - |
| 视在功率 | S | kVA | - | - |
| 功率因数 | PF | - | 0.85-1.0 | <0.85 |
| 谐波失真 | THD_Va, THD_Vb, THD_Vc | % | 0-5 | >5 |
| 电压不平衡 | V_unbalance | % | 0-4 | >4 |
| 电流不平衡 | I_unbalance | % | 0-10 | >10 |
| 频率 | Freq | Hz | 49.5-50.5 | 偏差>0.5 |

### 2.2 数据增强策略

```
实验编号: DATA-001 至 DATA-005
```

| 实验ID | 方法 | 说明 | 预期效果 |
|--------|------|------|----------|
| DATA-001 | 时间窗口滑动 | step=1/5/10 对比 | 增加样本多样性 |
| DATA-002 | 高斯噪声注入 | σ=0.01/0.05/0.1 | 提高鲁棒性 |
| DATA-003 | 时间拉伸/压缩 | 0.8x-1.2x | 模拟采样率变化 |
| DATA-004 | 异常混合注入 | 多类型叠加 | 复杂场景模拟 |
| DATA-005 | 三相相位扰动 | ±5°相位偏移 | 模拟实际测量误差 |

### 2.3 特征工程实验

```
实验编号: FEAT-001 至 FEAT-004
```

| 实验ID | 方法 | 新增特征 | 目的 |
|--------|------|----------|------|
| FEAT-001 | 统计特征 | 滑动窗口均值/方差/峰度 | 捕获统计分布变化 |
| FEAT-002 | 频域特征 | FFT主频率/能量分布 | 识别谐波异常 |
| FEAT-003 | 相间特征 | Va-Vb, Vb-Vc, Vc-Va差值 | 检测不平衡 |
| FEAT-004 | 对称分量 | 正序/负序/零序分量 | 故障类型诊断 |

---

## 3. 模型创新实验方案

### 3.1 创新方向概述

基于 TimesNet 的核心思想（1D→2D变换 + Inception卷积），提出以下创新模型：

| 模型名称 | 核心创新点 | 适用场景 |
|----------|------------|----------|
| **TPATimesNet** | 三相注意力机制 | 三相不平衡检测 |
| **MTSTimesNet** | 多尺度时间块 | 多周期模式捕获 |
| **HybridTimesNet** | 混合周期发现 | 已知+未知周期融合 |
| **LightVoltageNet** | 轻量化设计 | 边缘部署场景 |

### 3.2 模型1: TPATimesNet (Three-Phase Attention TimesNet)

**创新点**: 引入三相注意力机制，显式建模三相电压间的相关性

**架构设计**:
```
输入 [B,T,17] 
    → 特征分组(三相电压/电流/功率/质量指标)
    → ThreePhaseAttention(Va,Vb,Vc) 
    → TimesBlock×N
    → 投影层
    → 输出 [B,T,17]
```

**关键组件**:
1. **三相交叉注意力**: Q=Va, K/V=[Vb,Vc]
2. **相位感知位置编码**: 考虑120°相位差
3. **不平衡度感知损失**: 增加对不平衡异常的敏感性

### 3.3 模型2: MTSTimesNet (Multi-scale Temporal TimesNet)

**创新点**: 多尺度时间块并行处理，同时捕获短期波动和长期趋势

**架构设计**:
```
输入 [B,T,C]
    → Embedding
    → 并行分支:
        Branch1: TimesBlock(short_period: 5-20)
        Branch2: TimesBlock(medium_period: 20-60) 
        Branch3: TimesBlock(long_period: 60-200)
    → 特征融合层
    → 输出
```

**关键组件**:
1. **分层周期发现**: 不同分支专注不同时间尺度
2. **自适应融合门**: 根据输入动态调整分支权重
3. **跨尺度残差连接**: 增强信息流动

### 3.4 模型3: HybridTimesNet (Hybrid Period Discovery TimesNet)

**创新点**: 融合FFT自动发现与电网先验周期，提高周期发现准确性

**预设电网周期**:
- 工频周期: 20ms (50Hz)
- 分钟级波动: 60s
- 负荷5分钟周期: 300s  
- 15分钟调度周期: 900s
- 小时级负荷模式: 3600s

**周期融合策略**:
```python
final_periods = α × FFT_periods + (1-α) × preset_periods
# α通过可学习参数自适应调整
```

### 3.5 模型4: LightVoltageNet (轻量化电压异常检测网络)

**创新点**: 面向边缘计算设备的轻量化设计

**优化策略**:
1. **深度可分离卷积**: 替代标准2D卷积
2. **知识蒸馏**: 从大模型蒸馏到小模型
3. **动态推理**: 简单样本提前退出
4. **量化感知训练**: INT8量化部署

---

## 4. 训练与评估实验方案

### 4.1 基线模型对比实验

```
实验编号: BASELINE-001 至 BASELINE-010
```

| 实验ID | 模型 | 类型 | 参数量 | 备注 |
|--------|------|------|--------|------|
| BASELINE-001 | TimesNet | 深度学习 | ~500K | 原始模型 |
| BASELINE-002 | DLinear | 深度学习 | ~50K | 轻量级基线 |
| BASELINE-003 | Transformer | 深度学习 | ~300K | 注意力基线 |
| BASELINE-004 | Informer | 深度学习 | ~400K | 稀疏注意力 |
| BASELINE-005 | PatchTST | 深度学习 | ~350K | Patch方法 |
| BASELINE-006 | Autoformer | 深度学习 | ~400K | 分解+注意力 |
| BASELINE-007 | iTransformer | 深度学习 | ~350K | 倒置Transformer |
| BASELINE-008 | LSTM-AE | 深度学习 | ~200K | 传统RNN |
| BASELINE-009 | Isolation Forest | 机器学习 | - | 无监督基线 |
| BASELINE-010 | One-Class SVM | 机器学习 | - | 无监督基线 |

### 4.2 消融实验设计

```
实验编号: ABLATION-001 至 ABLATION-012
```

**TPATimesNet消融**:
| 实验ID | 变体 | 移除组件 |
|--------|------|----------|
| ABLATION-001 | w/o ThreePhaseAttention | 移除三相注意力 |
| ABLATION-002 | w/o PhaseEncoding | 移除相位编码 |
| ABLATION-003 | w/o UnbalanceLoss | 移除不平衡损失 |

**MTSTimesNet消融**:
| 实验ID | 变体 | 移除组件 |
|--------|------|----------|
| ABLATION-004 | w/o ShortBranch | 仅中长期分支 |
| ABLATION-005 | w/o LongBranch | 仅短中期分支 |
| ABLATION-006 | w/o AdaptiveFusion | 固定权重融合 |

**HybridTimesNet消融**:
| 实验ID | 变体 | 配置 |
|--------|------|------|
| ABLATION-007 | FFT-only | 仅FFT周期发现 |
| ABLATION-008 | Preset-only | 仅预设周期 |
| ABLATION-009 | α=0.3/0.5/0.7 | 不同融合权重 |

**通用消融**:
| 实验ID | 变体 | 配置 |
|--------|------|------|
| ABLATION-010 | 不同seq_len | 50/100/200/500 |
| ABLATION-011 | 不同d_model | 32/64/128/256 |
| ABLATION-012 | 不同e_layers | 1/2/3/4 |

### 4.3 超参数搜索空间

```python
hyperparameter_space = {
    'seq_len': [50, 100, 200, 300],
    'd_model': [32, 64, 128],
    'd_ff': [64, 128, 256],
    'e_layers': [1, 2, 3],
    'top_k': [3, 5, 7],
    'num_kernels': [4, 6, 8],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'anomaly_ratio': [0.5, 1.0, 2.0, 5.0],
}
```

### 4.4 评估指标

**主要指标**:
| 指标 | 公式 | 说明 |
|------|------|------|
| Precision | TP/(TP+FP) | 检测准确性 |
| Recall | TP/(TP+FN) | 检测完整性 |
| F1-Score | 2×P×R/(P+R) | 综合指标 |
| AUC-ROC | ROC曲线下面积 | 分类能力 |
| AUC-PR | PR曲线下面积 | 不平衡数据评估 |

**农网专用指标**:
| 指标 | 说明 |
|------|------|
| 欠压检出率 | 欠压异常的召回率 |
| 过压检出率 | 过压异常的召回率 |
| 不平衡检出率 | 三相不平衡的召回率 |
| 平均检测延迟 | 异常发生到检测的时间差 |
| 误报率 | 正常数据被判为异常的比例 |

---

## 5. 实验执行计划

### 5.1 第一阶段: 数据准备 (1-2周)

**任务清单**:
- [ ] 实现增强版数据生成器 (generate_sample_data_v2.py)
- [ ] 实现特征工程模块 (feature_engineering.py)
- [ ] 生成训练/验证/测试数据集
- [ ] 数据质量验证与可视化

**输出物**:
- `dataset/RuralVoltageV2/` 目录
- 数据描述统计报告

### 5.2 第二阶段: 模型实现 (2-3周)

**任务清单**:
- [ ] 实现 TPATimesNet (models/TPATimesNet.py)
- [ ] 实现 MTSTimesNet (models/MTSTimesNet.py)  
- [ ] 实现 HybridTimesNet (models/HybridTimesNet.py)
- [ ] 实现 LightVoltageNet (models/LightVoltageNet.py)
- [ ] 单元测试验证

**输出物**:
- 4个新模型代码文件
- 模型架构图

### 5.3 第三阶段: 基线实验 (1周)

**任务清单**:
- [ ] 运行所有基线模型 (BASELINE-001 ~ BASELINE-010)
- [ ] 记录实验结果
- [ ] 生成对比表格

### 5.4 第四阶段: 创新模型实验 (2周)

**任务清单**:
- [ ] 运行创新模型完整实验
- [ ] 消融实验 (ABLATION-001 ~ ABLATION-012)
- [ ] 超参数调优
- [ ] 结果分析

### 5.5 第五阶段: 结果整理 (1周)

**任务清单**:
- [ ] 生成实验结果表格
- [ ] 绘制对比图表
- [ ] 撰写实验分析报告
- [ ] 更新论文实验章节

---

## 6. 目录结构设计

```
experiments/
├── README_EXPERIMENT_PLAN.md      # 本文档
├── configs/                        # 实验配置文件
│   ├── baseline_configs.yaml
│   ├── tpatimesnet_configs.yaml
│   ├── mtstimesnet_configs.yaml
│   └── ...
├── data/                           # 数据处理模块
│   ├── generate_sample_data_v2.py
│   ├── feature_engineering.py
│   └── data_augmentation.py
├── models/                         # 创新模型
│   ├── TPATimesNet.py
│   ├── MTSTimesNet.py
│   ├── HybridTimesNet.py
│   └── LightVoltageNet.py
├── scripts/                        # 实验脚本
│   ├── run_baseline.sh
│   ├── run_tpatimesnet.sh
│   ├── run_ablation.sh
│   └── run_all_experiments.sh
├── analysis/                       # 结果分析
│   ├── visualize_results.py
│   ├── statistical_tests.py
│   └── generate_tables.py
└── results/                        # 实验结果
    ├── baseline/
    ├── tpatimesnet/
    ├── mtstimesnet/
    └── ...
```

---

## 7. 快速开始命令

```bash
# 1. 生成数据集
python experiments/data/generate_sample_data_v2.py \
    --train_samples 50000 --test_samples 10000 --anomaly_ratio 0.1

# 2. 运行基线实验
bash experiments/scripts/run_baseline.sh

# 3. 运行TPATimesNet实验
python run.py --is_training 1 --model TPATimesNet --data RuralVoltage \
    --root_path ./dataset/RuralVoltageV2/ --enc_in 17 --c_out 17 \
    --seq_len 100 --d_model 64 --e_layers 2 --top_k 5 \
    --train_epochs 20 --batch_size 32 --learning_rate 0.0001

# 4. 运行消融实验
bash experiments/scripts/run_ablation.sh

# 5. 生成结果报告
python experiments/analysis/generate_tables.py
```

---

## 8. 预期创新贡献

1. **TPATimesNet**: 首次在时序异常检测中引入三相电气特性建模
2. **MTSTimesNet**: 多尺度并行架构适应农网复杂时间模式
3. **HybridTimesNet**: 领域知识与数据驱动的周期发现融合方法
4. **RuralVoltageV2数据集**: 高质量农网电压异常检测基准数据集
5. **农网专用评估指标**: 面向实际应用的细粒度异常类型评估

---

*文档版本: v1.0*  
*最后更新: 2026-01-24*
