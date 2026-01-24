# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

农村低压配电网电压异常检测研究生毕业论文项目。核心代码位于 `code/voltage_anomaly_detection/`，基于清华大学 Time-Series-Library 框架构建，支持 15 种深度学习模型进行时序异常检测。

## 常用命令

### 环境配置
```bash
conda create -n tslib python=3.11
conda activate tslib
pip install -r code/voltage_anomaly_detection/requirements.txt
```

### 训练模型
```bash
cd code/voltage_anomaly_detection

# 使用 TimesNet 在 PSM 数据集训练
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 \
  --train_epochs 10 --batch_size 32

# 使用定制 VoltageTimesNet 在农村电压数据集训练
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ --enc_in 17 --c_out 17 \
  --seq_len 100 --d_model 64 --d_ff 128 --top_k 5

# 仅测试（加载已有检查点）
python run.py --is_training 0 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/
```

### 生成样本数据
```bash
cd code/voltage_anomaly_detection/dataset/RuralVoltage
python generate_sample_data.py --train_samples 10000 --test_samples 2000 --anomaly_ratio 0.1
```

### 使用训练脚本
```bash
cd code/voltage_anomaly_detection
bash scripts/train_psm.sh                    # PSM 数据集
bash scripts/RuralVoltage/VoltageTimesNet.sh # 农村电压数据集
```

## 代码架构

### 核心入口
- `run.py` - 主入口脚本，解析命令行参数并执行训练/测试流程

### 模块结构
```
code/voltage_anomaly_detection/
├── exp/exp_anomaly_detection.py    # 实验类：train() 和 test() 方法
├── models/                         # 15 个模型实现
│   ├── TimesNet.py                 # 核心模型：FFT + 2D卷积周期建模
│   ├── VoltageTimesNet.py          # 定制模型：预设电网周期 + TimesNet
│   └── DLinear.py                  # 轻量级：序列分解 + 线性层
├── data_provider/
│   ├── data_factory.py             # 数据工厂：data_provider(args, flag)
│   └── data_loader.py              # 6 个数据集加载器
├── layers/                         # 网络层组件
│   ├── Embed.py                    # TokenEmbedding, PositionalEmbedding
│   ├── SelfAttention_Family.py     # FullAttention, ProbAttention
│   └── Conv_Blocks.py              # Inception 2D 卷积块
└── utils/
    ├── tools.py                    # EarlyStopping, StandardScaler
    └── voltage_metrics.py          # 电压异常检测指标
```

### 异常检测原理
采用**重构误差**方法：模型学习正常数据的时序模式，通过重构误差超过阈值来判定异常。
```python
threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
pred = (test_energy > threshold).astype(int)
```
评估时使用**点调整**(Point Adjustment)策略：异常段内任何点被正确检测，则整段标记为正确。

### 数据流程
1. `data_provider()` 加载数据，使用 StandardScaler 标准化
2. `exp.train()` 执行训练循环，MSE 重构损失，早停机制
3. `exp.test()` 在训练集计算阈值，在测试集判定异常并评估

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 模型隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet 的 top-k 周期数 |
| `--enc_in` | 25 | 输入特征数（PSM=25, RuralVoltage=17） |
| `--anomaly_ratio` | 1.0 | 预期异常比例(%) |

## 数据集

- **PSM/MSL/SMAP/SMD/SWAT**: 标准异常检测数据集（符号链接到 Time-Series-Library）
- **RuralVoltage**: 自定义农村电压数据集，17 维特征（Va/Vb/Vc 三相电压、电流、功率、谐波失真率、不平衡因子等）

数据格式：`train.csv`, `test.csv`, `test_label.csv`（0: 正常, 1+: 异常类型）

## TimesNet 模型详解

### 核心创新：1D→2D 时序建模
TimesNet 的核心思想是将一维时间序列转换为二维张量，利用 2D 卷积捕获周期内和周期间的复杂模式。

### 模型架构
```
输入 [B, T, C] → TokenEmbedding → TimesBlock×N → LayerNorm → 投影层 → 输出 [B, T, C]
```

### TimesBlock 工作流程
```python
# 1. FFT 周期发现
def FFT_for_Period(x, k=5):
    xf = torch.fft.rfft(x, dim=1)           # 频域变换
    frequency_list = abs(xf).mean(0).mean(-1)  # 取幅值
    top_k_freq = frequency_list.topk(k)      # 选 top-k 频率
    period = (T / top_k_freq.indices).int()  # 计算周期长度
    return period, top_k_freq.values

# 2. 1D → 2D 重塑
x_2d = x.reshape(B, period, num_periods, C)  # [B, T, C] → [B, p, T/p, C]

# 3. Inception 2D 卷积
x_2d = Inception_Block_V1(x_2d)  # 多尺度 2D 卷积核 (1×1, 3×3, 5×5, MaxPool)

# 4. 2D → 1D 还原
x_1d = x_2d.reshape(B, T, C)

# 5. 多周期自适应聚合
output = Σ(softmax(weights) × x_1d_i)  # 加权融合各周期结果
```

### VoltageTimesNet 改进
针对农村电压数据的定制优化：

1. **预设周期与 FFT 混合策略**
   - 电网固有周期：1min(60点), 5min(300点), 15min(900点), 1h(3600点)
   - 70% FFT 动态发现 + 30% 预设周期权重

2. **时域平滑层**
   - 深度可分离 1D 卷积抑制高频噪声
   - 保留电压波动的主要趋势

## 论文章节结构

论文位于 `thesis/` 目录，采用北京林业大学官方模板 (BJFUThesis)。

### 章节-代码对应关系

| 章节 | 文件 | 对应代码模块 |
|------|------|-------------|
| 第一章 绪论 | `chap00.tex` | - |
| 第二章 数据采集与预处理 | `chap01.tex` | `data_provider/`, `dataset/RuralVoltage/` |
| 第三章 基于TimesNet的电压异常检测算法 | `chap02.tex` | `models/TimesNet.py`, `models/VoltageTimesNet.py` |
| 第四章 实验设计与结果分析 | `chap03.tex` | `run.py`, `scripts/`, `exp/` |
| 第五章 农村电网低电压监管平台设计与实现 | `chap04.tex` | 系统设计文档 |
| 第六章 结论与展望 | `chap05.tex` | - |

### 论文编译
```bash
cd thesis
xelatex bjfuthesis-main.tex
biber bjfuthesis-main
xelatex bjfuthesis-main.tex
xelatex bjfuthesis-main.tex
```

### RuralVoltage 数据集 17 维特征
| 特征类别 | 特征名称 | 说明 |
|---------|---------|------|
| 三相电压 | Va, Vb, Vc | 200-240V 范围 |
| 三相电流 | Ia, Ib, Ic | 10-20A 范围 |
| 功率指标 | P, Q, S, PF | 有功/无功/视在功率及功率因数 |
| 电能质量 | THD_Va, THD_Vb, THD_Vc | 谐波失真率 (GB/T 12325-2008) |
| 不平衡因子 | V_unbalance, I_unbalance | 三相不平衡程度 |
| 频率 | Freq | 50Hz 标准频率 |

### 5 种异常类型
1. **Undervoltage (欠压)**: 电压低于 198V
2. **Overvoltage (过压)**: 电压高于 235V
3. **Voltage_Sag (电压骤降)**: 电压突然下降 10%+
4. **Harmonic (谐波畸变)**: THD 超过 5%
5. **Unbalance (三相不平衡)**: 不平衡度超过 2%

## 添加新模型

1. 在 `models/` 创建模型文件，实现 `__init__` 和 `forward` 方法
2. 在 `models/__init__.py` 的 `model_dict` 中注册
3. 确保 `forward` 返回与输入维度相同的重构输出

## Commit 规范

使用中文，格式：`类型: 描述`
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `refactor:` 重构代码

示例：`feat: 添加三相电压注意力层`
