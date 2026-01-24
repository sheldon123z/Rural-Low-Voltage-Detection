# 电压异常检测模块 (Voltage Anomaly Detection)

一个独立的时间序列异常检测框架，基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 构建，专门用于农村低压配电网电压异常检测研究。

## 📋 特性

- **15 种深度学习模型**支持异常检测任务
- **完全独立**：不依赖外部项目，可作为独立模块使用
- **即插即用**：简洁的命令行接口和配置系统
- **多数据集支持**：PSM、MSL、SMAP、SMD、SWAT 等标准异常检测数据集

## 🚀 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n tslib python=3.11
conda activate tslib

# 安装依赖
pip install -r requirements.txt
```

### 基础训练

```bash
# 使用 TimesNet 在 PSM 数据集上进行异常检测
python run.py --is_training 1 \
    --task_name anomaly_detection \
    --model TimesNet \
    --data PSM \
    --root_path ./dataset/PSM \
    --seq_len 100 \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --train_epochs 3 \
    --batch_size 32
```

## 📦 支持的模型

| 模型名称 | 参数量 | 论文 |
|---------|-------|------|
| **TimesNet** | 4.7M | [ICLR 2023](https://openreview.net/pdf?id=ju_Uqw384Oq) |
| **Transformer** | 107K | [NeurIPS 2017](https://arxiv.org/abs/1706.03762) |
| **DLinear** | 20K | [AAAI 2023](https://arxiv.org/abs/2205.13504) |
| **PatchTST** | 178K | [ICLR 2023](https://arxiv.org/abs/2211.14730) |
| **iTransformer** | 113K | [ICLR 2024](https://arxiv.org/abs/2310.06625) |
| **Autoformer** | 106K | [NeurIPS 2021](https://arxiv.org/abs/2106.13008) |
| **Informer** | 180K | [AAAI 2021](https://arxiv.org/abs/2012.07436) |
| **FiLM** | 12.6M | [NeurIPS 2022](https://arxiv.org/abs/2205.08897) |
| **LightTS** | 16K | [arXiv 2022](https://arxiv.org/abs/2207.01186) |
| **SegRNN** | 27K | [arXiv 2023](https://arxiv.org/abs/2308.11200) |
| **KANAD** | 111K | Kolmogorov-Arnold 网络 |
| **Nonstationary_Transformer** | 166K | [NeurIPS 2022](https://openreview.net/pdf?id=ucNDIDRNjjv) |
| **MICN** | 403K | [ICLR 2023](https://openreview.net/pdf?id=zt53IDUR1U) |
| **TimeMixer** | 124K | [ICLR 2024](https://arxiv.org/abs/2405.14616) |
| **Reformer** | 98K | [ICLR 2020](https://openreview.net/forum?id=rkgNKkHtvB) |

## 🔧 核心参数说明

### 任务相关
- `--task_name`: 任务类型，固定为 `anomaly_detection`
- `--model`: 模型名称（见上表）
- `--data`: 数据集名称 (PSM, MSL, SMAP, SMD, SWAT)

### 数据相关
- `--root_path`: 数据集根目录
- `--seq_len`: 输入序列长度 (默认: 100)
- `--batch_size`: 批量大小 (默认: 32)

### 模型相关
- `--d_model`: 模型维度 (默认: 64)
- `--d_ff`: 前馈网络维度 (默认: 64)
- `--e_layers`: 编码器层数 (默认: 2)
- `--n_heads`: 注意力头数 (默认: 8)
- `--dropout`: Dropout 率 (默认: 0.1)

### 训练相关
- `--train_epochs`: 训练轮数 (默认: 10)
- `--learning_rate`: 学习率 (默认: 0.0001)
- `--patience`: 早停耐心值 (默认: 3)

## 📁 项目结构

```
voltage_anomaly_detection/
├── data_provider/        # 数据加载模块
│   ├── data_factory.py   # 数据工厂
│   └── data_loader.py    # 数据集类
├── exp/                  # 实验模块
│   └── exp_anomaly_detection.py  # 异常检测实验类
├── layers/               # 网络层组件
│   ├── Embed.py          # 嵌入层
│   ├── SelfAttention_Family.py   # 注意力机制
│   ├── Transformer_EncDec.py     # Transformer 编解码器
│   ├── Autoformer_EncDec.py      # Autoformer 组件
│   └── ...
├── models/               # 模型定义
│   ├── TimesNet.py
│   ├── Transformer.py
│   ├── DLinear.py
│   └── ...  (15 个模型)
├── utils/                # 工具函数
│   ├── tools.py          # 工具函数
│   ├── metrics.py        # 评估指标
│   └── masking.py        # 掩码工具
├── dataset/              # 数据集目录
├── checkpoints/          # 模型检查点
├── run.py                # 主运行脚本
├── test_models.py        # 模型测试脚本
└── README.md
```

## 💡 使用示例

### 1. 测试所有模型

```bash
python test_models.py
```

### 2. 在 PSM 数据集上训练 TimesNet

```bash
python run.py --is_training 1 \
    --task_name anomaly_detection \
    --model TimesNet \
    --data PSM \
    --root_path ./dataset/PSM \
    --seq_len 100 \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --top_k 5 \
    --num_kernels 6 \
    --train_epochs 10 \
    --batch_size 32
```

### 3. 比较不同模型

```bash
# TimesNet
python run.py --model TimesNet --data PSM --root_path ./dataset/PSM

# Transformer
python run.py --model Transformer --data PSM --root_path ./dataset/PSM

# DLinear (轻量级)
python run.py --model DLinear --data PSM --root_path ./dataset/PSM

# PatchTST
python run.py --model PatchTST --data PSM --root_path ./dataset/PSM
```

### 4. 使用自定义数据集

将数据准备为以下格式并放入 `dataset/` 目录：

```
dataset/
└── MyDataset/
    ├── train.csv      # 训练数据
    ├── test.csv       # 测试数据
    └── test_label.csv # 测试标签 (0: 正常, 1: 异常)
```

然后运行：
```bash
python run.py --data custom --root_path ./dataset/MyDataset
```

## 📊 评估指标

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1 分数
- **AUC-ROC**: ROC 曲线下面积

## 🔍 异常检测原理

本模块采用**重构误差**方法进行异常检测：

1. **训练阶段**: 模型学习正常数据的时序模式
2. **推理阶段**: 计算每个时间点的重构误差
3. **异常判定**: 重构误差超过阈值的点被标记为异常

阈值计算方法：
```python
threshold = np.percentile(train_anomaly_scores, 100 - anomaly_ratio)
```

## 🎯 针对低压配电网的应用

本模块可用于检测低压配电网中的电压异常：

1. **电压骤降/骤升**: 突然的电压变化
2. **谐波异常**: 非正弦波形
3. **负荷波动**: 异常的负荷变化模式
4. **设备故障**: 由设备故障引起的电压异常

## 📜 许可证

MIT License

## 🙏 致谢

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - 基础框架
- [TimesNet](https://github.com/thuml/TimesNet) - 核心模型
