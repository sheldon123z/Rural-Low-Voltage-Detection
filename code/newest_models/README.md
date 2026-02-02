---
language:
  - zh
  - en
tags:
  - time-series
  - anomaly-detection
  - voltage
  - power-grid
  - timesnet
  - pytorch
datasets:
  - custom
metrics:
  - f1
  - precision
  - recall
pipeline_tag: time-series-forecasting
---

# Rural Low-Voltage Detection Models

农村低压配电网电压异常检测实验模型检查点。

## 🔗 相关链接

- **数据集**: [Sheldon123z/rural-voltage-datasets](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets)
- **代码仓库**: [GitHub - Rural-Low-Voltage-Detection](https://github.com/sheldon123z/Rural-Low-Voltage-Detection)

## 目录结构

```
├── RuralVoltage/          # 农村电压数据集（论文核心）
│   ├── TimesNet_sl100_dm64/
│   ├── VoltageTimesNet_sl100_dm64/
│   ├── VoltageTimesNet_v2_sl100_dm64/  # 论文最优模型
│   ├── TPATimesNet_sl100_dm64/
│   ├── DLinear_sl100_dm64/
│   └── PatchTST_sl100_dm64/
├── PSM/                   # PSM 公开基准数据集
│   ├── TimesNet_sl100_dm64/
│   ├── VoltageTimesNet_sl100_dm64/
│   ├── TPATimesNet_sl100_dm64/
│   ├── MTSTimesNet_sl100_dm64/
│   ├── DLinear_sl100_dm64/
│   ├── Autoformer_sl100_dm64/
│   ├── iTransformer_sl100_dm64/
│   ├── Transformer_sl100_dm64/
│   └── HybridTimesNet_sl100_dm64/
├── KagglePQ/              # Kaggle 电力质量数据集
│   ├── TimesNet_sl64_dm64/
│   ├── VoltageTimesNet_sl64_dm64/
│   ├── VoltageTimesNet_v2_sl64_dm64/
│   ├── DLinear_sl64_dm64/
│   └── PatchTST_sl64_dm64/
├── MSL/                   # MSL 数据集
│   └── TimesNet_sl100_dm64/
├── SMAP/                  # SMAP 数据集
│   └── TimesNet_sl100_dm64/
├── SMD/                   # SMD 数据集
│   └── TimesNet_sl100_dm64/
└── ablation/              # 消融实验
    ├── seq_len/           # 序列长度消融
    └── alpha/             # 预设权重消融
```

## 模型说明

| 模型 | 说明 |
|------|------|
| **VoltageTimesNet_v2** | 论文最终版本，包含可学习预设权重、异常敏感度增强、多尺度时域卷积 |
| **VoltageTimesNet** | 早期版本，预设周期 + FFT 混合 |
| **TPATimesNet** | 三相注意力机制 |
| **TimesNet** | 基线模型，FFT + 2D卷积 |
| **DLinear** | 轻量级基线 |
| **PatchTST** | Patch 时序 Transformer |

## 论文最优结果

| 数据集 | 模型 | F1 Score |
|--------|------|:--------:|
| RuralVoltage | VoltageTimesNet_v2 | 0.6622 |
| PSM | TimesNet | 0.9735 |

## 使用方法

```python
import torch
from huggingface_hub import hf_hub_download

# 下载模型
checkpoint_path = hf_hub_download(
    repo_id="Sheldon123z/rural-voltage-detection-models",
    filename="RuralVoltage/VoltageTimesNet_v2_sl100_dm64/checkpoint.pth"
)

# 加载模型
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
```

## 配置说明

- `sl100`: seq_len=100（输入序列长度）
- `dm64`: d_model=64（隐藏维度）
