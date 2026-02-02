# Scripts 目录说明

本目录包含农村低压配电网电压异常检测项目的训练和分析脚本。

## 当前可用脚本

### 主实验脚本 (推荐)

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `full_experiment.sh` | 完整论文实验套件，运行所有模型对比和阈值分析 | `bash full_experiment.sh` |
| `supplementary_experiments.py` | 补充实验（经典基线、统计显著性检验、超参数优化） | `python supplementary_experiments.py --help` |

### 消融实验脚本

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `ablation_alpha.sh` | VoltageTimesNet α参数敏感性分析 | `bash ablation_alpha.sh` |
| `ablation_seq_len.sh` | 序列长度消融实验 | `bash ablation_seq_len.sh` |

### 单模型训练脚本

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `train_rural_voltage.sh` | RuralVoltage 数据集快速训练 | `bash train_rural_voltage.sh` |

### 结果分析脚本

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `analyze_comparison_results.py` | 多模型对比结果分析与可视化 | `python analyze_comparison_results.py --result_dir ./results/xxx` |

### 架构图生成脚本

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `draw_timesnet_architecture.py` | 绘制 TimesNet 架构图 | `python draw_timesnet_architecture.py` |
| `generate_voltagetimesnet_architecture.py` | 绘制 VoltageTimesNet 架构图 | `python generate_voltagetimesnet_architecture.py` |

## 数据集专用脚本目录

| 目录 | 说明 |
|------|------|
| `PSM/` | PSM 数据集配置和运行脚本 |
| `RuralVoltage/` | RuralVoltage 数据集各模型训练脚本 |
| `MSL/` | MSL 数据集脚本 |
| `SMAP/` | SMAP 数据集脚本 |
| `SMD/` | SMD 数据集脚本 |
| `SWAT/` | SWAT 数据集脚本 |

## 其他目录

| 目录 | 说明 |
|------|------|
| `common/` | 通用工具函数 |
| `bindplotting/` | 绑图相关工具 |
| `archived/` | 已归档的过时脚本（保留供参考） |

## 典型工作流

### 1. 完整论文实验
```bash
# 运行所有对比实验
bash full_experiment.sh

# 运行补充实验（基线对比、统计检验）
python supplementary_experiments.py --all
```

### 2. 消融实验
```bash
# α 参数消融
bash ablation_alpha.sh

# 序列长度消融
bash ablation_seq_len.sh
```

### 3. 单数据集快速测试
```bash
# RuralVoltage
bash train_rural_voltage.sh

# 或使用数据集专用目录
bash RuralVoltage/TimesNet.sh
```

### 4. 结果分析
```bash
# 分析实验结果
python analyze_comparison_results.py --result_dir ./results/full_experiment_XXXXXXXX
```

## 注意事项

1. 运行脚本前确保已激活正确的 conda 环境
2. 检查 GPU 可用性：`nvidia-smi`
3. 如遇到数据加载问题，尝试设置 `--num_workers 0`
4. 实验结果保存在 `../results/` 目录
5. 训练日志保存在 `../logs/` 目录

---
更新时间: 2026-02-02
