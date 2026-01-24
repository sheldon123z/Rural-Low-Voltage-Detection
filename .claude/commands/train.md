---
name: train
description: 快速生成和运行时序异常检测模型训练脚本
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
---

# /train 命令

快速生成并可选执行时序异常检测模型训练脚本。

## 命令格式

```bash
/train [参数列表] [--run] [--compare]
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model | TimesNet | 模型名称，多个用逗号分隔 |
| dataset | RuralVoltage | 数据集名称 |
| seq_len | 100 | 序列长度 |
| d_model | 64 | 隐藏维度 |
| d_ff | 128 | FFN维度 |
| e_layers | 2 | 编码器层数 |
| top_k | 5 | 周期数 |
| batch_size | 32 | 批次大小 |
| epochs | 10 | 训练轮数 |
| lr | 0.0001 | 学习率 |
| patience | 3 | 早停耐心 |
| anomaly_ratio | 1.0 | 异常比例(%) |
| gpu | 0 | GPU设备号 |

## 特殊标志

| 标志 | 说明 |
|------|------|
| --run | 生成后立即执行脚本 |
| --compare | 生成多模型对比实验 |
| --test-only | 仅测试模式（加载检查点） |
| --dry-run | 仅显示将执行的命令 |

## 使用示例

### 快速开始

```bash
# 生成 TimesNet 训练脚本
/train

# 生成并运行
/train model=VoltageTimesNet --run

# 使用 PSM 数据集
/train dataset=PSM model=TimesNet
```

### 自定义配置

```bash
# 调整超参数
/train model=TimesNet seq_len=200 d_model=128 epochs=20

# 多模型对比
/train model=TimesNet,DLinear,Informer --compare

# 测试已训练模型
/train model=TimesNet --test-only
```

### 完整配置示例

```bash
/train model=VoltageTimesNet dataset=RuralVoltage seq_len=100 d_model=64 d_ff=128 e_layers=2 top_k=5 batch_size=32 epochs=10 lr=0.0001 patience=3 anomaly_ratio=1.0 gpu=0 --run
```

## 执行流程

1. **解析参数**: 从命令行提取所有参数
2. **验证配置**: 检查模型和数据集有效性
3. **生成脚本**: 创建 bash 训练脚本
4. **设置权限**: chmod +x
5. **可选执行**: 如有 --run 则立即执行
6. **输出结果**: 显示脚本路径和下一步操作

## 输出位置

```
code/voltage_anomaly_detection/scripts/{dataset}/{model}.sh
```

## 快速运行模板

执行此命令后，Claude 应该：

1. 读取 gen-train-script 技能获取模板
2. 解析用户参数，填充默认值
3. 映射数据集到特征数 (enc_in, c_out)
4. 生成 bash 脚本到指定目录
5. 设置可执行权限
6. 如有 --run 标志，执行脚本并监控输出
7. 返回脚本路径和运行状态

## 数据集特征数映射

```python
DATASET_FEATURES = {
    'RuralVoltage': 17,
    'PSM': 25,
    'MSL': 55,
    'SMAP': 25,
    'SMD': 38,
    'SWAT': 51
}
```

## 生成脚本模板

```bash
#!/bin/bash
# {model} Training Script for {dataset}
# Generated: {timestamp}
# Command: /train {args}

export CUDA_VISIBLE_DEVICES={gpu}

cd "$(dirname "$0")/../.."

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/{dataset}/ \
  --model_id {dataset}_{model}_{seq_len} \
  --model {model} \
  --data {dataset} \
  --features M \
  --seq_len {seq_len} \
  --pred_len 0 \
  --d_model {d_model} \
  --d_ff {d_ff} \
  --e_layers {e_layers} \
  --enc_in {enc_in} \
  --c_out {c_out} \
  --top_k {top_k} \
  --num_kernels 6 \
  --batch_size {batch_size} \
  --train_epochs {epochs} \
  --patience {patience} \
  --learning_rate {lr} \
  --anomaly_ratio {anomaly_ratio} \
  --des '{model}_Exp'
```

## 关联资源

- 技能文件: `.claude/skills/gen-train-script/SKILL.md`
- 模型目录: `code/voltage_anomaly_detection/models/`
- 数据集目录: `code/voltage_anomaly_detection/dataset/`
- 已有脚本: `code/voltage_anomaly_detection/scripts/`
