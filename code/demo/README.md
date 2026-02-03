---
title: 农村低压配电网电压异常检测
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 农村低压配电网电压异常检测系统

基于 TimesNet 的时间序列异常检测方法研究与应用

## 功能

- **原理演示**: FFT 周期发现可视化
- **创新对比**: VoltageTimesNet vs TimesNet
- **模型竞技场**: 6 模型性能对比
- **模型结构**: 网络架构图展示
- **自定义检测**: 上传 CSV 进行异常检测

## 模型

| 模型 | F1 | Recall | 说明 |
|:-----|:--:|:------:|:-----|
| VoltageTimesNet_v2 | 0.6622 | 0.5858 | 最优模型 |
| TimesNet | 0.6520 | 0.5705 | 基线 |

## 链接

- [GitHub](https://github.com/sheldon123z/Rural-Low-Voltage-Detection)
- [数据集](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets)
- [模型权重](https://huggingface.co/Sheldon123z/rural-voltage-detection-models)
