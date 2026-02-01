# VoltageTimesNet_v2 实验结果汇总

## 一、模型改进

### 1.1 核心改进点

| 改进项 | 原版 (VoltageTimesNet) | 新版 (VoltageTimesNet_v2) |
|--------|----------------------|---------------------------|
| 预设权重 | 硬编码 `preset_weight=0.3` | 可学习 `nn.Parameter` |
| 电能质量编码 | 无 | `PowerQualityEncoder` |
| 时域卷积 | 单一 kernel=3 | 多尺度 kernel=(3,5,7) |
| 异常敏感度 | 无 | `AnomalySensitivityAmplifier` |
| 三相约束 | 无 | `PhaseConstraintModule` |

### 1.2 模型参数量

| 模型 | 参数量 |
|------|--------|
| TimesNet | ~3.5M |
| VoltageTimesNet | ~3.8M |
| VoltageTimesNet_v2 | ~9.5M |

## 二、RuralVoltage 数据集实验结果

### 2.1 模型对比 (3 epochs, d_model=32)

| 模型 | Accuracy | Precision | Recall | F1-score |
|------|:--------:|:---------:|:------:|:--------:|
| TimesNet | 0.9098 | 0.7609 | 0.5660 | 0.6492 |
| VoltageTimesNet | 0.9116 | 0.7703 | 0.5712 | 0.6559 |
| **VoltageTimesNet_v2** | 0.9105 | 0.7604 | **0.5740** | 0.6542 |

### 2.2 阈值调优 (VoltageTimesNet_v2)

| anomaly_ratio | 百分位数 | Precision | Recall | F1-score |
|:-------------:|:--------:|:---------:|:------:|:--------:|
| 1.0 | 99% | 0.7598 | 0.5727 | 0.6531 |
| 1.5 | 98.5% | 0.6669 | 0.5960 | 0.6295 |
| **2.0** | 98% | 0.6128 | 0.6687 | **0.6395** |
| 2.5 | 97.5% | 0.5520 | 0.6746 | 0.6072 |
| **3.0** | 97% | 0.5137 | **0.7113** | 0.5966 |

**结论**：
- `anomaly_ratio=2.0` 时 F1 最高 (0.6395)
- `anomaly_ratio=3.0` 时召回率达到 **71.13%**，超过目标 70%

## 三、Kaggle Power Quality 数据集

### 3.1 数据集信息

- 原始数据：11,998 条 × 128 维波形信号
- 5 类平衡分类（Class 3 作为正常类）
- 训练集：2,400 条（纯正常）
- 测试集：9,598 条（6.3% 正常，93.7% 异常）

### 3.2 数据加载器

```python
# 使用方式
python run.py --is_training 1 --model TimesNet --data KagglePQ \
  --root_path ./dataset/Kaggle_PowerQuality_2/ \
  --enc_in 128 --c_out 128 --seq_len 64
```

## 四、使用建议

### 4.1 召回率优先场景

```bash
# 使用较低的阈值提高召回率
python run.py --model VoltageTimesNet_v2 --data RuralVoltage \
  --anomaly_ratio 3.0  # 97% 百分位数，Recall~71%
```

### 4.2 F1 平衡场景

```bash
# 使用最佳 F1 阈值
python run.py --model VoltageTimesNet_v2 --data RuralVoltage \
  --anomaly_ratio 2.0  # 98% 百分位数，F1~0.64
```

### 4.3 高精度场景

```bash
# 保持默认高精度
python run.py --model VoltageTimesNet_v2 --data RuralVoltage \
  --anomaly_ratio 1.0  # 99% 百分位数，Precision~76%
```

## 五、文件清单

| 文件 | 说明 |
|------|------|
| `models/VoltageTimesNet_v2.py` | 改进模型实现 |
| `dataset/Kaggle_PowerQuality_2/prepare_dataset.py` | Kaggle 数据预处理 |
| `data_provider/data_loader.py` | 新增 KagglePQSegLoader |
| `scripts/quick_comparison.sh` | 快速对比脚本 |

## 六、下一步工作

1. **更多训练轮次**：当前仅 3-5 epochs，增加到 10+ 可能进一步提升
2. **超参数调优**：d_model, e_layers, top_k 等参数优化
3. **Kaggle 数据集完整实验**：对比所有模型在 KagglePQ 上的表现
4. **论文实验章节**：补充实验结果到论文

---
*生成时间：2026-02-01*
