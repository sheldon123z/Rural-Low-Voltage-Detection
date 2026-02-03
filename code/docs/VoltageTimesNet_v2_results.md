# VoltageTimesNet_v2 实验结果汇总

> **⚠️ 文档状态**：本文档部分数据已过时（2026-02-01）
> **最新结果（2026-02-03）**：经 Optuna 30-trial 超参数优化后
> - **F1 = 0.8149** (原 0.6395) | **Recall = 0.9110** (原 0.7113) | **Accuracy = 0.9393**
> - 详见 `实验结果汇总_论文写作材料.md`

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

## 六、Optuna 超参数优化结果（2026-02-03 更新）

### 6.1 最优配置（Trial 10, 30-trial TPE Sampler 搜索）

| 参数 | 默认值 | 优化值 |
|------|:------:|:------:|
| d_model | 64 | **128** |
| e_layers | 2 | **3** |
| d_ff | 64 | **256** |
| top_k | 5 | **2** |
| num_kernels | 6 | **8** |
| learning_rate | 0.0001 | **0.0042** |
| batch_size | 128 | **32** |
| dropout | 0.1 | **0.017** |
| seq_len | 100 | **50** |
| anomaly_ratio | 1.0 | **2.08** |

### 6.2 优化后性能

| 指标 | 默认参数 | Optuna 优化 | 提升 |
|------|:--------:|:----------:|:----:|
| F1 | 0.6178 | **0.8149** | +31.9% |
| Recall | 0.7113 | **0.9110** | +28.1% |
| Precision | 0.5463 | **0.7371** | +34.9% |
| Accuracy | 0.8652 | **0.9393** | +8.6% |

### 6.3 关键发现

1. 较短序列 (seq_len=50) 更适合电压异常检测
2. 较高学习率 (0.004) + 小批量 (32) 组合更优
3. 仅需 top_k=2 个周期即可有效建模
4. 低 dropout (0.017) 说明模型不易过拟合

## 七、下一步工作

1. ✅ ~~超参数调优~~ → 已通过 Optuna 完成
2. [ ] 使用最优配置训练更多 epochs（10+）
3. [ ] Kaggle 数据集完整实验
4. [ ] 统计显著性测试（多次运行标准差）

---
*生成时间：2026-02-01 | 更新时间：2026-02-03*
