---
name: thesis-plot
description: 生成符合论文格式要求的高质量科研图表，专注于时序异常检测可视化
user-invocable: true
---

# 论文绘图技能

为农村电压异常检测论文生成符合学术规范的高质量图表。

## 支持的图表类型

### 时序可视化

| 类型代码 | 名称 | 用途 |
|---------|------|------|
| `timeseries` | 电压时序曲线 | 展示三相电压变化和异常区间 |
| `reconstruction` | 重构误差分布 | 展示正常/异常样本的误差分布 |
| `detection` | 检测结果对比 | 展示预测结果与真实标签 |

### 性能评估

| 类型代码 | 名称 | 用途 |
|---------|------|------|
| `comparison` | 模型对比柱状图 | 多模型多指标性能对比 |
| `roc` | ROC曲线 | 不同阈值下的性能 |
| `pr` | PR曲线 | 精确率-召回率曲线 |
| `confusion` | 混淆矩阵 | 分类性能详细分布 |
| `loss` | 训练损失曲线 | 训练过程可视化 |

### 特征分析

| 类型代码 | 名称 | 用途 |
|---------|------|------|
| `tsne` | t-SNE降维 | 高维特征分布可视化 |
| `pca` | PCA降维 | 主成分分析可视化 |
| `correlation` | 相关性热力图 | 特征相关性分析 |
| `radar` | 雷达图 | 电压质量指标展示 |

### 周期/频域分析

| 类型代码 | 名称 | 用途 |
|---------|------|------|
| `fft` | FFT频谱图 | 时序信号频域特征 |
| `period` | 周期热力图 | TimesNet 2D周期建模 |

### 异常分析

| 类型代码 | 名称 | 用途 |
|---------|------|------|
| `anomaly_dist` | 异常类型分布 | 不同异常类型占比 |
| `difference` | 模型差异图 | 多模型检测差异对比 |

## 快速使用

### 单图生成

```bash
# 电压时序图
/plot type=timeseries data=dataset/RuralVoltage/test.csv

# 模型对比图
/plot type=comparison

# 混淆矩阵
/plot type=confusion model=TimesNet

# t-SNE 可视化
/plot type=tsne features=embeddings.npy labels=test_label.csv
```

### 批量生成

```bash
# 生成第3章所有图表
/plot chapter=3 --all

# 生成第4章实验图表
/plot chapter=4 types=comparison,confusion,roc
```

## 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| type | string | - | 图表类型（必需） |
| data | path | - | 数据文件路径 |
| output | path | thesis/figures/ | 输出目录 |
| format | string | pdf | 输出格式 (pdf/png/svg) |
| dpi | int | 300 | 分辨率 |
| chapter | int | 3 | 章节号 |
| fig_num | int | auto | 图编号 |
| width | float | 7.0 | 图宽度（英寸） |
| height | float | auto | 图高度（英寸） |

### 样式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| style | string | thesis | 样式模板 |
| colormap | string | default | 颜色方案 |
| font_cn | string | SimHei | 中文字体 |
| font_en | string | Times New Roman | 英文字体 |
| grid | bool | true | 显示网格 |

### 标题参数

| 参数 | 类型 | 说明 |
|------|------|------|
| title_cn | string | 中文标题 |
| title_en | string | 英文标题 |
| xlabel | string | X轴标签 |
| ylabel | string | Y轴标签 |

## 图表规格

### 论文格式标准

```yaml
字体大小:
  标题: 五号 (10.5pt)
  坐标轴标签: 五号 (10.5pt)
  刻度标签: 小五号 (9pt)
  图例: 小五号 (9pt)

图片规格:
  单栏宽度: 3.5 英寸 (约 8.9 cm)
  双栏宽度: 7.0 英寸 (约 17.8 cm)
  最大宽度: 6.0 英寸 (约 15.2 cm，版心宽度）
  分辨率: 300 DPI

颜色方案:
  - '#1f77b4'  # 蓝色 - 主色
  - '#ff7f0e'  # 橙色
  - '#2ca02c'  # 绿色
  - '#d62728'  # 红色 - 异常标注
  - '#9467bd'  # 紫色
```

### 图例规范

```yaml
位置: 自动最佳 (loc='best')
字体: 小五号 Times New Roman
边框: 黑色细线
透明度: 0.9
单列显示: 默认
```

## 输出示例

### 文件命名规则

```
fig_{chapter}_{number}_{type}.pdf

示例:
fig_3_1_voltage_timeseries.pdf
fig_3_2_model_comparison.pdf
fig_3_3_confusion_matrix.pdf
fig_4_1_tsne_visualization.pdf
```

### 生成结果

```
✅ 图表生成成功:
   - 文件: thesis/figures/chap3/fig_3_1_voltage_timeseries.pdf
   - 格式: PDF (矢量)
   - 尺寸: 7.0 x 3.5 英寸
   - 分辨率: 300 DPI

📝 LaTeX 引用代码:
   \begin{figure}[htbp]
     \centering
     \includegraphics[width=0.9\textwidth]{figures/chap3/fig_3_1_voltage_timeseries.pdf}
     \caption{农村电压三相时序曲线}
     \label{fig:voltage_timeseries}
   \end{figure}
```

## 高级用法

### 自定义样式

```bash
/plot type=comparison \
  colormap=viridis \
  style=thesis \
  width=6.5 \
  title_cn="模型性能对比分析" \
  title_en="Model Performance Comparison"
```

### 子图组合

```bash
/plot type=subplot \
  layout=2x2 \
  types=timeseries,confusion,roc,tsne \
  chapter=4 \
  fig_num=1
```

### 数据驱动

```bash
# 从实验结果文件直接生成
/plot type=all \
  results=code/voltage_anomaly_detection/result_anomaly_detection.txt \
  chapter=4
```

## 依赖工具

```python
# 核心依赖
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0

# 可选依赖
plotly>=5.0.0  # 交互式图表
```

## 相关资源

- Agent: `.claude/agents/thesis-plotter.md`
- 命令: `.claude/commands/plot.md`
- 代码库: `code/voltage_anomaly_detection/visualization/`
- 输出目录: `thesis/figures/`
