---
name: plot
description: 快速生成符合论文格式的科研图表
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
---

# /plot 命令

快速生成符合北京林业大学硕士论文格式要求的科研图表。

## 命令格式

```bash
/plot type=<图表类型> [参数...] [--preview] [--save]
```

## 图表类型速查

| 类型 | 命令 | 说明 |
|------|------|------|
| 电压时序 | `/plot type=timeseries` | 三相电压波形 + 异常标注 |
| 模型对比 | `/plot type=comparison` | 多模型性能柱状图 |
| 混淆矩阵 | `/plot type=confusion` | 分类混淆矩阵热力图 |
| ROC曲线 | `/plot type=roc` | ROC + AUC |
| PR曲线 | `/plot type=pr` | Precision-Recall |
| t-SNE | `/plot type=tsne` | 特征降维可视化 |
| 误差分布 | `/plot type=reconstruction` | 重构误差直方图 |
| 训练曲线 | `/plot type=loss` | 训练/验证损失 |
| 相关性 | `/plot type=correlation` | 特征相关性热力图 |
| 雷达图 | `/plot type=radar` | 电压质量指标 |
| FFT频谱 | `/plot type=fft` | 频域分析 |

## 快速示例

### 基础用法

```bash
# 生成电压时序图
/plot type=timeseries

# 生成模型对比图
/plot type=comparison

# 生成混淆矩阵
/plot type=confusion model=TimesNet
```

### 指定数据

```bash
# 使用指定数据文件
/plot type=timeseries data=dataset/RuralVoltage/test.csv

# 使用实验结果
/plot type=comparison results=result_anomaly_detection.txt
```

### 自定义输出

```bash
# 指定输出位置和格式
/plot type=comparison output=thesis/figures/chap4/ format=pdf

# 指定章节和编号
/plot type=confusion chapter=4 fig_num=3
```

### 批量生成

```bash
# 生成第3章所有图
/plot chapter=3 --all

# 生成多种类型
/plot types=comparison,confusion,roc chapter=4
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| type | 图表类型 | timeseries, comparison |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| data | auto | 数据文件路径 |
| model | all | 模型名称 |
| dataset | RuralVoltage | 数据集名称 |
| output | thesis/figures/ | 输出目录 |
| format | pdf | 输出格式 |
| chapter | 3 | 章节号 |
| fig_num | auto | 图编号 |
| dpi | 300 | 分辨率 |
| width | 7.0 | 图宽度(英寸) |

### 标志参数

| 标志 | 说明 |
|------|------|
| --preview | 仅预览不保存 |
| --save | 保存并显示路径 |
| --all | 生成所有类型 |
| --no-title | 不添加标题 |
| --bilingual | 中英文双语 |

## 输出格式

### 文件命名

```
fig_{chapter}_{number}_{description}.{format}
```

### 输出位置

```
thesis/figures/chap{chapter}/
```

### 生成后输出

```
✅ 图表生成成功!

📊 文件信息:
   路径: thesis/figures/chap3/fig_3_1_voltage_timeseries.pdf
   格式: PDF (矢量)
   尺寸: 7.0 × 3.5 英寸
   分辨率: 300 DPI

📝 LaTeX 引用:
   \begin{figure}[htbp]
     \centering
     \includegraphics[width=0.9\textwidth]{figures/chap3/fig_3_1_voltage_timeseries.pdf}
     \caption{农村电压三相时序曲线}
     \label{fig:voltage_timeseries}
   \end{figure}
```

## 执行流程

1. **解析参数**: 提取类型、数据源、输出配置
2. **加载数据**: 读取指定数据文件或实验结果
3. **调用绑图函数**: 使用 thesis_plots 模块
4. **应用论文样式**: 字体、大小、颜色规范
5. **保存图表**: 输出到指定位置
6. **生成引用代码**: 输出 LaTeX 引用片段

## 常用场景

### 场景1: 实验结果可视化

```bash
# 训练完模型后，生成所有结果图
/plot type=comparison,confusion,roc chapter=4

# 生成训练过程图
/plot type=loss data=checkpoints/train_log.csv
```

### 场景2: 数据分析图

```bash
# 分析电压数据
/plot type=timeseries,fft,correlation chapter=2 data=dataset/RuralVoltage/train.csv
```

### 场景3: 特征可视化

```bash
# t-SNE 可视化正常/异常样本
/plot type=tsne features=embeddings.npy labels=test_label.csv
```

### 场景4: 论文图表批量生成

```bash
# 生成第3章（算法）所有图
/plot chapter=3 types=timeseries,reconstruction,fft --all

# 生成第4章（实验）所有图
/plot chapter=4 types=comparison,confusion,roc,loss --all
```

## 相关资源

- 技能: `.claude/skills/thesis-plot/SKILL.md`
- Agent: `.claude/agents/thesis-plotter.md`
- 代码库: `code/voltage_anomaly_detection/visualization/thesis_plots.py`

## 注意事项

1. 确保已安装 matplotlib, seaborn 等依赖
2. 中文显示需要安装 SimHei 或 Microsoft YaHei 字体
3. PDF 格式适合论文，PNG 适合预览和PPT
4. 图表宽度不应超过论文版心宽度（约15cm）
