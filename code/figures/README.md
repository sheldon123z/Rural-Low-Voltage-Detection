# 论文图表目录

农村低压配电网电压异常检测论文的所有图表集中存储在此目录。

## 目录结构

```
figures/
├── architecture/              # 模型架构图（HTML/draw.io 交互式）
│   ├── fig_timesnet_overview*.html      # TimesNet 整体架构
│   ├── fig_fft_period_discovery.html    # FFT 周期发现模块
│   ├── fig_2d_conv_inception.html       # 2D 卷积 Inception 块
│   ├── fig_voltage_timesnet*.html       # VoltageTimesNet 架构
│   ├── fig_anomaly_pipeline.html        # 异常检测流程
│   └── fig_model_comparison.html        # 模型对比
│
├── thesis/                    # 论文正文图表（PDF/PNG 高清）
│   ├── chapter3_method/       # 第三章：研究方法
│   │   └── fig_3_1_sliding_window.*     # 滑动窗口预测示意图
│   │
│   ├── chapter4_experiments/  # 第四章：实验结果
│   │   ├── fig_4_1_f1_comparison.*      # 多模型 F1 对比
│   │   ├── fig_4_2_roc_pr_curves.*      # ROC/PR 曲线
│   │   ├── fig_4_3_confusion_matrices.* # 混淆矩阵
│   │   ├── fig_4_4_precision_recall_tradeoff.* # P-R 权衡
│   │   ├── fig_4_5_radar_comparison.*   # 雷达图对比
│   │   ├── fig_4_6_score_distribution.* # 异常得分分布
│   │   ├── fig_4_7_threshold_sensitivity.* # 阈值敏感性
│   │   ├── fig_4_8_detection_visualization.* # 检测可视化
│   │   ├── fig_4_9_training_loss.*      # 训练损失曲线
│   │   ├── fig_4_10_seq_len_ablation.*  # 序列长度消融
│   │   └── fig_4_11_alpha_ablation.*    # Alpha 参数消融
│   │
│   └── scripts/               # 绑图脚本
│       ├── fig_*.py           # 独立绑图脚本（每图一脚本）
│       └── archived/          # 归档的合并脚本
│
├── experiments/               # 实验分析图（从 results/ 复制）
│   ├── psm/                   # PSM 数据集实验图
│   └── rural_voltage/         # RuralVoltage 数据集实验图
│
└── archived/                  # 归档的旧图表
    └── old_outputs/           # 旧的输出图表
```

## 图表规格

| 类型 | 格式 | 分辨率 | 用途 |
|------|------|--------|------|
| 论文正文 | PDF | 600 DPI | 插入 Word/LaTeX |
| 预览/PPT | PNG | 300 DPI | 演示和预览 |
| 架构图 | HTML | 矢量 | 交互式查看 |

## 重新生成图表

```bash
# 生成单张图表
cd figures/thesis/scripts
python fig_4_1_f1_comparison.py

# 批量生成所有图表
for f in fig_*.py; do python "$f"; done
```

## 图表命名规范

- `fig_[章节]_[序号]_[描述].[格式]`
- 例：`fig_4_1_f1_comparison.pdf`

## 相关资源

- 数据集：[HuggingFace](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets)
- 模型：[HuggingFace](https://huggingface.co/Sheldon123z/rural-voltage-detection-models)
