#!/usr/bin/env python3
"""
图4-1: 多模型F1分数对比柱状图（RuralVoltage 数据集）
Fig 4-1: F1-score Comparison of Multiple Models on RuralVoltage Dataset

输出文件: ../chapter4_experiments/fig_4_1_f1_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os
import sys

# 导入论文统一样式
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import (
    setup_thesis_style, save_thesis_figure, remove_spines, get_model_colors,
    TIMESNET_COLORS, OTHER_COLORS
)

# 初始化样式
setup_thesis_style()

# ============================================================
# 实验数据（RuralVoltage 数据集）
# ============================================================
models = ['VoltageTimesNet', 'TimesNet',
          'Isolation Forest', 'One-Class SVM', 'LSTMAutoEncoder']
f1_scores = [0.8149, 0.5970, 0.5157, 0.5157, 0.4457]

# 按 F1 降序排列（已排好）
sorted_indices = np.argsort(f1_scores)[::-1]
models = [models[i] for i in sorted_indices]
f1_scores = [f1_scores[i] for i in sorted_indices]

# 模型简称（用于 x 轴显示）
label_map = {
    'VoltageTimesNet': 'VoltageTimesNet',
    'TimesNet': 'TimesNet',
    'LSTMAutoEncoder': 'LSTM-AE',
    'Isolation Forest': 'iForest',
    'One-Class SVM': 'OC-SVM',
}
model_labels = [label_map.get(m, m) for m in models]

# 获取模型颜色
model_colors = get_model_colors(models)


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, f1_scores, width=0.6, color=model_colors,
                  edgecolor='black', linewidth=0.8)

    # 在柱子上方标注数值
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # 设置坐标轴
    ax.set_xlabel('模型/Model', fontsize=10.5)
    ax.set_ylabel('F1分数/F1-Score', fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.set_axisbelow(True)

    # 添加图例说明颜色含义
    legend_elements = [
        mpatches.Patch(facecolor=TIMESNET_COLORS[0], edgecolor='black',
                       label='TimesNet系列'),
        mpatches.Patch(facecolor=OTHER_COLORS[1], edgecolor='black',
                       label='其他深度学习模型'),
        mpatches.Patch(facecolor=OTHER_COLORS[2], edgecolor='black',
                       label='传统机器学习模型'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # 三线表风格
    remove_spines(ax)

    # 保存到 chapter4_experiments 目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'chapter4_experiments')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_1_f1_comparison.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
