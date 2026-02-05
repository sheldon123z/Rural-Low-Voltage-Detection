#!/usr/bin/env python3
"""
图4-5: TimesNet系列模型演进对比雷达图（RuralVoltage 数据集）
Fig 4-5: TimesNet Series Model Comparison Radar Chart on RuralVoltage Dataset

输出文件: ../chapter4_experiments/fig_4_5_radar_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 导入论文统一样式
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, get_output_dir, TIMESNET_COLORS

# 初始化样式
setup_thesis_style()

# ============================================================
# 实验数据（RuralVoltage 数据集）
# ============================================================
COLORS = {
    'TimesNet': TIMESNET_COLORS[0],         # 柔和蓝
    'VoltageTimesNet': TIMESNET_COLORS[1],   # 柔和绿
    'TPATimesNet': TIMESNET_COLORS[2],       # 柔和紫
    'MTSTimesNet': TIMESNET_COLORS[3],       # 柔和橙
}

RESULTS = {
    'VoltageTimesNet': {'准确率': 0.9393, '精确率': 0.7371, '召回率': 0.9110, 'F1分数': 0.8149},
    'TPATimesNet': {'准确率': 0.8303, '精确率': 0.8983, '召回率': 0.4723, 'F1分数': 0.6191},
    'TimesNet': {'准确率': 0.8259, '精确率': 0.8939, '召回率': 0.4582, 'F1分数': 0.6059},
    'MTSTimesNet': {'准确率': 0.7863, '精确率': 0.8509, '召回率': 0.3251, 'F1分数': 0.4705},
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))

    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 设置径向范围，适配新数据范围
    ax.set_ylim(0.1, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')

    # 绘制顺序：VoltageTimesNet 在最后绘制以突出显示
    models = ['TimesNet', 'MTSTimesNet', 'TPATimesNet', 'VoltageTimesNet']
    linestyles = ['-', '--', '-.', '-']
    markers = ['o', 'd', '^', 'D']
    linewidths = [1.5, 1.5, 1.5, 2.5]
    alphas = [0.05, 0.05, 0.05, 0.15]

    for i, model in enumerate(models):
        values = [RESULTS[model][m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, linestyle=linestyles[i], linewidth=linewidths[i],
                label=model, color=COLORS[model],
                marker=markers[i], markersize=6)
        ax.fill(angles, values, alpha=alphas[i], color=COLORS[model])

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=True)

    # 保存到 output/chap4 目录
    output_dir = get_output_dir(4)
    output_path = os.path.join(output_dir, 'fig_4_5_radar_comparison.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
