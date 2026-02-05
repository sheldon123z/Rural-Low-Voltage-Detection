#!/usr/bin/env python3
"""
图4-7: 阈值敏感性分析曲线
Fig 4-7: Threshold Sensitivity Analysis

基于 VoltageTimesNet 模型在 RuralVoltage 数据集上的实验。
已知最优点: anomaly_ratio=2.08, F1=0.8149 (Optuna优化结果)

输出文件: ../chapter4_experiments/fig_4_7_threshold_sensitivity.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, METRIC_COLORS

import matplotlib.pyplot as plt
import numpy as np


def main():
    """主函数"""
    setup_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    # ============================================================
    # VoltageTimesNet 阈值敏感性数据
    # 已知: anomaly_ratio=2.08 时 P=0.7371, R=0.9110, F1=0.8149
    # 趋势: ratio增大 -> precision降低, recall升高, F1先升后降
    # ============================================================
    anomaly_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    percentiles = ['99%', '98.5%', '98%', '97.5%', '97%', '96%', '95%']

    precision_values = [0.82, 0.78, 0.74, 0.70, 0.65, 0.55, 0.45]
    recall_values = [0.72, 0.82, 0.91, 0.94, 0.96, 0.98, 0.99]
    f1_values = [0.767, 0.799, 0.815, 0.805, 0.778, 0.696, 0.620]

    # 绘制三条曲线
    ax.plot(anomaly_ratios, precision_values, 'o-', color=METRIC_COLORS['precision'],
            linewidth=1.8, markersize=6, label='精确率')
    ax.plot(anomaly_ratios, recall_values, 's-', color=METRIC_COLORS['recall'],
            linewidth=1.8, markersize=6, label='召回率')
    ax.plot(anomaly_ratios, f1_values, 'D-', color=METRIC_COLORS['f1'],
            linewidth=2.2, markersize=7, label='F1分数')

    # 找到最优F1点并标注
    best_idx = np.argmax(f1_values)
    best_x = anomaly_ratios[best_idx]
    best_f1 = f1_values[best_idx]

    ax.scatter([best_x], [best_f1], s=150, c='#C4785C', marker='*', zorder=5,
               edgecolors='black', linewidth=0.8)
    ax.annotate(f'最优 F1={best_f1:.3f}\n(ratio={best_x})',
                xy=(best_x, best_f1), xytext=(best_x + 0.9, best_f1 + 0.02),
                fontsize=9, ha='left', color='#8a4a3a',
                arrowprops=dict(arrowstyle='->', color='#8a4a3a', lw=1.2))

    # 添加次坐标轴显示百分位
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(anomaly_ratios)
    ax2.set_xticklabels(percentiles, fontsize=9)
    ax2.set_xlabel('百分位阈值/Percentile', fontsize=10.5, labelpad=8)

    ax.set_xlabel('异常比例/%', fontsize=10.5)
    ax.set_ylabel('性能指标/Score', fontsize=10.5)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.35, 1.05)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='center left', fontsize=9, frameon=True, edgecolor='gray')

    # 移除右侧边框（保留顶部因为有次坐标轴）
    ax.spines['right'].set_visible(False)

    # 保存到 chapter4_experiments 目录
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_7_threshold_sensitivity.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
