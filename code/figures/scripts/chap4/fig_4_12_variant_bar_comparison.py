#!/usr/bin/env python3
"""
图4-12: TimesNet变体性能对比柱状图（RuralVoltage 数据集）
Fig 4-12: TimesNet Variant Performance Comparison Bar Chart on RuralVoltage Dataset

输出文件: ../chapter4_experiments/fig_4_12_variant_bar_comparison.png
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib.pyplot as plt
import numpy as np

# 导入论文统一样式
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, TIMESNET_COLORS

# 初始化样式
setup_thesis_style()

# ============================================================
# 实验数据（RuralVoltage 数据集）
# ============================================================
MODELS = ['VoltageTimesNet', 'TPATimesNet', 'TimesNet', 'MTSTimesNet']
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1']

RESULTS = {
    'VoltageTimesNet': [0.9393, 0.7371, 0.9110, 0.8149],
    'TPATimesNet':     [0.8303, 0.8983, 0.4723, 0.6191],
    'TimesNet':        [0.8259, 0.8939, 0.4582, 0.6059],
    'MTSTimesNet':     [0.7863, 0.8509, 0.3251, 0.4705],
}

COLORS = {
    'VoltageTimesNet': TIMESNET_COLORS[1],   # 柔和绿
    'TPATimesNet':     TIMESNET_COLORS[2],   # 柔和紫
    'TimesNet':        TIMESNET_COLORS[0],   # 柔和蓝
    'MTSTimesNet':     TIMESNET_COLORS[3],   # 柔和橙
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    num_metrics = len(METRICS)
    num_models = len(MODELS)
    bar_width = 0.18
    x = np.arange(num_metrics)

    for i, model in enumerate(MODELS):
        offset = (i - (num_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, RESULTS[model], bar_width,
                      label=model, color=COLORS[model], edgecolor='white',
                      linewidth=0.5)

        # 在柱子顶部标注数值
        for bar, val in zip(bars, RESULTS[model]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('指标值/Value', fontsize=10.5)

    remove_spines(ax)
    ax.legend(loc='upper right', fontsize=9, frameon=True)

    # 保存到 chapter4_experiments 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_12_variant_bar_comparison.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
