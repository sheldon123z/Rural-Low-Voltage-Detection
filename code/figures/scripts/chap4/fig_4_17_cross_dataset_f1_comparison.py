#!/usr/bin/env python3
"""
图 4-17: 跨数据集 F1 分数分组柱状图

展示 TimesNet 和 VoltageTimesNet 在多个公开数据集及 RuralVoltage 上的 F1 表现，
体现模型泛化能力及 VoltageTimesNet 在电压数据上的优势。
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, THESIS_COLORS

setup_thesis_style()

import matplotlib.pyplot as plt


def main():
    # 实验结果数据
    datasets = ['PSM', 'SMD', 'MSL', 'SMAP', 'RuralVoltage']
    datasets_zh = ['PSM\n(服务器)', 'SMD\n(服务器集群)', 'MSL\n(航天器)', 'SMAP\n(航天器)', 'RuralVoltage\n(农村电压)']

    # F1 scores for each model on each dataset
    # TimesNet results
    timesnet_f1 = [0.9735, 0.8246, 0.7636, 0.6865, 0.5970]

    # VoltageTimesNet (论文中即 VoltageTimesNet_v2 Optuna 优化版)
    # PSM 上 VoltageTimesNet 接近 TimesNet, 其他数据集仅 TimesNet 有结果
    voltagetimesnet_f1 = [0.9731, None, None, None, 0.8149]

    # LSTMAutoEncoder (仅 RuralVoltage 有)
    lstmae_f1 = [None, None, None, None, 0.4457]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(datasets))
    width = 0.25

    # TimesNet 柱状图（所有数据集都有）
    bars1 = ax.bar(x - width, timesnet_f1, width, label='TimesNet',
                   color=THESIS_COLORS['primary'], edgecolor='white', linewidth=0.5, zorder=3)

    # VoltageTimesNet 柱状图（仅 PSM 和 RuralVoltage）
    vtn_values = []
    vtn_positions = []
    for i, v in enumerate(voltagetimesnet_f1):
        if v is not None:
            vtn_values.append(v)
            vtn_positions.append(x[i])
    bars2 = ax.bar(np.array(vtn_positions), vtn_values, width,
                   label='VoltageTimesNet',
                   color=THESIS_COLORS['secondary'], edgecolor='white', linewidth=0.5, zorder=3)

    # LSTMAutoEncoder（仅 RuralVoltage）
    lstmae_values = []
    lstmae_positions = []
    for i, v in enumerate(lstmae_f1):
        if v is not None:
            lstmae_values.append(v)
            lstmae_positions.append(x[i] + width)
    bars3 = ax.bar(np.array(lstmae_positions), lstmae_values, width,
                   label='LSTMAutoEncoder',
                   color=THESIS_COLORS['warning'], edgecolor='white', linewidth=0.5, zorder=3)

    # 在柱顶添加数值标注
    for bar_group in [bars1, bars2, bars3]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 添加 RuralVoltage 提升标注
    if voltagetimesnet_f1[4] is not None:
        improvement = (voltagetimesnet_f1[4] - timesnet_f1[4]) / timesnet_f1[4] * 100
        mid_x = x[4]
        mid_y = (timesnet_f1[4] + voltagetimesnet_f1[4]) / 2
        ax.annotate(f'+{improvement:.1f}%',
                    xy=(mid_x, voltagetimesnet_f1[4] + 0.01),
                    xytext=(mid_x + 0.6, mid_y + 0.12),
                    fontsize=9, color=THESIS_COLORS['negative'],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=THESIS_COLORS['negative'], lw=1.2),
                    ha='center')

    ax.set_xlabel('数据集/Dataset', fontsize=10.5)
    ax.set_ylabel('F1分数/F1-Score', fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_zh, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, frameon=True, edgecolor='#CCCCCC', fancybox=False,
              loc='upper left', ncol=3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    remove_spines(ax)

    # 保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', '..', 'output', 'chap4')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_17_cross_dataset_f1_comparison.png')
    save_thesis_figure(fig, output_path)

if __name__ == '__main__':
    main()
