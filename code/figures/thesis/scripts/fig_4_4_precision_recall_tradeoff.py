#!/usr/bin/env python3
"""
图4-4: 精确率-召回率权衡散点图
Fig 4-4: Precision-Recall Trade-off Analysis

输出文件: ../chapter4_experiments/fig_4_4_precision_recall_tradeoff.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# ============================================================
# 论文格式配置：中文宋体 + 英文 Times New Roman，五号字 (10.5pt)
# ============================================================
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman']
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# 色盲友好调色板
COLORS_TIMESNET = ['#0072B2', '#56B4E9', '#009E73', '#CC79A7']
COLORS_OTHER = ['#999999', '#666666']

# 实验数据
model_labels = ['TimesNet', 'V-TimesNet', 'V-TimesNet_v2',
                'TPA-TimesNet', 'DLinear', 'PatchTST']
model_colors = [COLORS_TIMESNET[0], COLORS_TIMESNET[1], COLORS_TIMESNET[2],
                COLORS_TIMESNET[3], COLORS_OTHER[0], COLORS_OTHER[1]]

precision = [0.7606, 0.7541, 0.7614, 0.7524, 0.7936, 0.7366]
recall = [0.5705, 0.5726, 0.5858, 0.5710, 0.9837, 0.5735]
f1_score = [0.6520, 0.6509, 0.6622, 0.6493, 0.8785, 0.6449]


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制F1等值线
    recall_line = np.linspace(0.01, 1, 100)
    f1_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    for f1 in f1_values:
        precision_line = (f1 * recall_line) / (2 * recall_line - f1)
        valid_mask = (precision_line > 0) & (precision_line <= 1)
        ax.plot(recall_line[valid_mask], precision_line[valid_mask],
                'k--', alpha=0.3, linewidth=1)
        if f1 == 0.8:
            idx = np.argmin(np.abs(precision_line - 0.85))
            if valid_mask[idx]:
                ax.text(recall_line[idx] + 0.02, precision_line[idx] + 0.02,
                        f'F1={f1}', fontsize=8, color='gray', alpha=0.8)
        elif f1 == 0.6:
            idx = np.argmin(np.abs(precision_line - 0.65))
            if valid_mask[idx]:
                ax.text(recall_line[idx] + 0.02, precision_line[idx] + 0.02,
                        f'F1={f1}', fontsize=8, color='gray', alpha=0.8)

    # 绘制散点
    markers = ['o', 's', '^', 'v', 'D', 'p']
    marker_sizes = [150, 150, 150, 150, 200, 150]

    for i, (model, prec, rec, f1) in enumerate(zip(model_labels, precision, recall, f1_score)):
        ax.scatter(rec, prec, c=[model_colors[i]], s=marker_sizes[i],
                   marker=markers[i], edgecolors='black', linewidth=1.5,
                   label=f'{model} (F1={f1:.3f})', zorder=5)

        # 标注位置调整
        offset_x = 0.015
        offset_y = 0.015

        if model == 'DLinear':
            offset_x = -0.08
            offset_y = 0.01
        elif model == 'V-TimesNet_v2':
            offset_y = -0.025
        elif model == 'TPA-TimesNet':
            offset_y = 0.025
        elif model == 'PatchTST':
            offset_x = -0.06
            offset_y = -0.015

        ax.annotate(model, (rec, prec),
                    xytext=(rec + offset_x, prec + offset_y),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5))

    ax.set_xlabel('召回率', fontsize=10.5)
    ax.set_ylabel('精确率', fontsize=10.5)
    ax.set_xlim([0.5, 1.05])
    ax.set_ylim([0.7, 0.85])

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    ax.text(0.98, 0.72, '虚线为F1等值线',
            fontsize=8, color='gray', ha='right', style='italic')

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_4_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_4_precision_recall_tradeoff.png")
    plt.close()


if __name__ == '__main__':
    main()
