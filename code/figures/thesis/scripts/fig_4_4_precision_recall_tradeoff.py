#!/usr/bin/env python3
"""
图4-4: 精确率-召回率权衡散点图
Fig 4-4: Precision-Recall Trade-off Analysis

- 散点图: X轴 Recall, Y轴 Precision
- F1 等值线 (0.4, 0.5, 0.6, 0.7, 0.8)
- VoltageTimesNet 用星形标记突出

数据集: RuralVoltage realistic_v2
测试集: 10000 样本, 异常比例 14.6% (1460 异常, 8540 正常)

输出文件: ../chapter4_experiments/fig_4_4_precision_recall_tradeoff.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 导入论文统一样式
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines

setup_thesis_style()

# ============================================================
# RuralVoltage 实验数据 (5个模型)
# ============================================================
models_data = {
    'VoltageTimesNet':    {'accuracy': 0.9393, 'precision': 0.7371, 'recall': 0.9110, 'f1': 0.8149},
    'TimesNet':           {'accuracy': 0.8584, 'precision': 0.5143, 'recall': 0.7115, 'f1': 0.5970},
    'LSTMAutoEncoder':    {'accuracy': 0.7905, 'precision': 0.3654, 'recall': 0.5712, 'f1': 0.4457},
    'Isolation Forest':   {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
    'One-Class SVM':      {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
}

# 模型显示顺序与标签
model_keys = ['VoltageTimesNet', 'TimesNet',
              'LSTMAutoEncoder', 'Isolation Forest', 'One-Class SVM']
model_labels = ['VoltageTimesNet', 'TimesNet',
                'LSTM-AE', 'iForest', 'OC-SVM']

# 柔和科研配色
model_colors = [
    '#72A86D',   # VoltageTimesNet - 柔和绿
    '#4878A8',   # TimesNet - 柔和蓝
    '#B8860B',   # LSTMAutoEncoder - 暗金
    '#8B4513',   # Isolation Forest - 棕色
    '#CD853F',   # One-Class SVM - 秘鲁色
]

# 标记样式: 主模型用星形(*)，其余用不同形状
model_markers = ['*', 'o', 's', '^', 'v']
model_sizes = [300, 150, 150, 150, 150]  # 主模型更大


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # ----------------------------------------------------------
    # 绘制 F1 等值线
    # F1 = 2*P*R / (P+R)  =>  P = F1*R / (2*R - F1)
    # ----------------------------------------------------------
    recall_line = np.linspace(0.01, 1.0, 300)
    f1_values = [0.4, 0.5, 0.6, 0.7, 0.8]

    for f1 in f1_values:
        precision_line = (f1 * recall_line) / (2 * recall_line - f1)
        valid_mask = (precision_line > 0) & (precision_line <= 1.0)
        ax.plot(recall_line[valid_mask], precision_line[valid_mask],
                'k--', alpha=0.25, linewidth=0.8)

        # 标注 F1 值: 在曲线右侧标注
        # 找到 recall=0.98 附近的位置
        target_r = 0.98
        idx = np.argmin(np.abs(recall_line - target_r))
        if valid_mask[idx] and 0.15 < precision_line[idx] < 0.95:
            ax.text(recall_line[idx] + 0.01, precision_line[idx],
                    f'F1={f1}', fontsize=7.5, color='gray', alpha=0.7,
                    va='center')

    # ----------------------------------------------------------
    # 绘制各模型散点
    # ----------------------------------------------------------
    for i, key in enumerate(model_keys):
        d = models_data[key]
        ax.scatter(d['recall'], d['precision'],
                   c=[model_colors[i]], s=model_sizes[i],
                   marker=model_markers[i],
                   edgecolors='black', linewidth=1.5,
                   label=f'{model_labels[i]} (F1={d["f1"]:.3f})',
                   zorder=5)

    # ----------------------------------------------------------
    # 标注模型名称（手动微调偏移避免重叠）
    # ----------------------------------------------------------
    offsets = {
        'VoltageTimesNet':    (-0.06, 0.03),    # 左上
        'TimesNet':           (0.02, 0.02),     # 右上
        'LSTMAutoEncoder':    (0.02, 0.02),     # 右上
        'Isolation Forest':   (-0.15, 0.02),    # 左上
        'One-Class SVM':      (-0.13, -0.03),   # 左下
    }

    for i, key in enumerate(model_keys):
        d = models_data[key]
        ox, oy = offsets[key]
        ax.annotate(model_labels[i],
                    (d['recall'], d['precision']),
                    xytext=(d['recall'] + ox, d['precision'] + oy),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray',
                                    alpha=0.5, lw=0.5))

    # ----------------------------------------------------------
    # 坐标轴与网格
    # ----------------------------------------------------------
    ax.set_xlabel('召回率/Recall', fontsize=10.5)
    ax.set_ylabel('精确率/Precision', fontsize=10.5)

    # 根据数据范围调整轴范围
    # Recall: 0.5712 ~ 1.0, Precision: 0.3474 ~ 0.7371
    ax.set_xlim([0.45, 1.08])
    ax.set_ylim([0.25, 0.85])

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # 注释说明
    ax.text(0.98, 0.03, '虚线为F1等值线',
            transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom',
            style='italic')

    remove_spines(ax)

    # 保存到 chapter4_experiments 目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'chapter4_experiments')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_4_precision_recall_tradeoff.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
