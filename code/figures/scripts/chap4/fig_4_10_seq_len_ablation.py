#!/usr/bin/env python3
"""
图4-10: 序列长度消融实验
Fig 4-10: Sequence Length Ablation Study

基于 Optuna 30-trial 超参数搜索结果，展示 VoltageTimesNet 在不同
序列长度下的最优 F1 分数。seq_len=50 为最优。

数据来源:
- seq_len=50:  best F1=0.8149 (Trial 10), 共20次试验
- seq_len=100: best F1=0.6750 (Trial 14), 共5次试验
- seq_len=200: best F1=0.6623 (Trial 2),  共5次试验

输出文件: ../chapter4_experiments/fig_4_10_seq_len_ablation.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, TIMESNET_COLORS

import matplotlib.pyplot as plt
import numpy as np


def main():
    """主函数"""
    setup_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # ============================================================
    # VoltageTimesNet 在 RuralVoltage 上的序列长度消融数据
    # 来自 Optuna 30-trial 搜索
    # ============================================================
    seq_lens = [50, 100, 200]
    best_f1_scores = [0.8149, 0.6750, 0.6623]

    # 绘制 VoltageTimesNet 曲线
    ax.plot(seq_lens, best_f1_scores, 'o-', color=TIMESNET_COLORS[1],
            linewidth=2.0, markersize=8, label='VoltageTimesNet',
            markeredgecolor='white', markeredgewidth=0.8)

    # 标注最优点
    best_idx = np.argmax(best_f1_scores)
    best_seq = seq_lens[best_idx]
    best_f1 = best_f1_scores[best_idx]

    ax.scatter([best_seq], [best_f1], s=150, color=TIMESNET_COLORS[1],
               marker='*', zorder=5, edgecolors='black', linewidths=0.8)
    ax.annotate(f'最优: seq_len={best_seq}\nF1={best_f1:.4f}',
                xy=(best_seq, best_f1),
                xytext=(best_seq + 30, best_f1 - 0.03),
                fontsize=9, color=TIMESNET_COLORS[1],
                arrowprops=dict(arrowstyle='->', color=TIMESNET_COLORS[1], lw=1.0))

    # 标注其他数据点的值
    for i, (sl, f1) in enumerate(zip(seq_lens, best_f1_scores)):
        if i != best_idx:
            ax.annotate(f'{f1:.4f}',
                        xy=(sl, f1),
                        xytext=(sl + 10, f1 + 0.015),
                        fontsize=8.5, color=TIMESNET_COLORS[1])

    # 添加试验次数标注
    trial_counts = [20, 5, 5]
    for sl, f1, tc in zip(seq_lens, best_f1_scores, trial_counts):
        ax.text(sl, f1 - 0.025, f'({tc}次试验)',
                fontsize=7.5, ha='center', color='gray')

    ax.set_xlabel('序列长度/步', fontsize=10.5)
    ax.set_ylabel('最优F1分数/F1-Score', fontsize=10.5)
    ax.set_xlim(20, 230)
    ax.set_ylim(0.60, 0.88)
    ax.set_xticks(seq_lens)
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    remove_spines(ax)

    # 保存到 chapter4_experiments 目录
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_10_seq_len_ablation.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
