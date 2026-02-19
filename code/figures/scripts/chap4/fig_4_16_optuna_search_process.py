#!/usr/bin/env python3
"""
图 4-16: Optuna 超参数搜索过程图

展示 30-trial TPE 搜索过程中 F1 分数的变化趋势，
标注最优 trial 和累计最优线，体现自动超参数优化的科学性。
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, THESIS_COLORS

setup_thesis_style()

import matplotlib.pyplot as plt


def main():
    # 读取 Optuna 搜索结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', '..', '..', 'results',
                             'optuna', 'full_search_20260203_202801.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    trials = data['all_trials']
    trial_numbers = [t['number'] for t in trials]
    f1_scores = [t['value'] for t in trials]

    # 计算累计最优
    best_so_far = []
    current_best = 0
    for f1 in f1_scores:
        current_best = max(current_best, f1)
        best_so_far.append(current_best)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_trial = trial_numbers[best_idx]

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 所有 trial 的散点
    ax.scatter(trial_numbers, f1_scores, c=THESIS_COLORS['primary'],
               s=40, alpha=0.7, zorder=3, edgecolors='white', linewidths=0.5,
               label='各次搜索 F1')

    # 累计最优线
    ax.plot(trial_numbers, best_so_far, color=THESIS_COLORS['accent'],
            linewidth=1.8, linestyle='-', alpha=0.9, label='累计最优 F1', zorder=2)

    # 标注最优点
    ax.scatter([best_trial], [best_f1], c=THESIS_COLORS['negative'],
               s=120, zorder=5, edgecolors='white', linewidths=1.5, marker='*')
    ax.annotate(f'Trial {best_trial}\nF1={best_f1:.4f}',
                xy=(best_trial, best_f1),
                xytext=(best_trial + 4, best_f1 + 0.02),
                fontsize=9, color=THESIS_COLORS['negative'],
                arrowprops=dict(arrowstyle='->', color=THESIS_COLORS['negative'],
                                lw=1.2),
                ha='left', va='bottom')

    # 基线参考线（默认参数 TimesNet F1=0.5970）
    ax.axhline(y=0.5970, color=THESIS_COLORS['neutral'], linestyle='--',
               linewidth=1, alpha=0.6, label='TimesNet 基线 (F1=0.5970)')

    # 填充搜索区域
    ax.fill_between(trial_numbers, min(f1_scores) - 0.02, f1_scores,
                     alpha=0.08, color=THESIS_COLORS['primary'])

    ax.set_xlabel('搜索轮次/Trial', fontsize=10.5)
    ax.set_ylabel('F1分数/F1-Score', fontsize=10.5)
    ax.set_xlim(-1, 30)
    ax.set_ylim(min(f1_scores) - 0.03, max(f1_scores) + 0.05)
    ax.legend(fontsize=9, frameon=True, edgecolor='#CCCCCC', fancybox=False, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    remove_spines(ax)

    # 保存
    output_dir = os.path.join(script_dir, '..', '..', 'output', 'chap4')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_16_optuna_search_process.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
