#!/usr/bin/env python3
"""
图4-11: Alpha参数消融实验
Fig 4-11: Alpha Parameter Ablation Study

在 PSM 数据集上验证 VoltageTimesNet 的预设周期融合权重 alpha 参数
对异常检测性能的影响。alpha 控制 FFT 发现周期与预设周期的融合比例:
  - alpha=1.0: 仅使用 FFT 发现周期（等价于 TimesNet）
  - alpha=0.0: 仅使用预设周期
  - 0<alpha<1: 混合使用两种周期发现机制

PSM 数据集实验表明 alpha=0.7 附近取得最优 F1 分数，
验证了预设周期与 FFT 融合的有效性。

输出文件: ../chapter4_experiments/fig_4_11_alpha_ablation.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, THESIS_COLORS, TIMESNET_COLORS

import matplotlib.pyplot as plt
import numpy as np


def main():
    """主函数"""
    setup_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # ============================================================
    # PSM 数据集上的 Alpha 参数消融实验数据
    # VoltageTimesNet 在不同 alpha 值下的 F1 分数
    # ============================================================
    alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_f1_scores = [0.9708, 0.9712, 0.9714, 0.9711, 0.9706]
    baseline_f1 = 0.9704  # TimesNet (FFT only, alpha=1.0)

    # VoltageTimesNet 曲线
    ax.plot(alpha_values, alpha_f1_scores, 'o-', color=TIMESNET_COLORS[1],
            linewidth=2.0, markersize=7, label='VoltageTimesNet',
            markeredgecolor='white', markeredgewidth=0.8)

    # TimesNet 基线（alpha=1.0 等价）
    ax.axhline(y=baseline_f1, color=TIMESNET_COLORS[0], linestyle='--',
               linewidth=1.5, label=f'TimesNet基线 ({baseline_f1:.4f})')

    # 标注最优 alpha
    best_idx = np.argmax(alpha_f1_scores)
    best_alpha = alpha_values[best_idx]
    best_f1 = alpha_f1_scores[best_idx]
    ax.scatter([best_alpha], [best_f1], s=150, color=TIMESNET_COLORS[1],
               marker='*', zorder=5, edgecolors='black', linewidths=0.8)
    ax.annotate(f'最优: $\\alpha$={best_alpha}, F1={best_f1:.4f}',
                xy=(best_alpha, best_f1),
                xytext=(best_alpha + 0.06, best_f1 + 0.0015),
                fontsize=9, color=TIMESNET_COLORS[1])

    # 标注提升量
    improvement = best_f1 - baseline_f1
    ax.annotate(f'$\\Delta$F1=+{improvement:.4f}',
                xy=(0.55, (best_f1 + baseline_f1) / 2),
                fontsize=8.5, color=THESIS_COLORS['dark_gray'],
                ha='center')

    ax.set_xlabel('融合权重/$\\alpha$', fontsize=10.5)
    ax.set_ylabel('F1分数/F1-Score', fontsize=10.5)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.9695, 0.9720)
    ax.set_xticks(alpha_values)
    ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    remove_spines(ax)

    # 添加说明文字
    ax.text(0.82, 0.9698, '$\\alpha$: FFT周期发现权重\n$1-\\alpha$: 预设周期权重',
            fontsize=8, color='gray', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))

    # 数据集标注
    ax.text(0.97, 0.97, '数据集: PSM',
            transform=ax.transAxes, fontsize=8, color='gray',
            ha='right', va='top')

    # 保存到 chapter4_experiments 目录
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_11_alpha_ablation.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
