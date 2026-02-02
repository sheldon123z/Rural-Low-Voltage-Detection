#!/usr/bin/env python3
"""
图4-10: 序列长度消融实验
Fig 4-10: Sequence Length Ablation Study

输出文件: ../chapter4_experiments/fig_4_10_seq_len_ablation.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

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

# 序列长度消融实验数据
timesnet_seq_lens = [50, 100, 200, 500]
timesnet_f1_scores = [0.8038, 0.7634, 0.7985, 0.8659]

voltage_seq_lens = [100, 360, 720]
voltage_f1_scores = [0.7661, 0.8367, 0.8550]


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 柔和科研配色
    ax.plot(timesnet_seq_lens, timesnet_f1_scores, 'o-', color='#4878A8',
            linewidth=1.5, markersize=6, label='TimesNet')

    ax.plot(voltage_seq_lens, voltage_f1_scores, 's-', color='#72A86D',
            linewidth=1.5, markersize=6, label='VoltageTimesNet')

    # 标注最优点
    best_timesnet_idx = np.argmax(timesnet_f1_scores)
    ax.annotate(f'最优: {timesnet_f1_scores[best_timesnet_idx]:.4f}',
                xy=(timesnet_seq_lens[best_timesnet_idx], timesnet_f1_scores[best_timesnet_idx]),
                xytext=(timesnet_seq_lens[best_timesnet_idx]-80, timesnet_f1_scores[best_timesnet_idx]+0.02),
                fontsize=9, color='#4878A8',
                arrowprops=dict(arrowstyle='->', color='#4878A8', lw=0.8))

    best_voltage_idx = np.argmax(voltage_f1_scores)
    ax.annotate(f'最优: {voltage_f1_scores[best_voltage_idx]:.4f}',
                xy=(voltage_seq_lens[best_voltage_idx], voltage_f1_scores[best_voltage_idx]),
                xytext=(voltage_seq_lens[best_voltage_idx]-150, voltage_f1_scores[best_voltage_idx]-0.03),
                fontsize=9, color='#72A86D',
                arrowprops=dict(arrowstyle='->', color='#72A86D', lw=0.8))

    ax.set_xlabel('序列长度/步', fontsize=10.5)
    ax.set_ylabel('F1分数', fontsize=10.5)
    ax.set_xlim(0, 800)
    ax.set_ylim(0.74, 0.90)
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_10_seq_len_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_10_seq_len_ablation.png")
    plt.close()


if __name__ == '__main__':
    main()
