#!/usr/bin/env python3
"""
图4-6: 异常得分分布直方图
Fig 4-6: Anomaly Score Distribution and Classification Regions

输出文件: ../chapter4_experiments/fig_4_6_score_distribution.png
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
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# 色盲友好配色
COLORS = {
    'normal': '#009E73',
    'anomaly': '#D55E00',
    'threshold': '#000000',
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 生成模拟的异常得分分布
    np.random.seed(42)

    # 正常样本得分
    normal_scores = np.concatenate([
        np.random.exponential(0.08, 7000),
        np.random.normal(0.15, 0.05, 1540),
    ])
    normal_scores = np.clip(normal_scores, 0, 0.6)

    # 异常样本得分
    anomaly_scores = np.concatenate([
        np.random.normal(0.35, 0.12, 800),
        np.random.normal(0.22, 0.08, 400),
        np.random.exponential(0.15, 260) + 0.2,
    ])
    anomaly_scores = np.clip(anomaly_scores, 0.05, 1.0)

    # 阈值
    threshold = np.percentile(normal_scores, 97)

    # 绘制直方图
    bins = np.linspace(0, 0.7, 50)

    ax.hist(normal_scores, bins=bins, alpha=0.7, color=COLORS['normal'],
            label='正常样本', edgecolor='white', linewidth=0.5)
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, color=COLORS['anomaly'],
            label='异常样本', edgecolor='white', linewidth=0.5)

    # 阈值线
    ax.axvline(x=threshold, color=COLORS['threshold'], linestyle='--',
               linewidth=2, label=f'阈值 ({threshold:.3f})')

    # 标注区域
    ymax = ax.get_ylim()[1]

    ax.fill_betweenx([0, ymax*0.3], 0, threshold, alpha=0.15, color='green')
    ax.text(threshold/2, ymax*0.85, 'TN', fontsize=12, ha='center', fontweight='bold', color='darkgreen')

    ax.fill_betweenx([0, ymax*0.3], threshold, 0.7, alpha=0.15, color='orange')
    ax.text((threshold+0.7)/2, ymax*0.85, 'FP', fontsize=12, ha='center', fontweight='bold', color='darkorange')

    ax.annotate('FN', xy=(threshold-0.08, ymax*0.4), fontsize=11,
                fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                xytext=(threshold-0.15, ymax*0.6))

    ax.text(0.5, ymax*0.65, 'TP', fontsize=12, ha='center', fontweight='bold', color='darkblue')

    ax.set_xlabel('重构误差', fontsize=10.5)
    ax.set_ylabel('样本数量', fontsize=10.5)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, ymax)

    ax.legend(loc='upper right', fontsize=9, frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_6_score_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_6_score_distribution.png")
    plt.close()


if __name__ == '__main__':
    main()
