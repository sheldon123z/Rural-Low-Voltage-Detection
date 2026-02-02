#!/usr/bin/env python3
"""
图4-5: TimesNet系列模型演进对比雷达图
Fig 4-5: TimesNet Series Model Comparison Radar Chart

输出文件: ../chapter4_experiments/fig_4_5_radar_comparison.png
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

# 柔和科研配色
COLORS = {
    'TimesNet': '#4878A8',         # 柔和蓝
    'VoltageTimesNet': '#5B9BD5',  # 浅蓝
    'VoltageTimesNet_v2': '#72A86D',  # 柔和绿
    'TPATimesNet': '#9B7BB8',      # 柔和紫
}

# 实验数据
RESULTS = {
    'TimesNet': {'准确率': 0.9102, '精确率': 0.7606, '召回率': 0.5705, 'F1分数': 0.6520},
    'VoltageTimesNet': {'准确率': 0.9094, '精确率': 0.7541, '召回率': 0.5726, 'F1分数': 0.6509},
    'VoltageTimesNet_v2': {'准确率': 0.9119, '精确率': 0.7614, '召回率': 0.5858, 'F1分数': 0.6622},
    'TPATimesNet': {'准确率': 0.9090, '精确率': 0.7524, '召回率': 0.5710, 'F1分数': 0.6493},
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))

    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')

    models = ['TimesNet', 'VoltageTimesNet', 'VoltageTimesNet_v2', 'TPATimesNet']
    linestyles = ['-', '--', '-', '-.']
    markers = ['o', 's', 'D', '^']

    for i, model in enumerate(models):
        values = [RESULTS[model][m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2,
                label=model, color=COLORS[model],
                linestyle=linestyles[i], marker=markers[i],
                markersize=6)
        ax.fill(angles, values, alpha=0.1, color=COLORS[model])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, frameon=True)

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_5_radar_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_5_radar_comparison.png")
    plt.close()


if __name__ == '__main__':
    main()
