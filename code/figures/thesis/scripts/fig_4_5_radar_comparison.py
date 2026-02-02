#!/usr/bin/env python3
"""
图4-5: TimesNet系列模型演进对比雷达图
Fig 4-5: TimesNet Series Model Comparison Radar Chart

输出文件: fig_4_5_radar_comparison.pdf, fig_4_5_radar_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 论文格式配置
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# 色盲友好配色
COLORS = {
    'TimesNet': '#0072B2',
    'VoltageTimesNet': '#009E73',
    'VoltageTimesNet_v2': '#D55E00',
    'TPATimesNet': '#CC79A7',
}

# 实验数据
RESULTS = {
    'TimesNet': {'Accuracy': 0.9102, 'Precision': 0.7606, 'Recall': 0.5705, 'F1': 0.6520},
    'VoltageTimesNet': {'Accuracy': 0.9094, 'Precision': 0.7541, 'Recall': 0.5726, 'F1': 0.6509},
    'VoltageTimesNet_v2': {'Accuracy': 0.9119, 'Precision': 0.7614, 'Recall': 0.5858, 'F1': 0.6622},
    'TPATimesNet': {'Accuracy': 0.9090, 'Precision': 0.7524, 'Recall': 0.5710, 'F1': 0.6493},
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')

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

    # 保存
    plt.savefig('fig_4_5_radar_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('fig_4_5_radar_comparison.png', dpi=300, bbox_inches='tight')
    print("已生成: fig_4_5_radar_comparison.pdf/png")
    plt.close()


if __name__ == '__main__':
    main()
