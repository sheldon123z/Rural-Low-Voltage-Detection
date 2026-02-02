#!/usr/bin/env python3
"""
图4-7: 阈值敏感性分析曲线
Fig 4-7: Threshold Sensitivity Analysis

输出文件: ../chapter4_experiments/fig_4_7_threshold_sensitivity.png
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
    'precision': '#4878A8',  # 柔和蓝
    'recall': '#72A86D',     # 柔和绿
    'f1': '#C4785C',         # 柔和橙
}

# 阈值敏感性数据
THRESHOLD_DATA = {
    'anomaly_ratio': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    'percentile': ['99%', '98.5%', '98%', '97.5%', '97%', '96%', '95%'],
    'Precision': [0.85, 0.82, 0.79, 0.77, 0.76, 0.70, 0.62],
    'Recall': [0.42, 0.48, 0.53, 0.56, 0.59, 0.65, 0.72],
    'F1': [0.56, 0.60, 0.63, 0.65, 0.66, 0.67, 0.67],
}


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(6, 4))

    x = THRESHOLD_DATA['anomaly_ratio']

    # 绘制三条曲线
    ax.plot(x, THRESHOLD_DATA['Precision'], 'o-', color=COLORS['precision'],
            linewidth=2, markersize=7, label='精确率')
    ax.plot(x, THRESHOLD_DATA['Recall'], 's-', color=COLORS['recall'],
            linewidth=2, markersize=7, label='召回率')
    ax.plot(x, THRESHOLD_DATA['F1'], 'D-', color=COLORS['f1'],
            linewidth=3, markersize=8, label='F1分数')

    # 找到最优F1点
    best_idx = np.argmax(THRESHOLD_DATA['F1'])
    best_x = x[best_idx]
    best_f1 = THRESHOLD_DATA['F1'][best_idx]

    # 标注最优点
    ax.scatter([best_x], [best_f1], s=150, c='red', marker='*', zorder=5,
               edgecolors='black', linewidth=1)
    ax.annotate(f'最优F1={best_f1:.2f}\n(ratio={best_x})',
                xy=(best_x, best_f1), xytext=(best_x+0.8, best_f1+0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # 添加次坐标轴显示percentile
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(THRESHOLD_DATA['percentile'], fontsize=9)
    ax2.set_xlabel('百分位阈值', fontsize=10.5, labelpad=8)

    ax.set_xlabel('异常比例/%', fontsize=10.5)
    ax.set_ylabel('性能指标', fontsize=10.5)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.35, 0.95)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_7_threshold_sensitivity.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_7_threshold_sensitivity.png")
    plt.close()


if __name__ == '__main__':
    main()
