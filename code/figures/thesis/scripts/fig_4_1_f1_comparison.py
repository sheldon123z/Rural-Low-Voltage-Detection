#!/usr/bin/env python3
"""
图4-1: 多模型F1分数对比柱状图
Fig 4-1: F1-score Comparison of Multiple Models

输出文件: fig_4_1_f1_comparison.pdf, fig_4_1_f1_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.patches as mpatches

# 论文格式配置
matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

# 色盲友好调色板 (Okabe-Ito)
COLORS_TIMESNET = ['#0072B2', '#56B4E9', '#009E73', '#CC79A7']
COLORS_OTHER = ['#999999', '#666666']

# 实验数据
models = ['TimesNet', 'VoltageTimesNet', 'VoltageTimesNet_v2',
          'TPATimesNet', 'DLinear', 'PatchTST']
f1_score = [0.6520, 0.6509, 0.6622, 0.6493, 0.8785, 0.6449]

# 模型简称
model_labels = ['TimesNet', 'V-TimesNet', 'V-TimesNet_v2',
                'TPA-TimesNet', 'DLinear', 'PatchTST']

# 模型颜色分配
model_colors = [COLORS_TIMESNET[0], COLORS_TIMESNET[1], COLORS_TIMESNET[2],
                COLORS_TIMESNET[3], COLORS_OTHER[0], COLORS_OTHER[1]]


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, f1_score, width=0.6, color=model_colors, edgecolor='black', linewidth=0.8)

    # 在柱子上标注数值
    for bar, score in zip(bars, f1_score):
        height = bar.get_height()
        ax.annotate(f'{score:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # 设置坐标轴
    ax.set_xlabel('模型 (Model)', fontsize=11)
    ax.set_ylabel('F1分数 (F1-score)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.set_axisbelow(True)

    # 添加图例说明颜色含义
    legend_elements = [
        mpatches.Patch(facecolor=COLORS_TIMESNET[0], edgecolor='black',
                       label='TimesNet系列 (TimesNet Family)'),
        mpatches.Patch(facecolor=COLORS_OTHER[0], edgecolor='black',
                       label='其他模型 (Other Models)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # 三线表风格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    plt.savefig('fig_4_1_f1_comparison.pdf', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig_4_1_f1_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("已生成: fig_4_1_f1_comparison.pdf/png")
    plt.close()


if __name__ == '__main__':
    main()
