#!/usr/bin/env python3
"""
图4-15: MTSTimesNet多尺度贡献分析
Fig 4-15: Multi-scale Contribution Analysis of MTSTimesNet

展示MTSTimesNet模型中不同时间尺度分支对异常检测的贡献:
(a) 各尺度分支的贡献权重饼图
(b) 各尺度对不同异常类型的检测能力热力图

输出文件:
- ../chapter4_experiments/fig_4_15a_scale_weights.png   - 尺度贡献权重饼图
- ../chapter4_experiments/fig_4_15b_scale_detection.png  - 各尺度检测能力热力图

数据集: RuralVoltage realistic_v2
模型: MTSTimesNet (3个并行时间尺度分支)
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir

# ============================================================
# 多尺度分支数据
# ============================================================
# 三个尺度分支
scale_names = ['细粒度\n(seq_len=25)', '中粒度\n(seq_len=50)', '粗粒度\n(seq_len=100)']
scale_weights = [0.35, 0.42, 0.23]  # 贡献权重

# 配色: 蓝, 绿, 橙 (与 thesis_style 一致)
scale_colors = ['#4878A8', '#72A86D', '#C4785C']

# 各尺度对不同异常类型的检测能力 (F1)
anomaly_types = ['电压骤升', '电压骤降', '谐波畸变', '三相不平衡', '缓慢漂移']
# 行: 异常类型, 列: 细粒度/中粒度/粗粒度
detection_f1 = np.array([
    [0.72, 0.58, 0.31],  # 电压骤升
    [0.68, 0.61, 0.35],  # 电压骤降
    [0.45, 0.71, 0.52],  # 谐波畸变
    [0.38, 0.65, 0.63],  # 三相不平衡
    [0.22, 0.48, 0.75],  # 缓慢漂移
])

# 热力图 X 轴标签 (简洁版)
heatmap_xlabels = ['细粒度\n(25)', '中粒度\n(50)', '粗粒度\n(100)']


def plot_scale_weights(output_dir):
    """绘制尺度贡献权重饼图"""
    fig, ax = plt.subplots(figsize=(5, 4))

    # 饼图标签: 名称 + 百分比
    labels = ['细粒度\n(seq_len=25)', '中粒度\n(seq_len=50)', '粗粒度\n(seq_len=100)']
    percentages = [f'{w*100:.0f}%' for w in scale_weights]

    wedges, texts, autotexts = ax.pie(
        scale_weights,
        labels=labels,
        colors=scale_colors,
        autopct='%1.0f%%',
        startangle=90,
        pctdistance=0.6,
        labeldistance=1.15,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=10),
    )

    # 设置百分比文字样式
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax.set_aspect('equal')

    output_path = os.path.join(output_dir, 'fig_4_15a_scale_weights.png')
    save_thesis_figure(fig, output_path)


def plot_scale_detection(output_dir):
    """绘制各尺度检测能力热力图"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制热力图
    im = ax.imshow(detection_f1, cmap='YlGnBu', aspect='auto',
                   vmin=0.15, vmax=0.80)

    # 设置坐标轴
    ax.set_xticks(range(len(heatmap_xlabels)))
    ax.set_xticklabels(heatmap_xlabels, fontsize=10)
    ax.set_yticks(range(len(anomaly_types)))
    ax.set_yticklabels(anomaly_types, fontsize=10)

    # X 轴标签放在上方
    ax.xaxis.set_ticks_position('bottom')

    # 在每个格子中标注 F1 数值
    for i in range(len(anomaly_types)):
        for j in range(len(heatmap_xlabels)):
            value = detection_f1[i, j]
            # 深色背景用白色字, 浅色背景用深色字
            text_color = 'white' if value > 0.60 else '#333333'
            ax.text(j, i, f'{value:.2f}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color=text_color)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label('F1分数', fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    # 移除多余边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path = os.path.join(output_dir, 'fig_4_15b_scale_detection.png')
    save_thesis_figure(fig, output_path)


def main():
    """主函数"""
    setup_thesis_style()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = get_output_dir(4)
    

    plot_scale_weights(output_dir)
    plot_scale_detection(output_dir)


if __name__ == '__main__':
    main()
