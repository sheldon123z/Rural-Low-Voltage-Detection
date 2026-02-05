#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 3-3：VoltageTimesNet 与 TimesNet 对比示意图
展示两种模型在周期发现策略上的差异

输出：单个 PNG 文件
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

from thesis_style import get_output_dir

# 设置中文字体
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = get_output_dir(3)

# 配色方案
COLORS = {
    'timesnet': '#1976D2',       # 蓝色 - TimesNet
    'voltagetimesnet': '#388E3C', # 绿色 - VoltageTimesNet
    'fft': '#F57C00',            # 橙色 - FFT
    'preset': '#7B1FA2',         # 紫色 - 预设周期
    'box_bg': '#FAFAFA',
    'text': '#212121',
    'light_blue': '#E3F2FD',
    'light_green': '#E8F5E9',
}


def draw_comparison():
    """绘制 VoltageTimesNet 与 TimesNet 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    # === 左侧：TimesNet 方案 ===
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # 背景框
    bg1 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['light_blue'], edgecolor=COLORS['timesnet'],
                         linewidth=2, alpha=0.3)
    ax1.add_patch(bg1)

    # 标题
    ax1.text(5, 9.3, 'TimesNet', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['timesnet'])
    ax1.text(5, 8.8, '(FFT 自适应周期发现)', ha='center', va='center',
            fontsize=10, color=COLORS['text'])

    # 输入框
    input_box1 = FancyBboxPatch((3.5, 7.5), 3, 0.8,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor='white', edgecolor='black', linewidth=1.5)
    ax1.add_patch(input_box1)
    ax1.text(5, 7.9, '输入时序 X', ha='center', va='center', fontsize=10, fontweight='bold')

    # FFT 模块
    fft_box = FancyBboxPatch((2.5, 5.5), 5, 1.2,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=COLORS['fft'], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.add_patch(fft_box)
    ax1.text(5, 6.3, 'FFT 频谱分析', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax1.text(5, 5.8, r'$\mathcal{F}(x) \rightarrow$ Top-k 频率', ha='center', va='center',
            fontsize=9, color='white')

    # 箭头
    ax1.annotate('', xy=(5, 6.7), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 周期输出
    period_text = ['$p_1$', '$p_2$', '...', '$p_k$']
    period_colors = ['#1565C0', '#42A5F5', '#90CAF9', '#BBDEFB']
    for i, (p, c) in enumerate(zip(period_text, period_colors)):
        x_pos = 2.5 + i * 1.5
        period_box = FancyBboxPatch((x_pos, 4), 1.2, 0.7,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=c, edgecolor='black', linewidth=1)
        ax1.add_patch(period_box)
        ax1.text(x_pos + 0.6, 4.35, p, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if i < 2 else 'black')

    ax1.annotate('', xy=(5, 4.7), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 特点说明
    features1 = [
        '✓ 自适应发现数据中的周期',
        '✓ 适用于未知周期场景',
        '✗ 可能错过电力系统固有周期',
        '✗ 噪声影响 FFT 精度',
    ]
    for i, feat in enumerate(features1):
        color = COLORS['timesnet'] if '✓' in feat else '#C62828'
        ax1.text(1, 3 - i * 0.6, feat, fontsize=9, color=color)

    # 处理模块
    process_box1 = FancyBboxPatch((3, 1), 4, 0.8,
                                  boxstyle="round,pad=0.03,rounding_size=0.1",
                                  facecolor=COLORS['timesnet'], edgecolor='black',
                                  linewidth=1.5, alpha=0.8)
    ax1.add_patch(process_box1)
    ax1.text(5, 1.4, '2D 卷积处理', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax1.annotate('', xy=(5, 1.8), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # === 右侧：VoltageTimesNet 方案 ===
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # 背景框
    bg2 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['light_green'], edgecolor=COLORS['voltagetimesnet'],
                         linewidth=2, alpha=0.3)
    ax2.add_patch(bg2)

    # 标题
    ax2.text(5, 9.3, 'VoltageTimesNet', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['voltagetimesnet'])
    ax2.text(5, 8.8, '(预设周期 + FFT 混合)', ha='center', va='center',
            fontsize=10, color=COLORS['text'])

    # 输入框
    input_box2 = FancyBboxPatch((3.5, 7.5), 3, 0.8,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor='white', edgecolor='black', linewidth=1.5)
    ax2.add_patch(input_box2)
    ax2.text(5, 7.9, '输入时序 X', ha='center', va='center', fontsize=10, fontweight='bold')

    # 两个分支
    # 左分支：预设周期
    preset_box = FancyBboxPatch((1, 5.5), 3.5, 1.2,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=COLORS['preset'], edgecolor='black',
                                linewidth=1.5, alpha=0.8)
    ax2.add_patch(preset_box)
    ax2.text(2.75, 6.3, '预设电力周期', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax2.text(2.75, 5.8, '1min, 5min, 15min', ha='center', va='center',
            fontsize=9, color='white')

    # 右分支：FFT
    fft_box2 = FancyBboxPatch((5.5, 5.5), 3.5, 1.2,
                              boxstyle="round,pad=0.03,rounding_size=0.1",
                              facecolor=COLORS['fft'], edgecolor='black',
                              linewidth=1.5, alpha=0.8)
    ax2.add_patch(fft_box2)
    ax2.text(7.25, 6.3, 'FFT 补充发现', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax2.text(7.25, 5.8, '数据驱动周期', ha='center', va='center',
            fontsize=9, color='white')

    # 箭头
    ax2.annotate('', xy=(2.75, 6.7), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                              connectionstyle='arc3,rad=0.2'))
    ax2.annotate('', xy=(7.25, 6.7), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                              connectionstyle='arc3,rad=-0.2'))

    # 融合模块
    fusion_box = FancyBboxPatch((2.5, 4), 5, 0.8,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor='#FFF3E0', edgecolor=COLORS['voltagetimesnet'],
                                linewidth=1.5)
    ax2.add_patch(fusion_box)
    ax2.text(5, 4.4, '周期融合 (α 加权)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['voltagetimesnet'])

    ax2.annotate('', xy=(2.75, 4.8), xytext=(2.75, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax2.annotate('', xy=(7.25, 4.8), xytext=(7.25, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 特点说明
    features2 = [
        '✓ 利用电力系统领域知识',
        '✓ 确保关键周期不遗漏',
        '✓ FFT 补充发现新模式',
        '✓ 更高的召回率',
    ]
    for i, feat in enumerate(features2):
        ax2.text(1, 3 - i * 0.6, feat, fontsize=9, color=COLORS['voltagetimesnet'])

    # 处理模块
    process_box2 = FancyBboxPatch((3, 1), 4, 0.8,
                                  boxstyle="round,pad=0.03,rounding_size=0.1",
                                  facecolor=COLORS['voltagetimesnet'], edgecolor='black',
                                  linewidth=1.5, alpha=0.8)
    ax2.add_patch(process_box2)
    ax2.text(5, 1.4, '2D 卷积处理', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax2.annotate('', xy=(5, 1.8), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.tight_layout()

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'fig_3_3_voltage_timesnet_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    draw_comparison()
