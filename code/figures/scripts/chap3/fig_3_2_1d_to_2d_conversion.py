#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 3-2：1D 到 2D 时序转换示意图
展示 TimesNet 如何将 1D 时序信号转换为 2D 张量进行处理

输出：单个 PNG 文件
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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
    'signal': '#1976D2',      # 蓝色 - 原始信号
    'period1': '#388E3C',     # 绿色 - 周期1
    'period2': '#F57C00',     # 橙色 - 周期2
    'period3': '#7B1FA2',     # 紫色 - 周期3
    'grid': '#E0E0E0',        # 网格
    'text': '#212121',        # 文字
    'box_bg': '#FAFAFA',      # 背景
}


def draw_1d_to_2d_conversion():
    """绘制 1D 到 2D 转换示意图"""
    fig = plt.figure(figsize=(14, 8), dpi=300)

    # 创建子图布局
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.3, 1.5],
                          height_ratios=[1, 1], hspace=0.4, wspace=0.1)

    ax_signal = fig.add_subplot(gs[:, 0])  # 左侧：原始1D信号
    ax_2d_top = fig.add_subplot(gs[0, 2])  # 右上：2D张量（周期视图）
    ax_2d_bottom = fig.add_subplot(gs[1, 2])  # 右下：2D张量（时间视图）

    # === 左侧：原始1D时序信号 ===
    np.random.seed(42)
    T = 100  # 序列长度
    t = np.arange(T)

    # 生成带周期性的信号（周期约为20）
    period = 20
    signal = np.sin(2 * np.pi * t / period) + 0.3 * np.sin(4 * np.pi * t / period)
    signal += 0.1 * np.random.randn(T)

    ax_signal.plot(t, signal, color=COLORS['signal'], linewidth=1.5, label='输入序列 $x$')
    ax_signal.fill_between(t, signal, alpha=0.2, color=COLORS['signal'])

    # 标记周期
    for i in range(5):
        start = i * period
        end = start + period
        if end <= T:
            ax_signal.axvspan(start, end, alpha=0.1 if i % 2 == 0 else 0.2,
                            color=COLORS['period1'])
            ax_signal.axvline(x=start, color=COLORS['period1'], linestyle='--',
                            linewidth=0.8, alpha=0.5)

    ax_signal.set_xlabel('时间步/t', fontsize=11)
    ax_signal.set_ylabel('信号值', fontsize=11)
    ax_signal.set_xlim(0, T)
    ax_signal.grid(True, alpha=0.3)
    ax_signal.legend(loc='upper right', fontsize=9)

    # 添加维度标注
    ax_signal.text(T/2, signal.max() + 0.5, '1D 时序信号: [B, T, C]',
                  ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 添加周期标注
    ax_signal.annotate('', xy=(0, signal.min()-0.3), xytext=(period, signal.min()-0.3),
                      arrowprops=dict(arrowstyle='<->', color=COLORS['period1'], lw=1.5))
    ax_signal.text(period/2, signal.min()-0.6, f'周期 p={period}',
                  ha='center', fontsize=9, color=COLORS['period1'])

    # === 右上：2D张量表示（周期×时间段） ===
    n_periods = T // period
    data_2d = signal[:n_periods*period].reshape(n_periods, period)

    im_top = ax_2d_top.imshow(data_2d, aspect='auto', cmap='RdBu_r',
                              interpolation='nearest')
    ax_2d_top.set_xlabel('周期内时间步/n', fontsize=10)
    ax_2d_top.set_ylabel('周期索引/f', fontsize=10)
    ax_2d_top.set_xticks(np.arange(0, period, 5))
    ax_2d_top.set_yticks(np.arange(n_periods))

    # 添加网格
    for i in range(n_periods + 1):
        ax_2d_top.axhline(y=i-0.5, color='white', linewidth=0.5)
    for j in range(period + 1):
        ax_2d_top.axvline(x=j-0.5, color='white', linewidth=0.5)

    ax_2d_top.text(period/2, -1.2, '2D 张量: [B, f, n, C]',
                  ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])

    # === 右下：2D卷积示意 ===
    ax_2d_bottom.imshow(data_2d, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_2d_bottom.set_xlabel('周期内时间步/n', fontsize=10)
    ax_2d_bottom.set_ylabel('周期索引/f', fontsize=10)
    ax_2d_bottom.set_xticks(np.arange(0, period, 5))
    ax_2d_bottom.set_yticks(np.arange(n_periods))

    # 绘制卷积核窗口
    kernel_h, kernel_w = 3, 3
    kernel_y, kernel_x = 1, 8
    rect = Rectangle((kernel_x-0.5, kernel_y-0.5), kernel_w, kernel_h,
                     linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax_2d_bottom.add_patch(rect)
    ax_2d_bottom.text(kernel_x + kernel_w/2 - 0.5, kernel_y + kernel_h + 0.3,
                     '2D 卷积核', ha='center', fontsize=9, color='red', fontweight='bold')

    ax_2d_bottom.text(period/2, -1.2, '2D 卷积处理',
                     ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])

    # === 中间箭头 ===
    fig.text(0.43, 0.55, '→', fontsize=40, ha='center', va='center',
             color=COLORS['period1'], fontweight='bold')
    fig.text(0.43, 0.45, 'Reshape\n(重塑)', fontsize=10, ha='center', va='center',
             color=COLORS['text'])

    # 添加公式说明
    formula_text = r'$x_{1D} \in \mathbb{R}^{T \times C} \rightarrow x_{2D} \in \mathbb{R}^{f \times n \times C}$'
    fig.text(0.43, 0.35, formula_text, fontsize=11, ha='center', va='center',
             color=COLORS['text'])
    fig.text(0.43, 0.28, r'其中 $f = T/p$, $n = p$', fontsize=10, ha='center', va='center',
             color=COLORS['text'])

    plt.tight_layout()

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'fig_3_2_1d_to_2d_conversion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    draw_1d_to_2d_conversion()
