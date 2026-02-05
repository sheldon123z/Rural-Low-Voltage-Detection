#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 3-2：1D 到 2D 时序转换示意图
展示 TimesNet 如何将 1D 时序信号转换为 2D 张量进行处理

优化版本：修复字体重叠、增大字体
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
plt.rcParams['font.size'] = 11
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
    'period1': '#388E3C',     # 绿色 - 周期标记
    'arrow': '#5D6D7E',       # 箭头
    'text': '#212121',        # 文字
    'highlight': '#C62828',   # 高亮（卷积核）
}


def draw_1d_to_2d_conversion():
    """绘制 1D 到 2D 转换示意图"""
    fig = plt.figure(figsize=(12, 5), dpi=300)

    # 使用更简单的布局：3列
    # 左：1D信号 | 中：箭头和说明 | 右：2D张量
    ax_signal = fig.add_axes([0.05, 0.15, 0.35, 0.75])  # 左侧1D信号
    ax_2d = fig.add_axes([0.58, 0.15, 0.38, 0.75])      # 右侧2D张量

    # === 生成信号数据 ===
    np.random.seed(42)
    T = 100  # 序列长度
    t = np.arange(T)
    period = 20

    # 生成带周期性的信号
    signal = np.sin(2 * np.pi * t / period) + 0.3 * np.sin(4 * np.pi * t / period)
    signal += 0.1 * np.random.randn(T)

    # === 左侧：原始1D时序信号 ===
    ax_signal.plot(t, signal, color=COLORS['signal'], linewidth=1.8, label='输入序列 $x$')
    ax_signal.fill_between(t, signal, alpha=0.15, color=COLORS['signal'])

    # 标记周期区间（交替颜色）
    for i in range(5):
        start = i * period
        end = start + period
        if end <= T:
            ax_signal.axvspan(start, end, alpha=0.08 if i % 2 == 0 else 0.15,
                            color=COLORS['period1'])

    ax_signal.set_xlabel('时间步 $t$', fontsize=12)
    ax_signal.set_ylabel('信号值', fontsize=12)
    ax_signal.set_xlim(0, T)
    ax_signal.set_ylim(signal.min() - 0.3, signal.max() + 0.6)
    ax_signal.grid(True, alpha=0.3, linestyle='--')
    ax_signal.legend(loc='upper right', fontsize=10)

    # 顶部标注
    ax_signal.set_title('1D 时序信号: $[B, T, C]$', fontsize=13, pad=10)

    # === 右侧：2D张量表示 ===
    n_periods = T // period
    data_2d = signal[:n_periods * period].reshape(n_periods, period)

    im = ax_2d.imshow(data_2d, aspect='auto', cmap='RdBu_r', interpolation='nearest')

    ax_2d.set_xlabel('周期内时间步 $n$', fontsize=12)
    ax_2d.set_ylabel('周期索引 $f$', fontsize=12)
    ax_2d.set_xticks(np.arange(0, period, 5))
    ax_2d.set_yticks(np.arange(n_periods))

    # 添加白色网格线
    for i in range(n_periods + 1):
        ax_2d.axhline(y=i - 0.5, color='white', linewidth=0.8)
    for j in range(0, period + 1, 5):
        ax_2d.axvline(x=j - 0.5, color='white', linewidth=0.8)

    # 绘制2D卷积核窗口
    kernel_h, kernel_w = 2, 5
    kernel_y, kernel_x = 1.5, 7
    rect = Rectangle((kernel_x - 0.5, kernel_y - 0.5), kernel_w, kernel_h,
                     linewidth=2.5, edgecolor=COLORS['highlight'],
                     facecolor='none', linestyle='-')
    ax_2d.add_patch(rect)
    ax_2d.text(kernel_x + kernel_w/2 - 0.5, kernel_y + kernel_h + 0.2,
               '2D卷积核', ha='center', fontsize=10,
               color=COLORS['highlight'], fontweight='bold')

    # 顶部标注
    ax_2d.set_title('2D 张量: $[B, f, n, C]$', fontsize=13, pad=10)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax_2d, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    # === 中间箭头和说明（使用fig坐标） ===
    # 大箭头
    fig.text(0.47, 0.55, '→', fontsize=50, ha='center', va='center',
             color=COLORS['period1'], fontweight='bold')

    # Reshape 说明
    fig.text(0.47, 0.42, 'Reshape', fontsize=12, ha='center', va='center',
             color=COLORS['text'], fontweight='bold')

    # 公式（在箭头下方）
    formula = r'$x \in \mathbb{R}^{T \times C} \rightarrow \mathbb{R}^{f \times n \times C}$'
    fig.text(0.47, 0.28, formula, fontsize=11, ha='center', va='center',
             color=COLORS['text'])

    # 说明文字
    fig.text(0.47, 0.18, r'$f = T/p$,  $n = p$', fontsize=10, ha='center', va='center',
             color=COLORS['text'], style='italic')

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'fig_3_2_1d_to_2d_conversion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    draw_1d_to_2d_conversion()
