#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT 周期发现模块流程图
用 matplotlib 绘制，300 DPI，无内部标题

输出: fig_fft_period_discovery.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import THESIS_COLORS

# 字体
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

# 配色
C = {
    'input_bg': '#E8EFF7',
    'input_border': '#4878A8',
    'fft_bg': '#EBF3EB',
    'fft_border': '#72A86D',
    'spectrum_bg': '#FFF3E6',
    'spectrum_border': '#C4785C',
    'period_bg': '#F0E8F5',
    'period_border': '#9B7BB8',
    'output_bg': '#FFF8E6',
    'output_border': '#D4A84C',
    'arrow': '#505050',
    'text': '#333333',
    'formula': '#505050',
    'white': '#FFFFFF',
    'step_circle': '#4878A8',
}


def draw_box(ax, x, y, w, h, bg, border, text, fontsize=10, bold=False):
    """绘制圆角矩形"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(box)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fontsize, fontweight=fw,
            ha='center', va='center', color=C['text'])


def draw_sub_box(ax, x, y, w, h, text, fontsize=9):
    """绘制子步骤框"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=C['white'], edgecolor='#CCCCCC', linewidth=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize,
            ha='center', va='center', color=C['text'])


def draw_arrow(ax, x1, y1, x2, y2):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', color=C['arrow'],
                            lw=1.5, mutation_scale=15,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


def draw_step_label(ax, x, y, num):
    """绘制步骤编号圆圈"""
    circle = plt.Circle((x, y), 0.22, facecolor=C['step_circle'],
                        edgecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ===== 布局参数 =====
    # 行: 主流程 y=3.2, 子步骤上排 y=4.5/4.8, 子步骤下排 y=1.5/1.8
    main_y = 3.2
    box_w = 2.2
    box_h = 1.0

    # ===== 1. 输入时序 =====
    ix = 1.2
    draw_box(ax, ix, main_y, 1.8, 1.0,
             C['input_bg'], C['input_border'],
             '输入时序\n$\\mathbf{x} \\in \\mathbb{R}^{T}$', fontsize=10, bold=True)

    # ===== 2. FFT 变换模块 =====
    fft_x = 4.3
    # 大背景框
    fft_bg = FancyBboxPatch((fft_x - 1.4, main_y - 1.8), 2.8, 3.6,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=C['fft_bg'], edgecolor=C['fft_border'],
                            linewidth=1.5, alpha=0.4)
    ax.add_patch(fft_bg)
    draw_step_label(ax, fft_x, main_y + 2.05, 1)
    ax.text(fft_x, main_y + 1.65, '快速傅里叶变换', fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=C['fft_border'])

    # 子步骤
    draw_sub_box(ax, fft_x, main_y + 0.85, 2.3, 0.55,
                 '$X_k = \\sum_{n} x_n \\, e^{-j2\\pi kn/N}$', fontsize=9)
    draw_sub_box(ax, fft_x, main_y + 0.0, 2.3, 0.55,
                 '幅值: $|X_k| = \\sqrt{\\mathrm{Re}^2 + \\mathrm{Im}^2}$', fontsize=9)
    draw_sub_box(ax, fft_x, main_y - 0.85, 2.3, 0.55,
                 '通道平均: $A_k = \\mathrm{mean}(|X_k|)$', fontsize=9)

    # 小箭头连接子步骤
    for dy in [0.55, -0.3]:
        ax.annotate('', xy=(fft_x, main_y + dy - 0.02),
                    xytext=(fft_x, main_y + dy + 0.23),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

    # ===== 3. 频谱分析模块 =====
    spec_x = 7.5
    spec_bg = FancyBboxPatch((spec_x - 1.3, main_y - 1.8), 2.6, 3.6,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=C['spectrum_bg'], edgecolor=C['spectrum_border'],
                             linewidth=1.5, alpha=0.4)
    ax.add_patch(spec_bg)
    draw_step_label(ax, spec_x, main_y + 2.05, 2)
    ax.text(spec_x, main_y + 1.65, '频谱分析', fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=C['spectrum_border'])

    # 频谱示意 - 简单柱状图
    bar_y_base = main_y + 0.2
    bar_x_start = spec_x - 0.9
    bar_heights = [0.8, 0.5, 0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    bar_w = 0.18
    for i, bh in enumerate(bar_heights):
        bx = bar_x_start + i * 0.23
        color = C['spectrum_border'] if i < 3 else '#D4C4B0'
        rect = plt.Rectangle((bx, bar_y_base), bar_w, bh,
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
    ax.text(spec_x, bar_y_base - 0.15, '频率', fontsize=8,
            ha='center', va='top', color='#888888')

    # 峰值检测
    draw_sub_box(ax, spec_x, main_y - 0.9, 2.2, 0.65,
                 '峰值检测\n选取 Top-$k$ 个频率', fontsize=9)

    # ===== 4. 周期转换模块 =====
    per_x = 10.8
    per_bg = FancyBboxPatch((per_x - 1.3, main_y - 1.8), 2.6, 3.6,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=C['period_bg'], edgecolor=C['period_border'],
                            linewidth=1.5, alpha=0.4)
    ax.add_patch(per_bg)
    draw_step_label(ax, per_x, main_y + 2.05, 3)
    ax.text(per_x, main_y + 1.65, '周期转换', fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=C['period_border'])

    draw_sub_box(ax, per_x, main_y + 0.7, 2.2, 0.55,
                 '周期: $p_k = T / f_k$', fontsize=9)
    draw_sub_box(ax, per_x, main_y - 0.1, 2.2, 0.55,
                 '权重: $w_k = \\mathrm{softmax}(A_k)$', fontsize=9)
    draw_sub_box(ax, per_x, main_y - 0.9, 2.2, 0.55,
                 '输出: $\\{(p_k, w_k)\\}_{k=1}^{K}$', fontsize=9)

    for dy in [0.1, -0.65]:
        ax.annotate('', xy=(per_x, main_y + dy + 0.05),
                    xytext=(per_x, main_y + dy + 0.33),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

    # ===== 5. 输出 =====
    out_x = 13.5
    draw_box(ax, out_x, main_y + 0.5, 1.6, 0.8,
             C['output_bg'], C['output_border'],
             '周期集合\n$\\{p_1, ..., p_K\\}$', fontsize=9.5, bold=True)
    draw_box(ax, out_x, main_y - 0.6, 1.6, 0.8,
             C['output_bg'], C['output_border'],
             '权重集合\n$\\{w_1, ..., w_K\\}$', fontsize=9.5, bold=True)

    # ===== 连接箭头 =====
    draw_arrow(ax, 2.1, main_y, 2.9, main_y)       # 输入 → FFT
    draw_arrow(ax, 5.7, main_y, 6.2, main_y)       # FFT → 频谱
    draw_arrow(ax, 8.8, main_y, 9.5, main_y)       # 频谱 → 周期
    draw_arrow(ax, 12.1, main_y + 0.3, 12.7, main_y + 0.5)   # 周期 → 输出上
    draw_arrow(ax, 12.1, main_y - 0.3, 12.7, main_y - 0.6)   # 周期 → 输出下

    plt.tight_layout()

    # 保存到论文目录
    output_path = '/home/zhengxiaodong/exps/Rural-Voltage-Thesis/figures/chap3/fig_fft_period_discovery.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"已保存: {output_path}")

    # 同时保存到代码项目输出目录
    code_output = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/figures/output/chap3/fig_fft_period_discovery.png'
    os.makedirs(os.path.dirname(code_output), exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(14, 6), dpi=300)
    # 不重复绘制，直接从文件拷贝
    plt.close()

    import shutil
    shutil.copy2(output_path, code_output)
    print(f"已复制: {code_output}")


if __name__ == '__main__':
    main()
