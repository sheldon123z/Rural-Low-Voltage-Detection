#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inception 2D 卷积模块示意图
展示 1D→2D 维度重组 + 多尺度卷积 + 维度还原

输出: fig_2d_conv_inception.png
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
    'reshape_bg': '#EBF3EB',
    'reshape_border': '#72A86D',
    'conv_bg': '#4878A8',
    'conv_border': '#3A6190',
    'pool_bg': '#9B7BB8',
    'pool_border': '#7D5FA0',
    'merge_bg': '#FFF3E6',
    'merge_border': '#C4785C',
    'output_bg': '#FFF8E6',
    'output_border': '#D4A84C',
    'arrow': '#505050',
    'text': '#333333',
    'white': '#FFFFFF',
    'step_circle': '#4878A8',
    'light_bg': '#F8FAFB',
}


def draw_box(ax, x, y, w, h, bg, border, text, fontsize=10, bold=False, text_color=None):
    """绘制圆角矩形"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(box)
    fw = 'bold' if bold else 'normal'
    tc = text_color if text_color else C['text']
    ax.text(x, y, text, fontsize=fontsize, fontweight=fw,
            ha='center', va='center', color=tc)


def draw_arrow(ax, x1, y1, x2, y2, color=None):
    """绘制箭头"""
    c = color if color else C['arrow']
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', color=c,
                            lw=1.5, mutation_scale=15,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


def draw_step_label(ax, x, y, num):
    """绘制步骤编号"""
    circle = plt.Circle((x, y), 0.25, facecolor=C['step_circle'],
                        edgecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', zorder=6)


def draw_2d_grid(ax, x, y, rows, cols, cell_w, cell_h, color, alpha=0.6):
    """绘制 2D 网格示意"""
    for r in range(rows):
        for c in range(cols):
            cx = x + c * cell_w
            cy = y - r * cell_h
            rect = plt.Rectangle((cx, cy), cell_w * 0.9, cell_h * 0.9,
                                facecolor=color, edgecolor='white',
                                linewidth=0.5, alpha=alpha)
            ax.add_patch(rect)


def main():
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    main_y = 3.5

    # ===== 1. 输入序列 =====
    draw_box(ax, 0.8, main_y, 1.5, 1.0,
             C['input_bg'], C['input_border'],
             '1D序列\n$\\mathbf{x} \\in \\mathbb{R}^{T}$', fontsize=9.5, bold=True)

    # ===== 2. 维度重组 (1D → 2D) =====
    rs_x = 3.3
    # 背景
    rs_bg = FancyBboxPatch((rs_x - 1.4, main_y - 2.0), 2.8, 4.0,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=C['reshape_bg'], edgecolor=C['reshape_border'],
                           linewidth=1.5, alpha=0.3)
    ax.add_patch(rs_bg)
    draw_step_label(ax, rs_x, main_y + 2.3, 1)
    ax.text(rs_x, main_y + 1.9, '维度重组', fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=C['reshape_border'])

    # 1D 条
    rect_1d = plt.Rectangle((rs_x - 1.1, main_y + 0.8), 2.2, 0.35,
                             facecolor=C['reshape_border'], edgecolor='white',
                             linewidth=0.5, alpha=0.5)
    ax.add_patch(rect_1d)
    ax.text(rs_x, main_y + 0.98, '原始序列 $T$', fontsize=8,
            ha='center', va='center', color=C['text'])

    # 箭头
    ax.annotate('', xy=(rs_x, main_y + 0.2),
                xytext=(rs_x, main_y + 0.7),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))
    ax.text(rs_x + 0.9, main_y + 0.45, '按周期$p$\n折叠', fontsize=8,
            ha='left', va='center', color='#777777')

    # 2D 矩阵
    draw_2d_grid(ax, rs_x - 0.8, main_y + 0.05, 3, 4, 0.4, 0.35,
                 C['reshape_border'], alpha=0.4)
    ax.text(rs_x, main_y - 1.15, '$\\mathbb{R}^{(T/p) \\times p}$', fontsize=9,
            ha='center', va='center', color=C['reshape_border'])

    # ===== 3. 多尺度卷积模块 (Inception) =====
    inc_x = 7.5
    inc_bg = FancyBboxPatch((inc_x - 2.8, main_y - 2.3), 5.6, 4.9,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=C['light_bg'], edgecolor=C['conv_border'],
                            linewidth=1.5, alpha=0.5)
    ax.add_patch(inc_bg)
    draw_step_label(ax, inc_x, main_y + 2.9, 2)
    ax.text(inc_x, main_y + 2.5, '多尺度卷积模块 (Inception Block)', fontsize=10.5,
            fontweight='bold', ha='center', va='center', color=C['conv_border'])

    # 四个并行分支
    branches = [
        ('$1 \\times 1$\n卷积', '点级特征', inc_x - 2.0),
        ('$3 \\times 3$\n卷积', '局部特征', inc_x - 0.7),
        ('$5 \\times 5$\n卷积', '中等特征', inc_x + 0.6),
        ('MaxPool\n$+ 1 \\times 1$', '全局特征', inc_x + 1.9),
    ]

    branch_y = main_y + 1.0
    branch_h = 1.0
    branch_w = 1.1

    for label, desc, bx in branches:
        bg = C['conv_bg'] if 'MaxPool' not in label else C['pool_bg']
        border = C['conv_border'] if 'MaxPool' not in label else C['pool_border']
        draw_box(ax, bx, branch_y, branch_w, branch_h,
                 bg, border, label, fontsize=8.5, bold=True, text_color=C['white'])
        ax.text(bx, branch_y - 0.7, desc, fontsize=7.5,
                ha='center', va='center', color='#888888')

    # 合并箭头 (从分支到拼接)
    concat_y = main_y - 0.5
    for _, _, bx in branches:
        ax.annotate('', xy=(inc_x, concat_y + 0.35),
                    xytext=(bx, branch_y - 0.95),
                    arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.8))

    # 拼接
    draw_box(ax, inc_x, concat_y, 2.0, 0.6,
             C['merge_bg'], C['merge_border'],
             '拼接 (Concat)', fontsize=9, bold=True)

    # 激活函数 + 输出卷积
    act_y = concat_y - 0.8
    draw_box(ax, inc_x, act_y, 2.0, 0.5,
             '#F0F0F0', '#AAAAAA',
             '激活函数 + 输出卷积', fontsize=8.5)

    ax.annotate('', xy=(inc_x, act_y + 0.25),
                xytext=(inc_x, concat_y - 0.32),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

    # ===== 4. 输出 =====
    out_x = 11.8
    # 2D 特征图
    draw_box(ax, out_x, main_y + 0.8, 1.8, 0.8,
             C['reshape_bg'], C['reshape_border'],
             '2D特征图\n$\\mathbb{R}^{(T/p) \\times p}$', fontsize=8.5, bold=True)

    # 维度还原
    rs2_x = 13.5
    rs2_bg = FancyBboxPatch((rs2_x - 1.0, main_y - 1.2), 2.0, 2.8,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=C['reshape_bg'], edgecolor=C['reshape_border'],
                            linewidth=1.5, alpha=0.3)
    ax.add_patch(rs2_bg)
    draw_step_label(ax, rs2_x, main_y + 1.9, 3)
    ax.text(rs2_x, main_y + 1.5, '维度还原', fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=C['reshape_border'])

    # 2D → 1D 示意
    draw_2d_grid(ax, rs2_x - 0.6, main_y + 0.6, 2, 3, 0.35, 0.3,
                 C['reshape_border'], alpha=0.4)
    ax.annotate('', xy=(rs2_x, main_y - 0.25),
                xytext=(rs2_x, main_y + 0.0),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))
    rect_out = plt.Rectangle((rs2_x - 0.9, main_y - 0.7), 1.8, 0.3,
                              facecolor=C['reshape_border'], edgecolor='white',
                              linewidth=0.5, alpha=0.5)
    ax.add_patch(rect_out)

    # 输出
    draw_box(ax, out_x, main_y - 1.5, 1.8, 0.8,
             C['output_bg'], C['output_border'],
             '输出序列\n$\\mathbf{y} \\in \\mathbb{R}^{T}$', fontsize=9, bold=True)

    # ===== 连接箭头 =====
    draw_arrow(ax, 1.55, main_y, 1.9, main_y)              # 输入 → 重组
    draw_arrow(ax, 4.7, main_y, 4.7, main_y)               # 重组 → Inception
    draw_arrow(ax, 10.1, main_y - 0.2, 10.9, main_y - 0.2) # Inception → 2D特征
    draw_arrow(ax, 10.1, main_y + 0.8, 10.9, main_y + 0.8) # Inception → 2D特征
    draw_arrow(ax, 12.7, main_y + 0.8, 12.5, main_y + 0.8) # 2D特征 → 还原
    draw_arrow(ax, rs2_x, main_y - 1.0, out_x + 0.0, main_y - 1.1) # 还原 → 输出

    plt.tight_layout()

    output_path = '/home/zhengxiaodong/exps/Rural-Voltage-Thesis/figures/chap3/fig_2d_conv_inception.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"已保存: {output_path}")

    import shutil
    code_output = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/figures/output/chap3/fig_2d_conv_inception.png'
    os.makedirs(os.path.dirname(code_output), exist_ok=True)
    shutil.copy2(output_path, code_output)
    print(f"已复制: {code_output}")


if __name__ == '__main__':
    main()
