#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 3-7：异常检测框架流程图
展示基于重构的异常检测完整流程

输出：单个 PNG 文件
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon
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

# 配色方案 - 蓝色系
COLORS = {
    'data': '#1E3A5F',        # 深蓝 - 数据节点
    'process': '#2E5984',     # 中蓝 - 处理节点
    'compute': '#4A7BA7',     # 浅蓝 - 计算节点
    'decision': '#6B9BC3',    # 淡蓝 - 判定节点
    'output': '#8FBCDB',      # 亮蓝 - 输出节点
    'arrow': '#1E3A5F',       # 深蓝 - 箭头
    'formula_bg': '#F0F5FA',  # 浅蓝背景 - 公式框
    'formula_border': '#4A7BA7',  # 公式边框
    'text': '#FFFFFF',        # 白色文字
    'text_dark': '#1E3A5F',   # 深蓝文字
}


def draw_rounded_box(ax, pos, width, height, color, text, fontsize=11):
    """绘制圆角矩形框"""
    x, y = pos
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=color, edgecolor='white',
                         linewidth=1.5, mutation_scale=1)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])


def draw_diamond(ax, pos, width, height, color, text, fontsize=11):
    """绘制菱形框"""
    x, y = pos
    diamond = RegularPolygon((x, y), numVertices=4, radius=width/2,
                             orientation=np.pi/4,
                             facecolor=color, edgecolor='white',
                             linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])


def draw_formula_box(ax, pos, width, height, formula, label, fontsize=12):
    """绘制公式框"""
    x, y = pos
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['formula_bg'],
                         edgecolor=COLORS['formula_border'],
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y + height/2 - 0.2, label, fontsize=10,
            ha='center', va='center', color=COLORS['text_dark'],
            fontweight='bold')
    ax.text(x, y - 0.1, formula, fontsize=fontsize,
            ha='center', va='center', color=COLORS['text_dark'])


def draw_arrow(ax, start, end, style):
    """绘制箭头"""
    arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0", **style)
    ax.add_patch(arrow)


def draw_anomaly_detection_framework():
    """创建异常检测框架流程图"""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # 节点参数
    box_width = 2.0
    box_height = 0.8

    # === 训练阶段 (左侧) ===
    train_data_pos = (1.5, 8.5)
    draw_rounded_box(ax, train_data_pos, box_width, box_height,
                     COLORS['data'], '训练集', fontsize=12)

    model_train_pos = (1.5, 6.8)
    draw_rounded_box(ax, model_train_pos, box_width, box_height,
                     COLORS['process'], '模型训练', fontsize=12)

    recon_error_pos = (1.5, 5.1)
    draw_rounded_box(ax, recon_error_pos, box_width, box_height,
                     COLORS['compute'], '重构误差计算', fontsize=11)

    threshold_pos = (1.5, 3.4)
    draw_rounded_box(ax, threshold_pos, box_width, box_height,
                     COLORS['decision'], '阈值设定', fontsize=12)

    # === 测试阶段 (中间) ===
    test_data_pos = (5.5, 8.5)
    draw_rounded_box(ax, test_data_pos, box_width, box_height,
                     COLORS['data'], '测试集', fontsize=12)

    test_recon_pos = (5.5, 6.8)
    draw_rounded_box(ax, test_recon_pos, box_width, box_height,
                     COLORS['compute'], '重构误差计算', fontsize=11)

    anomaly_detect_pos = (5.5, 5.1)
    draw_diamond(ax, anomaly_detect_pos, 1.8, 1.0,
                 COLORS['decision'], '异常判定', fontsize=11)

    evaluate_pos = (5.5, 3.4)
    draw_rounded_box(ax, evaluate_pos, box_width, box_height,
                     COLORS['output'], '评估指标', fontsize=12)

    # === 公式框 (右侧) ===
    formula1_pos = (9.5, 7.5)
    draw_formula_box(ax, formula1_pos, 4.0, 1.2,
                     r'$e_t = \| x_t - \hat{x}_t \|^2$',
                     '重构误差', fontsize=13)

    formula2_pos = (9.5, 5.5)
    draw_formula_box(ax, formula2_pos, 4.0, 1.2,
                     r'$\tau = \mathrm{percentile}(E, 100-r)$',
                     '阈值计算', fontsize=13)

    formula3_pos = (9.5, 3.5)
    draw_formula_box(ax, formula3_pos, 4.0, 1.2,
                     r'$\hat{y}_t = \mathbb{1}[e_t > \tau]$',
                     '异常判定', fontsize=13)

    # === 连接箭头 ===
    arrow_style = dict(arrowstyle='->', color=COLORS['arrow'],
                       lw=1.5, mutation_scale=15)

    # 训练阶段箭头
    draw_arrow(ax, (1.5, 8.1), (1.5, 7.2), arrow_style)
    draw_arrow(ax, (1.5, 6.4), (1.5, 5.5), arrow_style)
    draw_arrow(ax, (1.5, 4.7), (1.5, 3.8), arrow_style)

    # 测试阶段箭头
    draw_arrow(ax, (5.5, 8.1), (5.5, 7.2), arrow_style)
    draw_arrow(ax, (5.5, 6.4), (5.5, 5.6), arrow_style)
    draw_arrow(ax, (5.5, 4.6), (5.5, 3.8), arrow_style)

    # 阈值到异常判定的连接
    draw_arrow(ax, (2.5, 3.4), (4.5, 5.1), arrow_style)

    # 公式连接虚线箭头
    dashed_style = dict(arrowstyle='->', color=COLORS['formula_border'],
                        lw=1.2, linestyle='--', mutation_scale=12)
    draw_arrow(ax, (7.5, 7.5), (6.5, 6.8), dashed_style)
    draw_arrow(ax, (7.5, 7.5), (2.5, 5.1), dashed_style)
    draw_arrow(ax, (7.5, 5.5), (2.5, 3.4), dashed_style)
    draw_arrow(ax, (7.5, 3.5), (6.4, 5.1), dashed_style)

    # === 阶段标签 ===
    ax.text(1.5, 9.3, '训练阶段', fontsize=14, fontweight='bold',
            ha='center', va='center', color=COLORS['text_dark'])
    ax.text(5.5, 9.3, '测试阶段', fontsize=14, fontweight='bold',
            ha='center', va='center', color=COLORS['text_dark'])
    ax.text(9.5, 9.3, '数学表达', fontsize=14, fontweight='bold',
            ha='center', va='center', color=COLORS['text_dark'])

    # === 分隔线 ===
    ax.axvline(x=3.8, ymin=0.15, ymax=0.92, color='#CCCCCC',
               linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=7.2, ymin=0.15, ymax=0.92, color='#CCCCCC',
               linestyle='--', linewidth=1, alpha=0.7)

    # === Point Adjustment 说明 ===
    pa_box = FancyBboxPatch((3, 1.2), 5, 1.2,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#FFF8E7', edgecolor='#E8A838',
                            linewidth=1.5)
    ax.add_patch(pa_box)
    ax.text(5.5, 2.05, 'Point Adjustment 评估策略', fontsize=10,
            ha='center', va='center', color='#8B6914', fontweight='bold')
    ax.text(5.5, 1.55, '异常段内任意点正确检测 → 整段标记为正确',
            fontsize=9, ha='center', va='center', color='#5C4A0F')

    plt.tight_layout()

    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'fig_3_7_anomaly_detection_framework.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    draw_anomaly_detection_framework()
