#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农村电网数据采集分层架构图
用于论文第二章：数据采集与预处理

优化版本：紧凑布局，避免重叠
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from thesis_style import get_output_dir

# 设置中文字体
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

# 配色方案（柔和学术风格）
COLORS = {
    'perception': '#16A085',
    'perception_light': '#E8F6F3',
    'perception_border': '#1ABC9C',
    'network': '#D68910',
    'network_light': '#FEF9E7',
    'network_border': '#F5B041',
    'platform': '#7D3C98',
    'platform_light': '#F5EEF8',
    'platform_border': '#AF7AC5',
    'arrow': '#5D6D7E',
    'text': '#2C3E50',
    'text_light': '#7F8C8D',
}


def draw_layer(ax, y_center, height, color, light_color, border_color,
               title_cn, title_en, width=0.82, x_start=0.09):
    """绘制层级背景框"""
    box = FancyBboxPatch(
        (x_start, y_center - height/2), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=light_color,
        edgecolor=border_color,
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)

    # 层级标题（左上角）
    ax.text(x_start + 0.02, y_center + height/2 - 0.025, title_cn,
            fontsize=12, fontweight='bold', color=color, va='top')
    ax.text(x_start + 0.02, y_center + height/2 - 0.055, title_en,
            fontsize=8, color=color, style='italic', va='top')


def draw_component(ax, x_center, y_center, width, height, color,
                   label_cn, label_en):
    """绘制组件框"""
    box = FancyBboxPatch(
        (x_center - width/2, y_center - height/2), width, height,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor='white',
        edgecolor=color,
        linewidth=1.5,
        alpha=0.98
    )
    ax.add_patch(box)

    # 中文标签
    ax.text(x_center, y_center + 0.012, label_cn,
            ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLORS['text'])
    # 英文标签
    ax.text(x_center, y_center - 0.012, label_en,
            ha='center', va='center',
            fontsize=7, color=COLORS['text_light'], style='italic')


def draw_vertical_arrow(ax, x, y_start, y_end, color, label=None, label_side='right'):
    """绘制垂直箭头（带可选标签）"""
    arrow = FancyArrowPatch(
        (x, y_start), (x, y_end),
        arrowstyle='-|>',
        mutation_scale=12,
        color=color,
        linewidth=2,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)

    if label:
        mid_y = (y_start + y_end) / 2
        offset = 0.025 if label_side == 'right' else -0.025
        ha = 'left' if label_side == 'right' else 'right'
        ax.text(x + offset, mid_y, label,
                ha=ha, va='center',
                fontsize=8, color=color)


def draw_horizontal_arrow(ax, x_start, x_end, y, color):
    """绘制水平箭头"""
    arrow = FancyArrowPatch(
        (x_start, y), (x_end, y),
        arrowstyle='-|>',
        mutation_scale=10,
        color=color,
        linewidth=1.5
    )
    ax.add_patch(arrow)


def main():
    # 创建图形 - 紧凑尺寸
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # ==================== 层级参数 ====================
    layer_width = 0.82
    layer_height = 0.22
    layer_x = 0.09

    # 三层的中心 Y 坐标（从上到下：平台层、网络层、感知层）
    platform_y = 0.78
    network_y = 0.50
    perception_y = 0.22

    # ==================== 绘制三个层级 ====================

    # 平台层
    draw_layer(ax, platform_y, layer_height,
               COLORS['platform'], COLORS['platform_light'], COLORS['platform_border'],
               '平台层', 'Platform Layer')

    # 网络层
    draw_layer(ax, network_y, layer_height,
               COLORS['network'], COLORS['network_light'], COLORS['network_border'],
               '网络层', 'Network Layer')

    # 感知层
    draw_layer(ax, perception_y, layer_height,
               COLORS['perception'], COLORS['perception_light'], COLORS['perception_border'],
               '感知层', 'Perception Layer')

    # ==================== 感知层组件 ====================
    comp_h = 0.07
    comp_w = 0.16
    comp_y = perception_y - 0.015

    # 三个组件均匀分布
    comp_positions = [0.25, 0.50, 0.75]

    draw_component(ax, comp_positions[0], comp_y, comp_w, comp_h,
                   COLORS['perception'], '智能电表', 'Smart Meter')
    draw_component(ax, comp_positions[1], comp_y, comp_w + 0.02, comp_h,
                   COLORS['perception'], '配电自动化终端', 'DTU/FTU')
    draw_component(ax, comp_positions[2], comp_y, comp_w, comp_h,
                   COLORS['perception'], '环境传感器', 'Sensors')

    # ==================== 网络层组件 ====================
    net_comp_y = network_y - 0.015
    net_comp_w = 0.20

    draw_component(ax, 0.33, net_comp_y, net_comp_w, comp_h,
                   COLORS['network'], '4G/5G无线网络', 'Wireless Network')
    draw_component(ax, 0.67, net_comp_y, net_comp_w + 0.02, comp_h,
                   COLORS['network'], '电力载波通信', 'PLC')

    # ==================== 平台层组件 ====================
    plat_comp_y = platform_y - 0.015
    plat_comp_w = 0.16

    plat_x1, plat_x2, plat_x3 = 0.25, 0.50, 0.75

    draw_component(ax, plat_x1, plat_comp_y, plat_comp_w, comp_h,
                   COLORS['platform'], '时序数据库', 'TSDB')
    draw_component(ax, plat_x2, plat_comp_y, plat_comp_w, comp_h,
                   COLORS['platform'], '数据处理', 'Processing')
    draw_component(ax, plat_x3, plat_comp_y, plat_comp_w + 0.02, comp_h,
                   COLORS['platform'], '异常检测算法', 'Anomaly Detection')

    # 平台层内部箭头
    draw_horizontal_arrow(ax, plat_x1 + plat_comp_w/2 + 0.01,
                         plat_x2 - plat_comp_w/2 - 0.01, plat_comp_y, COLORS['platform'])
    draw_horizontal_arrow(ax, plat_x2 + plat_comp_w/2 + 0.01,
                         plat_x3 - (plat_comp_w + 0.02)/2 - 0.01, plat_comp_y, COLORS['platform'])

    # ==================== 层间连接箭头 ====================

    # 感知层 → 网络层
    arrow_gap = 0.015
    perception_top = perception_y + layer_height/2
    network_bottom = network_y - layer_height/2

    draw_vertical_arrow(ax, 0.33, perception_top + arrow_gap,
                       network_bottom - arrow_gap, COLORS['arrow'],
                       '数据上传', 'right')
    draw_vertical_arrow(ax, 0.67, perception_top + arrow_gap,
                       network_bottom - arrow_gap, COLORS['arrow'])

    # 网络层 → 平台层
    network_top = network_y + layer_height/2
    platform_bottom = platform_y - layer_height/2

    draw_vertical_arrow(ax, 0.50, network_top + arrow_gap,
                       platform_bottom - arrow_gap, COLORS['arrow'],
                       '数据汇聚', 'right')

    # ==================== 底部说明文字 ====================
    ax.text(0.50, 0.05, '采集数据：电压、电流、功率、谐波失真率、频率等实时数据',
            ha='center', va='center', fontsize=9, color=COLORS['text_light'])

    # ==================== 右侧参数说明 ====================
    # 使用更紧凑的位置
    param_x = 0.94
    param_fontsize = 7

    # 感知层参数
    ax.text(param_x, perception_y, '采样: 1次/分钟',
            ha='center', va='center', fontsize=param_fontsize,
            color=COLORS['perception'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['perception'], linewidth=1, alpha=0.95))

    # 网络层参数
    ax.text(param_x, network_y, '4G: 100Mbps\nPLC: 2Mbps',
            ha='center', va='center', fontsize=param_fontsize,
            color=COLORS['network'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['network'], linewidth=1, alpha=0.95))

    # 平台层参数
    ax.text(param_x, platform_y, '处理: >10万点/秒',
            ha='center', va='center', fontsize=param_fontsize,
            color=COLORS['platform'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['platform'], linewidth=1, alpha=0.95))

    # ==================== 保存图片 ====================
    output_dir = get_output_dir(2)
    output_path = os.path.join(output_dir, 'fig_2_1_data_collection_architecture.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"图片已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    main()
