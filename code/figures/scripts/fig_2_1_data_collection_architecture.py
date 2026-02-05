#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农村电网数据采集分层架构图
用于论文第二章：数据采集与预处理

IEEE风格，中文标签，300 DPI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.font_manager import FontProperties
import numpy as np
import os

from thesis_style import get_output_dir

# 设置中文字体（与 thesis_style.py 一致）
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

# IEEE 风格配色方案
COLORS = {
    'perception': '#16A085',      # 感知层 - 深青绿色
    'perception_light': '#E8F8F5',
    'perception_border': '#1ABC9C',
    'network': '#E67E22',          # 网络层 - 橙色
    'network_light': '#FEF5E7',
    'network_border': '#F39C12',
    'platform': '#8E44AD',         # 平台层 - 紫色
    'platform_light': '#F5EEF8',
    'platform_border': '#9B59B6',
    'arrow_up': '#27AE60',         # 上行箭头 - 绿色
    'arrow_down': '#3498DB',       # 下行箭头 - 蓝色
    'text': '#2C3E50',             # 文字颜色
    'text_light': '#7F8C8D',       # 浅色文字
    'bg': '#FFFFFF',               # 背景色
}


def draw_layer_box(ax, x, y, width, height, color, light_color, border_color):
    """绘制层级背景框"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=light_color,
        edgecolor=border_color,
        linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(box)


def draw_component_box(ax, x, y, width, height, color, label_cn, label_en):
    """绘制组件框"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor='white',
        edgecolor=color,
        linewidth=2,
        alpha=0.98
    )
    ax.add_patch(box)

    # 中文标签
    ax.text(x + width/2, y + height/2 + 0.018, label_cn,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=COLORS['text'])
    # 英文标签
    ax.text(x + width/2, y + height/2 - 0.018, label_en,
            ha='center', va='center',
            fontsize=9, color=COLORS['text_light'],
            style='italic')


def draw_arrow_with_label(ax, x, y_start, y_end, color, label, side='left'):
    """绘制带标签的垂直箭头"""
    arrow = FancyArrowPatch(
        (x, y_start), (x, y_end),
        arrowstyle='-|>',
        mutation_scale=18,
        color=color,
        linewidth=3,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)

    # 标签位置
    mid_y = (y_start + y_end) / 2
    if side == 'left':
        ax.text(x - 0.03, mid_y, label,
                ha='right', va='center',
                fontsize=10, color=color,
                fontweight='bold', rotation=90)
    else:
        ax.text(x + 0.03, mid_y, label,
                ha='left', va='center',
                fontsize=10, color=color,
                fontweight='bold', rotation=90)


def draw_horizontal_arrow(ax, x_start, x_end, y, color):
    """绘制水平箭头"""
    arrow = FancyArrowPatch(
        (x_start, y), (x_end, y),
        arrowstyle='-|>',
        mutation_scale=12,
        color=color,
        linewidth=2,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)


def draw_icon_meter(ax, x, y, size=0.025):
    """绘制电表图标"""
    # 外框
    rect = FancyBboxPatch(
        (x - size, y - size*0.7), size*2, size*1.4,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor='#ECF0F1',
        edgecolor=COLORS['perception'],
        linewidth=1.5
    )
    ax.add_patch(rect)
    # 显示屏
    screen = Rectangle((x - size*0.7, y - size*0.2), size*1.4, size*0.5,
                       facecolor='#A8E6CF', edgecolor=COLORS['perception'], linewidth=0.8)
    ax.add_patch(screen)


def draw_icon_antenna(ax, x, y, size=0.025):
    """绘制无线信号图标"""
    for i, r in enumerate([size*0.5, size*0.8, size*1.1]):
        arc = mpatches.Arc((x, y - size*0.3), r*2, r*2, angle=0,
                          theta1=40, theta2=140,
                          color=COLORS['network'],
                          linewidth=2.5 - i*0.5)
        ax.add_patch(arc)
    # 信号发射点
    circle = Circle((x, y - size*0.3), size*0.15,
                   facecolor=COLORS['network'], edgecolor='none')
    ax.add_patch(circle)


def draw_icon_database(ax, x, y, size=0.025):
    """绘制数据库图标"""
    from matplotlib.patches import Ellipse
    # 圆柱体
    rect = Rectangle((x - size, y - size*0.5), size*2, size*1.0,
                    facecolor='#D7BDE2', edgecolor=COLORS['platform'], linewidth=1.5)
    ax.add_patch(rect)
    # 顶部椭圆
    ellipse = Ellipse((x, y + size*0.5), size*2, size*0.5,
                     facecolor='#E8DAEF', edgecolor=COLORS['platform'], linewidth=1.5)
    ax.add_patch(ellipse)
    # 底部椭圆
    ellipse_bottom = Ellipse((x, y - size*0.5), size*2, size*0.5,
                            facecolor='#D7BDE2', edgecolor=COLORS['platform'], linewidth=1.5)
    ax.add_patch(ellipse_bottom)


def main():
    # 创建图形 - 调整为横向布局
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])

    # ==================== 层级定义 ====================
    layer_width = 0.85
    layer_height = 0.23
    layer_x = 0.075
    gap = 0.06

    # 平台层 (最上层)
    platform_y = 0.70
    # 网络层 (中间层)
    network_y = 0.41
    # 感知层 (最下层)
    perception_y = 0.12

    # ==================== 绘制三个主层级框 ====================

    # 平台层
    draw_layer_box(ax, layer_x, platform_y, layer_width, layer_height,
                  COLORS['platform'], COLORS['platform_light'], COLORS['platform_border'])
    ax.text(layer_x + 0.025, platform_y + layer_height - 0.035, '平台层',
            fontsize=16, fontweight='bold', color=COLORS['platform'])
    ax.text(layer_x + 0.025, platform_y + layer_height - 0.065, 'Platform Layer',
            fontsize=10, color=COLORS['platform'], style='italic')

    # 网络层
    draw_layer_box(ax, layer_x, network_y, layer_width, layer_height,
                  COLORS['network'], COLORS['network_light'], COLORS['network_border'])
    ax.text(layer_x + 0.025, network_y + layer_height - 0.035, '网络层',
            fontsize=16, fontweight='bold', color=COLORS['network'])
    ax.text(layer_x + 0.025, network_y + layer_height - 0.065, 'Network Layer',
            fontsize=10, color=COLORS['network'], style='italic')

    # 感知层
    draw_layer_box(ax, layer_x, perception_y, layer_width, layer_height,
                  COLORS['perception'], COLORS['perception_light'], COLORS['perception_border'])
    ax.text(layer_x + 0.025, perception_y + layer_height - 0.035, '感知层',
            fontsize=16, fontweight='bold', color=COLORS['perception'])
    ax.text(layer_x + 0.025, perception_y + layer_height - 0.065, 'Perception Layer',
            fontsize=10, color=COLORS['perception'], style='italic')

    # ==================== 感知层组件 ====================
    comp_width = 0.18
    comp_height = 0.10
    comp_y = perception_y + 0.035
    comp_start_x = layer_x + 0.12

    # 智能电表
    draw_component_box(ax, comp_start_x, comp_y, comp_width, comp_height,
                      COLORS['perception'], '智能电表', 'Smart Meter')

    # 配电自动化终端
    draw_component_box(ax, comp_start_x + comp_width + 0.06, comp_y, comp_width + 0.04, comp_height,
                      COLORS['perception'], '配电自动化终端', 'DTU/FTU')

    # 环境传感器
    draw_component_box(ax, comp_start_x + 2*comp_width + 0.14, comp_y, comp_width, comp_height,
                      COLORS['perception'], '环境传感器', 'Sensors')

    # ==================== 网络层组件 ====================
    net_comp_y = network_y + 0.035
    net_comp_width = 0.24

    # 4G/5G无线网络
    net1_x = layer_x + 0.15
    draw_component_box(ax, net1_x, net_comp_y, net_comp_width, comp_height,
                      COLORS['network'], '4G/5G无线网络', 'Wireless Network')

    # 电力载波通信
    net2_x = net1_x + net_comp_width + 0.12
    draw_component_box(ax, net2_x, net_comp_y, net_comp_width + 0.04, comp_height,
                      COLORS['network'], '电力载波通信(PLC)', 'Power Line Comm.')

    # ==================== 平台层组件 ====================
    plat_comp_y = platform_y + 0.035
    plat_comp_width = 0.17

    # 时序数据库
    plat1_x = layer_x + 0.12
    draw_component_box(ax, plat1_x, plat_comp_y, plat_comp_width, comp_height,
                      COLORS['platform'], '时序数据库', 'TSDB')

    # 数据处理
    plat2_x = plat1_x + plat_comp_width + 0.06
    draw_component_box(ax, plat2_x, plat_comp_y, plat_comp_width, comp_height,
                      COLORS['platform'], '数据处理', 'Processing')

    # 异常检测算法
    plat3_x = plat2_x + plat_comp_width + 0.06
    draw_component_box(ax, plat3_x, plat_comp_y, plat_comp_width + 0.06, comp_height,
                      COLORS['platform'], '异常检测算法', 'Anomaly Detection')

    # ==================== 绘制层间连接箭头 ====================

    # 感知层 → 网络层 (左侧)
    draw_arrow_with_label(ax, 0.28, perception_y + layer_height + 0.015,
                         network_y - 0.015, COLORS['arrow_up'], '数据上传', 'left')

    # 感知层 → 网络层 (右侧)
    draw_arrow_with_label(ax, 0.72, perception_y + layer_height + 0.015,
                         network_y - 0.015, COLORS['arrow_up'], '数据上传', 'right')

    # 网络层 → 平台层 (中间)
    draw_arrow_with_label(ax, 0.50, network_y + layer_height + 0.015,
                         platform_y - 0.015, COLORS['arrow_down'], '数据汇聚', 'left')

    # 平台层内部数据流向
    draw_horizontal_arrow(ax, plat1_x + plat_comp_width + 0.01,
                         plat2_x - 0.01, plat_comp_y + comp_height/2, COLORS['platform'])
    draw_horizontal_arrow(ax, plat2_x + plat_comp_width + 0.01,
                         plat3_x - 0.01, plat_comp_y + comp_height/2, COLORS['platform'])

    # ==================== 添加说明标注框 ====================

    # 感知层数据说明
    ax.text(0.5, perception_y - 0.045, '采集数据：电压、电流、功率、谐波失真率、频率等实时数据',
            ha='center', va='center', fontsize=10, color=COLORS['text_light'])

    # 右侧说明框
    # 感知层参数
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=COLORS['perception'], linewidth=1.5, alpha=0.95)
    ax.text(0.93, perception_y + 0.08, '采样频率\n1次/分钟',
            ha='center', va='center', fontsize=9,
            color=COLORS['perception'], fontweight='bold',
            bbox=bbox_props)

    # 网络层参数
    bbox_props['edgecolor'] = COLORS['network']
    ax.text(0.93, network_y + 0.08, '传输带宽\n4G: 100Mbps\nPLC: 2Mbps',
            ha='center', va='center', fontsize=9,
            color=COLORS['network'], fontweight='bold',
            bbox=bbox_props)

    # 平台层参数
    bbox_props['edgecolor'] = COLORS['platform']
    ax.text(0.93, platform_y + 0.08, '处理能力\n>10万点/秒',
            ha='center', va='center', fontsize=9,
            color=COLORS['platform'], fontweight='bold',
            bbox=bbox_props)

    # ==================== 保存图片 ====================
    output_dir = get_output_dir(2)
    output_path = os.path.join(output_dir, 'fig_2_1_data_collection_architecture.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none',
                pad_inches=0.15)
    plt.close()

    print(f"图片已保存: {output_path}")
    return output_path


if __name__ == '__main__':
    main()
