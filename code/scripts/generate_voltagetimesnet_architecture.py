#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 VoltageTimesNet 架构图
IEEE 风格，突出与 TimesNet 的改进点

改进点:
1. FFT周期发现 + 预设电网周期（60/300/900/3600秒）混合
2. 70% FFT + 30% 预设权重策略
3. 时域平滑卷积层

作者: Claude Code
日期: 2026-01-26
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.collections import PatchCollection
import numpy as np

# 设置中文字体 - 使用系统可用的 Noto Sans CJK SC
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 清除字体缓存以确保新字体设置生效
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

def create_rounded_box(ax, xy, width, height, color, text, fontsize=9,
                       text_color='black', edgecolor='black', linewidth=1.5,
                       alpha=1.0, bold=False):
    """创建圆角矩形框"""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(box)

    # 添加文本
    center_x = xy[0] + width / 2
    center_y = xy[1] + height / 2
    weight = 'bold' if bold else 'normal'
    ax.text(center_x, center_y, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=weight)

    return box

def draw_arrow(ax, start, end, color='black', style='->', linewidth=1.5,
               connectionstyle='arc3,rad=0'):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=linewidth,
                               connectionstyle=connectionstyle))

def draw_double_arrow(ax, start, end, color='black', linewidth=1.5):
    """绘制双向箭头（用于表示融合）"""
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    ax.annotate('', xy=mid, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))
    ax.annotate('', xy=mid, xytext=end,
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))

def main():
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 16), dpi=100)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # 颜色定义 - IEEE 风格
    color_input = '#E3F2FD'       # 浅蓝色 - 输入/输出
    color_embed = '#FFF3E0'       # 浅橙色 - 嵌入层
    color_fft = '#E8F5E9'         # 浅绿色 - FFT路径
    color_preset = '#FCE4EC'      # 浅粉色 - 预设周期路径
    color_fusion = '#F3E5F5'      # 浅紫色 - 融合
    color_conv = '#FFFDE7'        # 浅黄色 - 卷积层
    color_norm = '#E0F7FA'        # 浅青色 - 归一化
    color_block = '#E8EAF6'       # 浅靛蓝色 - 主模块背景
    color_smooth = '#FFF8E1'      # 浅琥珀色 - 时域平滑 (改进点)
    color_highlight = '#FFECB3'   # 高亮色 - 改进点

    # 边框颜色
    edge_fft = '#4CAF50'          # 绿色 - FFT路径
    edge_preset = '#E91E63'       # 粉色 - 预设周期路径
    edge_fusion = '#9C27B0'       # 紫色 - 融合
    edge_highlight = '#FF9800'    # 橙色 - 改进点高亮

    # ============ 顶部：输入 ============
    create_rounded_box(ax, (5.5, 15.0), 3, 0.6, color_input, '输入', fontsize=11, bold=True)
    ax.text(7, 14.5, r'$X \in \mathbb{R}^{B \times T \times C}$', ha='center', va='center', fontsize=10)

    # 右侧注释
    ax.text(12.5, 15.2, 'B: 批量大小', ha='left', va='center', fontsize=9, color='#666666')
    ax.text(12.5, 14.85, 'T: 序列长度', ha='left', va='center', fontsize=9, color='#666666')
    ax.text(12.5, 14.5, 'C: 特征维度', ha='left', va='center', fontsize=9, color='#666666')

    # 箭头：输入 -> Token Embedding
    draw_arrow(ax, (7, 14.2), (7, 13.7))

    # ============ Token Embedding ============
    create_rounded_box(ax, (5, 13.0), 4, 0.6, color_embed, 'Token Embedding', fontsize=11, bold=True)

    # 箭头：Token Embedding -> VoltageTimesBlock
    draw_arrow(ax, (7, 12.9), (7, 12.3))

    # ============ VoltageTimesBlock 主模块 ============
    # 背景框
    block_bg = FancyBboxPatch(
        (1.2, 3.5), 11.6, 8.6,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color_block,
        edgecolor='#3F51B5',
        linewidth=2.5,
        alpha=0.5
    )
    ax.add_patch(block_bg)
    ax.text(7, 11.85, 'VoltageTimesBlock', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1A237E')

    # ============ 混合周期发现模块 (核心改进) ============
    # 混合模块背景框
    hybrid_bg = FancyBboxPatch(
        (1.8, 8.8), 10.4, 2.7,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor='#FAFAFA',
        edgecolor=edge_highlight,
        linewidth=2.0,
        linestyle='--',
        alpha=0.8
    )
    ax.add_patch(hybrid_bg)
    ax.text(7, 11.25, '混合周期发现 (改进点1)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=edge_highlight)

    # FFT 周期发现 (左侧路径)
    create_rounded_box(ax, (2.5, 9.8), 3.5, 0.7, color_fft, 'FFT 周期发现',
                       fontsize=10, edgecolor=edge_fft, linewidth=2)
    ax.text(4.25, 9.3, r'FFT_for_Period$(x, k)$', ha='center', va='center',
            fontsize=8, color='#2E7D32', style='italic')

    # 预设电网周期 (右侧路径)
    create_rounded_box(ax, (8, 9.8), 3.5, 0.7, color_preset, '预设电网周期',
                       fontsize=10, edgecolor=edge_preset, linewidth=2)

    # 预设周期表格
    table_x, table_y = 8.2, 9.0
    table_data = [
        ('周期名称', '采样点数'),
        ('1分钟', '60'),
        ('5分钟', '300'),
        ('15分钟', '900'),
        ('1小时', '3600'),
    ]

    # 表格背景
    table_bg = Rectangle((table_x - 0.1, table_y - 0.95), 3.1, 1.05,
                         facecolor='white', edgecolor=edge_preset,
                         linewidth=1, alpha=0.9)
    ax.add_patch(table_bg)

    # 表格内容
    for i, (name, value) in enumerate(table_data):
        y_pos = table_y - i * 0.2
        if i == 0:
            ax.text(table_x + 0.7, y_pos, name, ha='center', va='center',
                   fontsize=7.5, fontweight='bold', color='#C2185B')
            ax.text(table_x + 2.2, y_pos, value, ha='center', va='center',
                   fontsize=7.5, fontweight='bold', color='#C2185B')
            # 分隔线
            ax.plot([table_x, table_x + 3], [y_pos - 0.1, y_pos - 0.1],
                   color=edge_preset, linewidth=0.5)
        else:
            ax.text(table_x + 0.7, y_pos, name, ha='center', va='center', fontsize=7.5)
            ax.text(table_x + 2.2, y_pos, value, ha='center', va='center', fontsize=7.5)

    # 权重融合框 (改进点2)
    create_rounded_box(ax, (5, 7.6), 4, 0.8, color_fusion, '权重融合',
                       fontsize=10, edgecolor=edge_fusion, linewidth=2, bold=True)
    ax.text(7, 7.1, '70% FFT + 30% 预设 (改进点2)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=edge_fusion)

    # 箭头：FFT -> 融合
    draw_arrow(ax, (4.25, 9.7), (5.8, 8.5), color=edge_fft, linewidth=1.5,
               connectionstyle='arc3,rad=0.15')

    # 箭头：预设周期 -> 融合
    draw_arrow(ax, (9.75, 9.7), (8.2, 8.5), color=edge_preset, linewidth=1.5,
               connectionstyle='arc3,rad=-0.15')

    # ============ 1D -> 2D 重塑 ============
    draw_arrow(ax, (7, 7.5), (7, 7.1))
    create_rounded_box(ax, (5, 6.35), 4, 0.6, color_conv, '1D → 2D 重塑', fontsize=10)
    ax.text(10.5, 6.65, r'Reshape$(B,T,C) \rightarrow (B,p,T/p,C)$',
            ha='left', va='center', fontsize=8, color='#666666')

    # ============ Inception 2D Conv ============
    draw_arrow(ax, (7, 6.25), (7, 5.85))
    create_rounded_box(ax, (4.5, 5.15), 5, 0.6, color_conv, 'Inception 2D Conv',
                       fontsize=10, bold=True)
    ax.text(10.5, 5.45, '多尺度卷积核', ha='left', va='center',
            fontsize=8, color='#666666')

    # ============ 2D -> 1D 还原 ============
    draw_arrow(ax, (7, 5.05), (7, 4.65))
    create_rounded_box(ax, (5, 3.95), 4, 0.6, color_conv, '2D → 1D 还原', fontsize=10)
    ax.text(10.5, 4.25, r'Reshape$(B,p,T/p,C) \rightarrow (B,T,C)$',
            ha='left', va='center', fontsize=8, color='#666666')

    # ============ 多周期自适应聚合 ============
    draw_arrow(ax, (7, 3.85), (7, 3.55))

    # 残差连接 (虚线框)
    residual_arrow_start = (1.5, 12.0)
    residual_arrow_end = (1.5, 3.2)
    ax.annotate('', xy=residual_arrow_end, xytext=residual_arrow_start,
                arrowprops=dict(arrowstyle='-', color='#757575', lw=1.5,
                               linestyle='--'))
    ax.text(1.3, 7.5, '残\n差\n连\n接', ha='center', va='center',
            fontsize=8, color='#757575')

    # ============ 时域平滑卷积 (改进点3) ============
    smooth_bg = FancyBboxPatch(
        (4.3, 2.0), 5.4, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color_highlight,
        edgecolor=edge_highlight,
        linewidth=2.0,
        linestyle='--',
        alpha=0.3
    )
    ax.add_patch(smooth_bg)

    create_rounded_box(ax, (4.5, 2.7), 5, 0.7, color_smooth, '时域平滑卷积 (改进点3)',
                       fontsize=10, edgecolor=edge_highlight, linewidth=2, bold=True)
    ax.text(7, 2.2, '深度可分离1D卷积，抑制高频噪声', ha='center', va='center',
            fontsize=8, color='#E65100')

    # 残差连接到时域平滑
    ax.annotate('', xy=(4.4, 3.0), xytext=(1.5, 3.0),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5,
                               linestyle='--'))

    # 循环标记
    ax.text(12.3, 7.5, r'$\times N_{layers}$', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#3F51B5')

    # 循环箭头
    loop_style = 'arc3,rad=0.3'
    ax.annotate('', xy=(12.0, 11.5), xytext=(12.0, 3.5),
                arrowprops=dict(arrowstyle='->', color='#3F51B5', lw=1.5,
                               connectionstyle=loop_style))

    # ============ Layer Normalization ============
    draw_arrow(ax, (7, 1.9), (7, 1.5))
    create_rounded_box(ax, (5, 0.8), 4, 0.6, color_norm, 'Layer Normalization',
                       fontsize=10, edgecolor='#00838F', linewidth=1.5, bold=True)

    # ============ 输出投影层 ============
    draw_arrow(ax, (7, 0.7), (7, 0.3))
    create_rounded_box(ax, (5.5, -0.4), 3, 0.6, color_embed, '输出投影层', fontsize=10)

    # ============ 输出 ============
    draw_arrow(ax, (7, -0.5), (7, -0.9))
    create_rounded_box(ax, (5.5, -1.6), 3, 0.6, color_input, '输出', fontsize=11, bold=True)
    ax.text(7, -2.1, r'$\hat{X} \in \mathbb{R}^{B \times T \times C}$',
            ha='center', va='center', fontsize=10)

    # ============ 图例 ============
    legend_x = 0.3
    legend_y = -0.5

    # 图例背景
    legend_bg = FancyBboxPatch(
        (legend_x - 0.1, legend_y - 1.7), 3.8, 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor='white',
        edgecolor='#BDBDBD',
        linewidth=1,
        alpha=0.95
    )
    ax.add_patch(legend_bg)

    ax.text(legend_x + 1.8, legend_y - 0.15, '图例', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # 图例项
    legend_items = [
        (color_fft, edge_fft, 'FFT 周期发现路径'),
        (color_preset, edge_preset, '预设周期路径'),
        (color_highlight, edge_highlight, '改进点 (相对TimesNet)'),
    ]

    for i, (fc, ec, text) in enumerate(legend_items):
        y = legend_y - 0.5 - i * 0.4
        rect = Rectangle((legend_x, y - 0.12), 0.5, 0.24,
                         facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(legend_x + 0.7, y, text, ha='left', va='center', fontsize=8)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/thesis/figures/VoltageTimesNet架构图.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f'架构图已保存至: {output_path}')

    # 同时保存 PDF 格式
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f'PDF 格式已保存至: {pdf_path}')

    plt.close()

if __name__ == '__main__':
    main()
