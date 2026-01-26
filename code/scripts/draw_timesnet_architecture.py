#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimesNet 模型整体架构图绘制脚本
符合 IEEE 论文风格要求
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib import font_manager
import numpy as np

# 设置中文字体 - 使用系统可用的 Noto 字体
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ZH_FONT = font_manager.FontProperties(fname=FONT_PATH)
ZH_FONT_BOLD = font_manager.FontProperties(fname=FONT_PATH, weight='bold')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern 字体显示数学公式
plt.rcParams['text.usetex'] = False

# IEEE 论文配色方案（专业蓝色系）
COLORS = {
    'input': '#E3F2FD',         # 浅蓝 - 输入输出
    'input_border': '#1976D2',   # 深蓝边框
    'embedding': '#BBDEFB',      # 蓝色 - Embedding
    'embedding_border': '#1565C0',
    'timesblock': '#90CAF9',     # 主蓝色 - TimesBlock
    'timesblock_border': '#0D47A1',
    'fft': '#C8E6C9',            # 浅绿 - FFT
    'fft_border': '#388E3C',
    'reshape': '#FFF9C4',        # 浅黄 - 重塑
    'reshape_border': '#F9A825',
    'conv': '#FFCCBC',           # 浅橙 - 卷积
    'conv_border': '#E64A19',
    'aggregate': '#E1BEE7',      # 浅紫 - 聚合
    'aggregate_border': '#7B1FA2',
    'norm': '#B2DFDB',           # 青色 - 归一化
    'norm_border': '#00796B',
    'output': '#E3F2FD',         # 浅蓝 - 输出
    'output_border': '#1976D2',
    'arrow': '#424242',          # 深灰箭头
    'text': '#212121',           # 文字
    'timesblock_bg': '#E8EAF6',  # TimesBlock 背景
}

def draw_rounded_box(ax, x, y, width, height, label, color, border_color,
                     fontsize=9, is_main=False, has_chinese=True):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2 if is_main else 1.5,
        zorder=3
    )
    ax.add_patch(box)

    if has_chinese:
        font_prop_sized = font_manager.FontProperties(fname=FONT_PATH,
                                                       weight='bold' if is_main else 'normal',
                                                       size=fontsize)
        ax.text(x, y, label, ha='center', va='center',
                fontproperties=font_prop_sized, color=COLORS['text'], zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if is_main else 'normal',
                color=COLORS['text'], zorder=4)

    return box

def draw_arrow(ax, start, end, style='->', color=None):
    """绘制箭头"""
    if color is None:
        color = COLORS['arrow']

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
                zorder=2)

def draw_timesblock_internal(ax, x_center, y_center, block_width, block_height):
    """绘制 TimesBlock 内部结构"""
    # 内部模块参数
    internal_width = 1.9
    internal_height = 0.5
    spacing = 0.7

    # 计算起始位置
    y_start = y_center + block_height/2 - 0.8

    modules = [
        ('FFT 周期发现', COLORS['fft'], COLORS['fft_border'], 'FFT_for_Period(x, k)'),
        ('1D → 2D 重塑', COLORS['reshape'], COLORS['reshape_border'], 'Reshape(B,T,C)→(B,p,f,C)'),
        ('Inception 2D Conv', COLORS['conv'], COLORS['conv_border'], '多尺度卷积核'),
        ('2D → 1D 还原', COLORS['reshape'], COLORS['reshape_border'], 'Reshape(B,p,f,C)→(B,T,C)'),
        ('多周期自适应聚合', COLORS['aggregate'], COLORS['aggregate_border'], 'Softmax 加权融合'),
    ]

    y_positions = []
    for i, (label, color, border, sublabel) in enumerate(modules):
        y_pos = y_start - i * spacing
        y_positions.append(y_pos)
        draw_rounded_box(ax, x_center, y_pos, internal_width, internal_height,
                        label, color, border, fontsize=8, has_chinese=True)
        # 添加小字说明
        font_prop_small = font_manager.FontProperties(fname=FONT_PATH, size=6, style='italic')
        ax.text(x_center + internal_width/2 + 0.1, y_pos, sublabel,
                ha='left', va='center', fontproperties=font_prop_small, color='#616161')

    # 绘制内部箭头
    for i in range(len(y_positions) - 1):
        draw_arrow(ax, (x_center, y_positions[i] - internal_height/2 - 0.02),
                  (x_center, y_positions[i+1] + internal_height/2 + 0.02))

    return y_positions[0] + internal_height/2, y_positions[-1] - internal_height/2

def main():
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)
    ax.set_xlim(-4, 6)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')

    # 主要组件参数
    main_width = 2.2
    main_height = 0.7
    timesblock_width = 4.5
    timesblock_height = 4.5

    # Y 坐标
    y_input = 12
    y_embedding = 10.8
    y_timesblock_center = 7
    y_norm = 3.2
    y_projection = 2
    y_output = 0.8

    # 字体定义
    font_main = font_manager.FontProperties(fname=FONT_PATH, size=10, weight='bold')
    font_label = font_manager.FontProperties(fname=FONT_PATH, size=10)
    font_small = font_manager.FontProperties(fname=FONT_PATH, size=8)
    font_tiny = font_manager.FontProperties(fname=FONT_PATH, size=7)
    font_title = font_manager.FontProperties(fname=FONT_PATH, size=12, weight='bold')

    # 1. 输入
    draw_rounded_box(ax, 0, y_input, main_width + 1, main_height,
                    '输入', COLORS['input'], COLORS['input_border'],
                    fontsize=10, is_main=True)
    # 使用 mathtext 显示数学公式，使用更简洁的格式
    ax.text(0, y_input - 0.5, r'$X \in R^{B \times T \times C}$',
            ha='center', va='center', fontsize=10, color=COLORS['text'],
            family='serif', style='italic')

    # 箭头：输入 -> Embedding
    draw_arrow(ax, (0, y_input - main_height/2 - 0.3), (0, y_embedding + main_height/2 + 0.05))

    # 2. Token Embedding
    draw_rounded_box(ax, 0, y_embedding, main_width, main_height,
                    'Token Embedding', COLORS['embedding'], COLORS['embedding_border'],
                    fontsize=10, is_main=True, has_chinese=False)

    # 箭头：Embedding -> TimesBlock
    draw_arrow(ax, (0, y_embedding - main_height/2 - 0.05),
              (0, y_timesblock_center + timesblock_height/2 + 0.05))

    # 3. TimesBlock 大框
    timesblock_box = FancyBboxPatch(
        (-timesblock_width/2, y_timesblock_center - timesblock_height/2),
        timesblock_width, timesblock_height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=COLORS['timesblock_bg'],
        edgecolor=COLORS['timesblock_border'],
        linewidth=2.5,
        zorder=1
    )
    ax.add_patch(timesblock_box)

    # TimesBlock 标题
    ax.text(0, y_timesblock_center + timesblock_height/2 - 0.25,
            'TimesBlock', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['timesblock_border'])

    # 绘制内部结构
    top_y, bottom_y = draw_timesblock_internal(ax, 0, y_timesblock_center,
                                               timesblock_width, timesblock_height)

    # 层数标注
    ax.text(timesblock_width/2 + 0.3, y_timesblock_center,
            r'$\times N_{layers}$', ha='left', va='center',
            fontsize=11, fontweight='bold', color=COLORS['timesblock_border'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['timesblock_border'], alpha=0.8))

    # 残差连接（虚线）
    ax.annotate('', xy=(-timesblock_width/2 - 0.3, y_timesblock_center - timesblock_height/2 + 0.3),
                xytext=(-timesblock_width/2 - 0.3, y_timesblock_center + timesblock_height/2 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5,
                               linestyle='--', connectionstyle='arc3,rad=0.3'))
    ax.text(-timesblock_width/2 - 0.6, y_timesblock_center, '残差',
            ha='center', va='center', fontproperties=font_tiny, color='#757575', rotation=90)

    # 箭头：TimesBlock -> LayerNorm
    draw_arrow(ax, (0, y_timesblock_center - timesblock_height/2 - 0.05),
              (0, y_norm + main_height/2 + 0.05))

    # 4. Layer Normalization
    draw_rounded_box(ax, 0, y_norm, main_width, main_height,
                    'Layer Normalization', COLORS['norm'], COLORS['norm_border'],
                    fontsize=10, is_main=True, has_chinese=False)

    # 箭头：LayerNorm -> Projection
    draw_arrow(ax, (0, y_norm - main_height/2 - 0.05),
              (0, y_projection + main_height/2 + 0.05))

    # 5. 输出投影层
    draw_rounded_box(ax, 0, y_projection, main_width, main_height,
                    '输出投影层', COLORS['embedding'], COLORS['embedding_border'],
                    fontsize=10, is_main=True)

    # 箭头：Projection -> 输出
    draw_arrow(ax, (0, y_projection - main_height/2 - 0.05),
              (0, y_output + main_height/2 + 0.25))

    # 6. 输出
    draw_rounded_box(ax, 0, y_output, main_width + 1, main_height,
                    '输出', COLORS['output'], COLORS['output_border'],
                    fontsize=10, is_main=True)
    ax.text(0, y_output - 0.5, r'$\hat{Y} \in R^{B \times T \times C}$',
            ha='center', va='center', fontsize=10, color=COLORS['text'],
            family='serif', style='italic')

    # 添加右侧说明
    explanation_x = 4.2
    explanations = [
        (y_input, 'B: 批量大小'),
        (y_input - 0.35, 'T: 序列长度'),
        (y_input - 0.7, 'C: 特征维度'),
    ]
    for y, text in explanations:
        ax.text(explanation_x, y, text, ha='left', va='center',
                fontproperties=font_small, color='#616161')

    # 保存图片
    output_path = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/thesis/figures/TimesNet模型整体架构图.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"图片已保存至: {output_path}")
    print(f"图片分辨率: 300 DPI")

if __name__ == '__main__':
    main()
