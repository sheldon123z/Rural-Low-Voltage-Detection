#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 IEEE 风格的 1D→2D 时序转换示意图

展示 TimesNet 模型中将 1D 时间序列转换为 2D 张量的过程：
- 1D 时序 [B, T, C] 如何转换为 2D 张量 [B, C, T/p, p]
- 使用实际数字示例（T=100, p=10, 得到 10×10 矩阵）
- 包含 padding 过程的说明
- 中文标签，300 DPI PNG 格式
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体（使用系统可用的 Noto Sans CJK SC）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# IEEE 风格配色
COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'green': '#70AD47',
    'red': '#C00000',
    'purple': '#7030A0',
    'gray': '#808080',
    'light_blue': '#B4C7E7',
    'light_orange': '#F8CBAD',
    'light_green': '#C6EFCE',
    'light_gray': '#D9D9D9',
    'white': '#FFFFFF',
}

def draw_1d_sequence(ax, x_start, y_center, width, height, T, C, label_below=True):
    """绘制 1D 时间序列表示"""
    # 绘制主矩形框
    rect = FancyBboxPatch(
        (x_start, y_center - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=COLORS['light_blue'],
        edgecolor=COLORS['blue'],
        linewidth=1.5
    )
    ax.add_patch(rect)

    # 绘制内部分割线（表示时间步）
    num_divisions = 10
    for i in range(1, num_divisions):
        x = x_start + i * width / num_divisions
        ax.plot([x, x], [y_center - height/2 + 0.02, y_center + height/2 - 0.02],
                color=COLORS['blue'], linewidth=0.5, alpha=0.5)

    # 维度标注
    ax.annotate('', xy=(x_start + width, y_center - height/2 - 0.08),
                xytext=(x_start, y_center - height/2 - 0.08),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1))
    ax.text(x_start + width/2, y_center - height/2 - 0.15, f'T = {T}',
            ha='center', va='top', fontsize=9, color=COLORS['gray'])

    # 通道标注（竖向）
    ax.annotate('', xy=(x_start - 0.05, y_center + height/2),
                xytext=(x_start - 0.05, y_center - height/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1))
    ax.text(x_start - 0.1, y_center, f'C',
            ha='right', va='center', fontsize=9, color=COLORS['gray'])

    # 张量形状标注
    if label_below:
        ax.text(x_start + width/2, y_center + height/2 + 0.08,
                r'$\mathbf{X}^{1D}$: [B, T, C]',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=COLORS['blue'])
        ax.text(x_start + width/2, y_center + height/2 + 0.2,
                f'[B, {T}, C]',
                ha='center', va='bottom', fontsize=9, color=COLORS['gray'])

def draw_2d_tensor(ax, x_start, y_center, size, periods, num_periods, is_padded=False):
    """绘制 2D 张量表示"""
    # 计算实际绘制尺寸
    cell_size = size / max(periods, num_periods)

    # 绘制网格
    for i in range(num_periods):
        for j in range(periods):
            # 判断是否为 padding 区域
            if is_padded and (i == num_periods - 1 and j >= periods - 2):
                color = COLORS['light_gray']
                edge_color = COLORS['gray']
            else:
                color = COLORS['light_orange']
                edge_color = COLORS['orange']

            rect = Rectangle(
                (x_start + j * cell_size, y_center - size/2 + i * cell_size),
                cell_size, cell_size,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=0.8
            )
            ax.add_patch(rect)

    # 外框
    outer_rect = Rectangle(
        (x_start, y_center - size/2),
        periods * cell_size, num_periods * cell_size,
        facecolor='none',
        edgecolor=COLORS['orange'],
        linewidth=2
    )
    ax.add_patch(outer_rect)

    # 维度标注 - 水平方向（周期内：intra-period）
    ax.annotate('', xy=(x_start + periods * cell_size, y_center - size/2 - 0.08),
                xytext=(x_start, y_center - size/2 - 0.08),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1))
    ax.text(x_start + periods * cell_size / 2, y_center - size/2 - 0.15,
            f'p = {periods}',
            ha='center', va='top', fontsize=9, color=COLORS['gray'])
    ax.text(x_start + periods * cell_size / 2, y_center - size/2 - 0.28,
            '周期内变化',
            ha='center', va='top', fontsize=8, color=COLORS['orange'])

    # 维度标注 - 垂直方向（周期间：inter-period）
    ax.annotate('', xy=(x_start - 0.05, y_center - size/2 + num_periods * cell_size),
                xytext=(x_start - 0.05, y_center - size/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1))
    ax.text(x_start - 0.1, y_center,
            f'T/p\n= {num_periods}',
            ha='right', va='center', fontsize=9, color=COLORS['gray'])
    ax.text(x_start - 0.25, y_center,
            '周\n期\n间\n变\n化',
            ha='right', va='center', fontsize=8, color=COLORS['orange'])

    # 张量形状标注
    ax.text(x_start + periods * cell_size / 2, y_center + size/2 + 0.08,
            r'$\mathbf{X}^{2D}$: [B, C, T/p, p]',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=COLORS['orange'])
    ax.text(x_start + periods * cell_size / 2, y_center + size/2 + 0.2,
            f'[B, C, {num_periods}, {periods}]',
            ha='center', va='bottom', fontsize=9, color=COLORS['gray'])

def draw_arrow(ax, start, end, label='', color='gray', style='simple'):
    """绘制箭头"""
    if style == 'simple':
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=COLORS[color],
                                   lw=2, connectionstyle='arc3,rad=0'))
    else:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=COLORS[color],
                                   lw=2, connectionstyle='arc3,rad=0.2'))

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.1
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=9, color=COLORS[color], fontweight='bold')

def draw_padding_illustration(ax, x_start, y_center):
    """绘制 padding 说明"""
    # 绘制原始序列
    rect1 = FancyBboxPatch(
        (x_start, y_center), 0.6, 0.15,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=COLORS['light_blue'],
        edgecolor=COLORS['blue'],
        linewidth=1
    )
    ax.add_patch(rect1)
    ax.text(x_start + 0.3, y_center + 0.075, 'T=102',
            ha='center', va='center', fontsize=8)

    # 绘制 padding 部分
    rect2 = FancyBboxPatch(
        (x_start + 0.6, y_center), 0.18, 0.15,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=COLORS['light_gray'],
        edgecolor=COLORS['gray'],
        linewidth=1
    )
    ax.add_patch(rect2)
    ax.text(x_start + 0.69, y_center + 0.075, '+8',
            ha='center', va='center', fontsize=8, color=COLORS['gray'])

    # 箭头指向
    ax.annotate('', xy=(x_start + 0.39, y_center - 0.02),
                xytext=(x_start + 0.39, y_center - 0.12),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1))
    ax.text(x_start + 0.39, y_center - 0.18,
            'padding至\n可整除长度\n(110=11×10)',
            ha='center', va='top', fontsize=7, color=COLORS['gray'])

def create_1d_to_2d_diagram():
    """创建完整的 1D→2D 转换示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # 参数设置
    T = 100  # 序列长度
    p = 10   # 周期
    num_periods = T // p  # 周期数 = 10
    C = 'C'  # 通道数（符号表示）

    # ========== 第一部分：1D 时间序列 ==========
    draw_1d_sequence(ax, x_start=0, y_center=0.5, width=1.2, height=0.3, T=T, C=C)

    # ========== 第二部分：FFT 周期发现框 ==========
    fft_x = 1.5
    fft_box = FancyBboxPatch(
        (fft_x, 0.25), 0.6, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS['light_green'],
        edgecolor=COLORS['green'],
        linewidth=1.5
    )
    ax.add_patch(fft_box)
    ax.text(fft_x + 0.3, 0.55, 'FFT', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['green'])
    ax.text(fft_x + 0.3, 0.4, '周期发现', ha='center', va='center',
            fontsize=9, color=COLORS['green'])
    ax.text(fft_x + 0.3, 0.28, f'p = {p}', ha='center', va='center',
            fontsize=9, color=COLORS['gray'])

    # 箭头：1D → FFT
    draw_arrow(ax, (1.25, 0.5), (1.48, 0.5), '', 'gray')

    # ========== 第三部分：Reshape 操作框 ==========
    reshape_x = 2.3
    reshape_box = FancyBboxPatch(
        (reshape_x, 0.15), 0.5, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS['white'],
        edgecolor=COLORS['purple'],
        linewidth=1.5
    )
    ax.add_patch(reshape_box)
    ax.text(reshape_x + 0.25, 0.65, 'Reshape', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['purple'])
    ax.text(reshape_x + 0.25, 0.5, '1D → 2D', ha='center', va='center',
            fontsize=9, color=COLORS['purple'])
    ax.text(reshape_x + 0.25, 0.35, f'[B,{T},C]', ha='center', va='center',
            fontsize=8, color=COLORS['gray'])
    ax.text(reshape_x + 0.25, 0.22, '↓', ha='center', va='center',
            fontsize=10, color=COLORS['purple'])
    ax.text(reshape_x + 0.25, 0.18, f'[B,C,{num_periods},{p}]', ha='center', va='center',
            fontsize=8, color=COLORS['gray'])

    # 箭头：FFT → Reshape
    draw_arrow(ax, (2.12, 0.5), (2.28, 0.5), '', 'gray')

    # ========== 第四部分：2D 张量 ==========
    draw_2d_tensor(ax, x_start=3.1, y_center=0.5, size=0.8,
                   periods=p, num_periods=num_periods, is_padded=False)

    # 箭头：Reshape → 2D
    draw_arrow(ax, (2.82, 0.5), (3.05, 0.5), '', 'gray')

    # ========== 底部：详细过程说明 ==========
    # 步骤说明框
    step_y = -0.35
    step_box_width = 3.8
    step_box = FancyBboxPatch(
        (0.3, step_y - 0.35), step_box_width, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor='#F5F5F5',
        edgecolor=COLORS['gray'],
        linewidth=1,
        linestyle='--'
    )
    ax.add_patch(step_box)

    # 步骤标题
    ax.text(2.2, step_y - 0.0, '转换过程详解', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['gray'])

    # 步骤1
    ax.text(0.5, step_y - 0.15, '① FFT周期发现:', ha='left', va='center',
            fontsize=9, fontweight='bold', color=COLORS['green'])
    ax.text(1.5, step_y - 0.15, r'$\mathbf{X}_f = \mathrm{FFT}(\mathbf{X}^{1D})$，取幅值最大的频率对应的周期 $p$',
            ha='left', va='center', fontsize=8, color=COLORS['gray'])

    # 步骤2
    ax.text(0.5, step_y - 0.35, '② Padding对齐:', ha='left', va='center',
            fontsize=9, fontweight='bold', color=COLORS['gray'])
    ax.text(1.5, step_y - 0.35, f'若 T 不能被 p 整除，零填充至 T\' = ⌈T/p⌉ × p（本例: {T}→{T}, 无需填充）',
            ha='left', va='center', fontsize=8, color=COLORS['gray'])

    # 步骤3
    ax.text(0.5, step_y - 0.55, '③ 维度重塑:', ha='left', va='center',
            fontsize=9, fontweight='bold', color=COLORS['purple'])
    ax.text(1.5, step_y - 0.55, r'$\mathbf{X}^{2D} = \mathrm{Reshape}(\mathbf{X}^{1D})$: [B, T, C] → [B, T/p, p, C] → [B, C, T/p, p]',
            ha='left', va='center', fontsize=8, color=COLORS['gray'])

    # ========== 右侧：2D 卷积优势说明 ==========
    adv_x = 3.1
    adv_y = -0.35
    adv_box = FancyBboxPatch(
        (adv_x, adv_y - 0.35), 1.2, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=COLORS['light_orange'],
        edgecolor=COLORS['orange'],
        linewidth=1
    )
    ax.add_patch(adv_box)

    ax.text(adv_x + 0.6, adv_y - 0.02, '2D卷积优势', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['orange'])
    ax.text(adv_x + 0.1, adv_y - 0.2, '• 行方向: 捕获周期间变化', ha='left', va='center',
            fontsize=8, color=COLORS['gray'])
    ax.text(adv_x + 0.1, adv_y - 0.35, '• 列方向: 捕获周期内变化', ha='left', va='center',
            fontsize=8, color=COLORS['gray'])
    ax.text(adv_x + 0.1, adv_y - 0.5, '• 多尺度: Inception模块', ha='left', va='center',
            fontsize=8, color=COLORS['gray'])
    ax.text(adv_x + 0.1, adv_y - 0.65, '• 并行: Top-k周期聚合', ha='left', va='center',
            fontsize=8, color=COLORS['gray'])

    # ========== 顶部：示例数值说明 ==========
    example_text = f'示例参数: 序列长度 T = {T}, 周期 p = {p}, 周期数 T/p = {num_periods}'
    ax.text(2.2, 1.05, example_text, ha='center', va='bottom',
            fontsize=10, color=COLORS['gray'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['gray'], linewidth=1))

    plt.tight_layout()
    return fig

def main():
    # 创建图形
    fig = create_1d_to_2d_diagram()

    # 保存路径
    save_path = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/thesis/figures/1D到2D时序转换示意图.png'

    # 保存为 300 DPI PNG
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f'图形已保存至: {save_path}')

    # 同时保存 PDF 版本（用于论文）
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'PDF版本已保存至: {pdf_path}')

    plt.close()

if __name__ == '__main__':
    main()
