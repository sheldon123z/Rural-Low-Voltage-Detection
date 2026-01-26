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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
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
    'gray': '#666666',
    'light_blue': '#D6E3F8',
    'light_orange': '#FDE5D4',
    'light_green': '#E2F0D9',
    'light_gray': '#E0E0E0',
    'white': '#FFFFFF',
    'dark_blue': '#2F5597',
    'dark_orange': '#C65911',
}


def create_1d_to_2d_diagram():
    """创建完整的 1D→2D 转换示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-1.0, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # 参数设置
    T = 100  # 序列长度
    p = 10   # 周期
    num_periods = T // p  # 周期数 = 10

    # ==================== 第一部分：1D 时间序列 ====================
    seq_x, seq_y = 0, 0.65
    seq_width, seq_height = 1.4, 0.25

    # 绘制 1D 序列（分段显示）
    num_segments = 10
    segment_width = seq_width / num_segments
    for i in range(num_segments):
        # 使用渐变色表示不同时间段
        intensity = 0.3 + 0.7 * (i / num_segments)
        color = plt.cm.Blues(intensity)
        rect = Rectangle(
            (seq_x + i * segment_width, seq_y - seq_height/2),
            segment_width, seq_height,
            facecolor=color,
            edgecolor=COLORS['dark_blue'],
            linewidth=0.8
        )
        ax.add_patch(rect)

    # 外框加粗
    outer_rect = Rectangle(
        (seq_x, seq_y - seq_height/2),
        seq_width, seq_height,
        facecolor='none',
        edgecolor=COLORS['dark_blue'],
        linewidth=2
    )
    ax.add_patch(outer_rect)

    # 时间轴标注
    ax.annotate('', xy=(seq_x + seq_width, seq_y - seq_height/2 - 0.08),
                xytext=(seq_x, seq_y - seq_height/2 - 0.08),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.2))
    ax.text(seq_x + seq_width/2, seq_y - seq_height/2 - 0.18, f'T = {T}',
            ha='center', va='top', fontsize=10, color=COLORS['gray'], fontweight='bold')

    # 通道标注
    ax.annotate('', xy=(seq_x - 0.06, seq_y + seq_height/2),
                xytext=(seq_x - 0.06, seq_y - seq_height/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.2))
    ax.text(seq_x - 0.12, seq_y, 'C',
            ha='right', va='center', fontsize=10, color=COLORS['gray'], fontweight='bold')

    # 张量形状标注
    ax.text(seq_x + seq_width/2, seq_y + seq_height/2 + 0.1,
            r'$\mathbf{X}^{1D}$: [B, T, C]',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['dark_blue'])

    # ==================== 第二部分：FFT 周期发现框 ====================
    fft_x, fft_y = 1.75, 0.5
    fft_width, fft_height = 0.65, 0.55

    # FFT 框
    fft_box = FancyBboxPatch(
        (fft_x, fft_y), fft_width, fft_height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=COLORS['light_green'],
        edgecolor=COLORS['green'],
        linewidth=2
    )
    ax.add_patch(fft_box)

    ax.text(fft_x + fft_width/2, fft_y + fft_height - 0.12, 'FFT',
            ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['green'])
    ax.text(fft_x + fft_width/2, fft_y + fft_height/2 - 0.02, '周期发现',
            ha='center', va='center', fontsize=10, color=COLORS['green'])
    ax.text(fft_x + fft_width/2, fft_y + 0.12, f'p = {p}',
            ha='center', va='center', fontsize=10, color=COLORS['gray'], fontweight='bold')

    # 箭头 1D → FFT
    ax.annotate('', xy=(fft_x - 0.02, seq_y),
                xytext=(seq_x + seq_width + 0.05, seq_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                               lw=2, connectionstyle='arc3,rad=0'))

    # ==================== 第三部分：Reshape 操作框 ====================
    reshape_x, reshape_y = 2.7, 0.42
    reshape_width, reshape_height = 0.6, 0.7

    # Reshape 框
    reshape_box = FancyBboxPatch(
        (reshape_x, reshape_y), reshape_width, reshape_height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor='#F3E5F5',
        edgecolor=COLORS['purple'],
        linewidth=2
    )
    ax.add_patch(reshape_box)

    ax.text(reshape_x + reshape_width/2, reshape_y + reshape_height - 0.12, 'Reshape',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['purple'])
    ax.text(reshape_x + reshape_width/2, reshape_y + reshape_height/2, '1D → 2D',
            ha='center', va='center', fontsize=10, color=COLORS['purple'])

    # 维度转换说明
    ax.text(reshape_x + reshape_width/2, reshape_y + 0.22, f'[B,{T},C]',
            ha='center', va='center', fontsize=8, color=COLORS['gray'])
    ax.text(reshape_x + reshape_width/2, reshape_y + 0.12, '↓',
            ha='center', va='center', fontsize=12, color=COLORS['purple'], fontweight='bold')
    ax.text(reshape_x + reshape_width/2, reshape_y + 0.04, f'[B,C,{num_periods},{p}]',
            ha='center', va='center', fontsize=8, color=COLORS['gray'])

    # 箭头 FFT → Reshape
    ax.annotate('', xy=(reshape_x - 0.02, fft_y + fft_height/2),
                xytext=(fft_x + fft_width + 0.02, fft_y + fft_height/2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                               lw=2, connectionstyle='arc3,rad=0'))

    # ==================== 第四部分：2D 张量（核心可视化） ====================
    tensor_x, tensor_y = 3.7, 0.25
    grid_size = 0.9
    cell_size = grid_size / num_periods

    # 绘制 2D 网格（10×10）
    for i in range(num_periods):
        for j in range(p):
            # 使用渐变色，展示数据填充顺序
            idx = i * p + j
            intensity = 0.2 + 0.8 * (idx / (T - 1))
            color = plt.cm.Oranges(intensity)

            rect = Rectangle(
                (tensor_x + j * cell_size, tensor_y + (num_periods - 1 - i) * cell_size),
                cell_size, cell_size,
                facecolor=color,
                edgecolor=COLORS['dark_orange'],
                linewidth=0.6
            )
            ax.add_patch(rect)

    # 外框加粗
    outer_rect = Rectangle(
        (tensor_x, tensor_y),
        grid_size, grid_size,
        facecolor='none',
        edgecolor=COLORS['dark_orange'],
        linewidth=2.5
    )
    ax.add_patch(outer_rect)

    # 维度标注 - 水平方向（周期内：intra-period）
    ax.annotate('', xy=(tensor_x + grid_size, tensor_y - 0.1),
                xytext=(tensor_x, tensor_y - 0.1),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.2))
    ax.text(tensor_x + grid_size/2, tensor_y - 0.2, f'p = {p}',
            ha='center', va='top', fontsize=10, color=COLORS['gray'], fontweight='bold')
    ax.text(tensor_x + grid_size/2, tensor_y - 0.35, '(周期内变化)',
            ha='center', va='top', fontsize=9, color=COLORS['dark_orange'])

    # 维度标注 - 垂直方向（周期间：inter-period）
    ax.annotate('', xy=(tensor_x - 0.1, tensor_y + grid_size),
                xytext=(tensor_x - 0.1, tensor_y),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.2))
    ax.text(tensor_x - 0.2, tensor_y + grid_size/2, f'T/p\n= {num_periods}',
            ha='right', va='center', fontsize=10, color=COLORS['gray'], fontweight='bold')
    ax.text(tensor_x - 0.45, tensor_y + grid_size/2, '(周期间\n 变化)',
            ha='right', va='center', fontsize=9, color=COLORS['dark_orange'])

    # 张量形状标注
    ax.text(tensor_x + grid_size/2, tensor_y + grid_size + 0.12,
            r'$\mathbf{X}^{2D}$: [B, C, T/p, p]',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['dark_orange'])

    # 箭头 Reshape → 2D
    ax.annotate('', xy=(tensor_x - 0.02, reshape_y + reshape_height/2),
                xytext=(reshape_x + reshape_width + 0.02, reshape_y + reshape_height/2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                               lw=2, connectionstyle='arc3,rad=0'))

    # ==================== 第五部分：2D 卷积说明框 ====================
    conv_x, conv_y = 4.8, 0.4
    conv_width, conv_height = 0.6, 0.7

    conv_box = FancyBboxPatch(
        (conv_x, conv_y), conv_width, conv_height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=COLORS['light_orange'],
        edgecolor=COLORS['dark_orange'],
        linewidth=2
    )
    ax.add_patch(conv_box)

    ax.text(conv_x + conv_width/2, conv_y + conv_height - 0.1, '2D Conv',
            ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['dark_orange'])
    ax.text(conv_x + conv_width/2, conv_y + conv_height/2 + 0.05, 'Inception',
            ha='center', va='center', fontsize=9, color=COLORS['dark_orange'])
    ax.text(conv_x + conv_width/2, conv_y + conv_height/2 - 0.1, '多尺度卷积',
            ha='center', va='center', fontsize=8, color=COLORS['gray'])
    ax.text(conv_x + conv_width/2, conv_y + 0.12, '(1×1,3×3,5×5)',
            ha='center', va='center', fontsize=7, color=COLORS['gray'])

    # 箭头 2D → Conv
    ax.annotate('', xy=(conv_x - 0.02, tensor_y + grid_size/2 + 0.15),
                xytext=(tensor_x + grid_size + 0.05, tensor_y + grid_size/2 + 0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                               lw=2, connectionstyle='arc3,rad=0'))

    # ==================== 底部：详细步骤说明 ====================
    step_y = -0.25
    step_box = FancyBboxPatch(
        (0, step_y - 0.55), 5.3, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor='#FAFAFA',
        edgecolor=COLORS['gray'],
        linewidth=1,
        linestyle='--'
    )
    ax.add_patch(step_box)

    # 步骤标题
    ax.text(2.65, step_y - 0.02, '转换过程详解',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['gray'])

    # 步骤内容
    step_texts = [
        ('① FFT周期发现:', r'$\mathbf{X}_f = \mathrm{FFT}(\mathbf{X}^{1D})$，提取幅值最大的频率分量，计算对应周期 $p$', COLORS['green']),
        ('② Padding对齐:', f'若 T 不能被 p 整除，零填充至 T\' = ⌈T/p⌉ × p（本例: {T} 可被 {p} 整除，无需填充）', COLORS['gray']),
        ('③ 维度重塑:', r'$\mathbf{X}^{2D} = \mathrm{Reshape}(\mathbf{X}^{1D})$: [B, T, C] → [B, T/p, p, C] → [B, C, T/p, p]', COLORS['purple']),
    ]

    for i, (label, content, color) in enumerate(step_texts):
        y_pos = step_y - 0.18 - i * 0.16
        ax.text(0.15, y_pos, label, ha='left', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(1.25, y_pos, content, ha='left', va='center',
                fontsize=8, color=COLORS['gray'])

    # ==================== 顶部：参数说明 ====================
    param_text = f'示例参数: 序列长度 T = {T}, 发现周期 p = {p}, 周期数 T/p = {num_periods}'
    param_box = FancyBboxPatch(
        (1.2, 1.35), 2.9, 0.2,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor='white',
        edgecolor=COLORS['gray'],
        linewidth=1
    )
    ax.add_patch(param_box)
    ax.text(2.65, 1.45, param_text,
            ha='center', va='center', fontsize=10, color=COLORS['gray'])

    # ==================== 添加数据填充顺序示意 ====================
    # 在 1D 序列上标注起止
    ax.text(seq_x + 0.02, seq_y, '0', ha='left', va='center',
            fontsize=7, color='white', fontweight='bold')
    ax.text(seq_x + seq_width - 0.08, seq_y, f'{T-1}', ha='right', va='center',
            fontsize=7, color='white', fontweight='bold')

    # 在 2D 张量上标注填充方向
    # 起点标注
    ax.text(tensor_x + cell_size/2, tensor_y + grid_size - cell_size/2, '0',
            ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    # 终点标注
    ax.text(tensor_x + grid_size - cell_size/2, tensor_y + cell_size/2, f'{T-1}',
            ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    # 添加行填充方向箭头（小箭头）
    arrow_y = tensor_y + grid_size - cell_size/2
    ax.annotate('', xy=(tensor_x + grid_size - 0.02, arrow_y),
                xytext=(tensor_x + cell_size, arrow_y),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

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
