#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滑动窗口预测示意图
用于论文第三章方法论部分，展示时序异常检测的滑动窗口机制

Fig 3-1: 滑动窗口预测示意图
- 上半部分：原始时序数据与滑动窗口
- 中间部分：模型处理流程
- 下半部分：重构误差与异常检测
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib

# 论文格式设置 - 使用系统可用的中文字体
matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Times New Roman', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 配色方案（colorblind-friendly）
COLORS = {
    'signal': '#0072B2',      # 蓝色 - 原始信号
    'window': '#E69F00',      # 橙色 - 滑动窗口边框
    'window_fill': '#E69F00', # 橙色填充（透明）
    'recon_error': '#009E73', # 绿色 - 重构误差
    'threshold': '#D55E00',   # 红橙色 - 阈值线
    'anomaly': '#CC79A7',     # 紫红色 - 异常区域
    'model_box': '#56B4E9',   # 浅蓝色 - 模型框
    'arrow': '#333333',       # 深灰色 - 箭头
    'input_box': '#E8F4F8',   # 浅蓝灰 - 输入框
    'output_box': '#F4E8E8',  # 浅粉色 - 输出框
}


def generate_voltage_signal(n_points=500):
    """生成模拟电压信号，包含正常波动和异常"""
    np.random.seed(42)
    t = np.arange(n_points)

    # 基础信号：正弦波 + 噪声
    signal = 220 + 5 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 1, n_points)

    # 添加异常区域（电压骤降）
    anomaly_start = 350
    anomaly_end = 400
    signal[anomaly_start:anomaly_end] -= 15  # 电压骤降
    signal[anomaly_start:anomaly_end] += np.random.normal(0, 2, anomaly_end - anomaly_start)

    return t, signal, (anomaly_start, anomaly_end)


def create_sliding_window_figure():
    """创建滑动窗口预测示意图"""

    # 创建图形，使用 constrained_layout
    fig = plt.figure(figsize=(8, 6.5), constrained_layout=True)

    # 使用 GridSpec 进行灵活布局
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 0.7, 1.2], hspace=0.15)

    # ========== 上半部分：原始时序数据与滑动窗口 ==========
    ax1 = fig.add_subplot(gs[0])

    # 生成数据
    t, signal, (anomaly_start, anomaly_end) = generate_voltage_signal(500)

    # 绘制原始信号
    ax1.plot(t, signal, color=COLORS['signal'], linewidth=1.2, label='原始电压信号')

    # 滑动窗口参数
    seq_len = 100
    stride = 80
    window_positions = [50, 130, 210, 290]  # 窗口起始位置

    # 绘制滑动窗口（矩形框）
    for i, start in enumerate(window_positions):
        alpha = 0.25 if i < len(window_positions) - 1 else 0.45
        linewidth = 1.5 if i < len(window_positions) - 1 else 2.5
        linestyle = '--' if i < len(window_positions) - 1 else '-'

        # 获取窗口内信号的 y 范围
        y_min = signal[start:start+seq_len].min() - 3
        y_max = signal[start:start+seq_len].max() + 3

        rect = mpatches.Rectangle(
            (start, y_min), seq_len, y_max - y_min,
            linewidth=linewidth, edgecolor=COLORS['window'],
            facecolor=COLORS['window_fill'], alpha=alpha,
            linestyle=linestyle
        )
        ax1.add_patch(rect)

        # 标注窗口序号
        if i == len(window_positions) - 1:
            ax1.text(start + seq_len/2, y_max + 4, '窗口 $t$',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        elif i == 0:
            ax1.text(start + seq_len/2, y_max + 4, '窗口 $t-3$',
                    ha='center', va='bottom', fontsize=9, color='gray')

    # 标注滑动方向箭头
    arrow_y = 233
    ax1.annotate('', xy=(window_positions[-1] + seq_len + 20, arrow_y),
                xytext=(window_positions[0], arrow_y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                              connectionstyle='arc3,rad=0'))
    ax1.text((window_positions[0] + window_positions[-1] + seq_len + 20) / 2, arrow_y + 2,
            '滑动方向', ha='center', va='bottom', fontsize=8, color='gray')

    # 标注 seq_len 参数
    param_y = 198
    ax1.annotate('', xy=(window_positions[-1] + seq_len, param_y),
                xytext=(window_positions[-1], param_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax1.text(window_positions[-1] + seq_len/2, param_y - 3, f'$L_{{seq}}={seq_len}$',
            ha='center', va='top', fontsize=9)

    # 标注 stride 参数
    stride_y = 193
    ax1.annotate('', xy=(window_positions[1], stride_y),
                xytext=(window_positions[0], stride_y),
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=0.8))
    ax1.text((window_positions[0] + window_positions[1]) / 2, stride_y - 2,
            f'stride={stride}', ha='center', va='top', fontsize=8, color='#666666')

    ax1.set_xlim(0, 480)
    ax1.set_ylim(188, 240)
    ax1.set_xlabel('时间步', fontsize=10)
    ax1.set_ylabel('电压 (V)', fontsize=10)
    ax1.set_title('(a) 原始时序数据与滑动窗口', fontsize=11, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', frameon=False, fontsize=9)

    # ========== 中间部分：模型处理流程 ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2)
    ax2.axis('off')

    # 流程框位置
    box_y = 1.0
    box_height = 0.9
    box_width_small = 2.0
    box_width_large = 2.8

    # 输入框
    input_x = 0.8
    input_box = FancyBboxPatch((input_x, box_y - box_height/2), box_width_small, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=COLORS['input_box'], edgecolor='#666666', linewidth=1.2)
    ax2.add_patch(input_box)
    ax2.text(input_x + box_width_small/2, box_y + 0.12, '输入窗口', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(input_x + box_width_small/2, box_y - 0.22, r'$\mathbf{X} \in \mathbb{R}^{B \times L \times C}$',
            ha='center', va='center', fontsize=9)

    # TimesNet 编码器框
    model_x = 3.6
    model_box = FancyBboxPatch((model_x, box_y - box_height/2), box_width_large, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=COLORS['model_box'], edgecolor='#0066AA', linewidth=1.8)
    ax2.add_patch(model_box)
    ax2.text(model_x + box_width_large/2, box_y + 0.15, 'TimesNet', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#003366')
    ax2.text(model_x + box_width_large/2, box_y - 0.22, '编码器-解码器', ha='center', va='center',
            fontsize=9, color='#444444')

    # 输出框
    output_x = 7.2
    output_box = FancyBboxPatch((output_x, box_y - box_height/2), box_width_small, box_height,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=COLORS['output_box'], edgecolor='#666666', linewidth=1.2)
    ax2.add_patch(output_box)
    ax2.text(output_x + box_width_small/2, box_y + 0.12, '重构输出', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(output_x + box_width_small/2, box_y - 0.22, r'$\hat{\mathbf{X}} \in \mathbb{R}^{B \times L \times C}$',
            ha='center', va='center', fontsize=9)

    # 箭头
    arrow_style = dict(arrowstyle='->', color=COLORS['arrow'], lw=1.8,
                      connectionstyle='arc3,rad=0')
    ax2.annotate('', xy=(model_x - 0.15, box_y), xytext=(input_x + box_width_small + 0.15, box_y),
                arrowprops=arrow_style)
    ax2.annotate('', xy=(output_x - 0.15, box_y), xytext=(model_x + box_width_large + 0.15, box_y),
                arrowprops=arrow_style)

    ax2.set_title('(b) 模型处理流程', fontsize=11, fontweight='bold', loc='left', y=1.05)

    # ========== 下半部分：重构误差与异常检测 ==========
    ax3 = fig.add_subplot(gs[2])

    # 生成重构误差（模拟）
    np.random.seed(42)
    recon_error = np.abs(np.random.normal(0, 1, 500)) * 2

    # 异常区域的重构误差更大
    recon_error[anomaly_start:anomaly_end] = np.abs(np.random.normal(8, 2, anomaly_end - anomaly_start))

    # 平滑处理
    from scipy.ndimage import gaussian_filter1d
    recon_error = gaussian_filter1d(recon_error, sigma=3)

    # 绘制重构误差
    ax3.plot(t, recon_error, color=COLORS['recon_error'], linewidth=1.2, label='重构误差')

    # 阈值线
    threshold = 5.0
    ax3.axhline(y=threshold, color=COLORS['threshold'], linestyle='--', linewidth=1.5, label=f'阈值 $\\tau$')

    # 标注异常区域
    anomaly_mask = recon_error > threshold
    ax3.fill_between(t, 0, recon_error, where=anomaly_mask,
                    color=COLORS['anomaly'], alpha=0.35, label='检测到的异常')

    # 标注关键参数
    ax3.annotate(f'$\\tau = {threshold}$', xy=(460, threshold),
                xytext=(460, threshold + 2.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['threshold'], lw=1))

    # 添加异常区域标注
    max_error_idx = np.argmax(recon_error)
    max_error = recon_error[max_error_idx]
    ax3.annotate('异常区域', xy=(max_error_idx, max_error),
                xytext=(max_error_idx - 80, max_error + 1.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['anomaly'], lw=1))

    ax3.set_xlim(0, 480)
    ax3.set_ylim(0, 16)
    ax3.set_xlabel('时间步', fontsize=10)
    ax3.set_ylabel('重构误差', fontsize=10)
    ax3.set_title('(c) 异常检测结果', fontsize=11, fontweight='bold', loc='left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(loc='upper right', frameon=False, fontsize=9, ncol=1)

    return fig


def main():
    """主函数"""
    # 创建图形
    fig = create_sliding_window_figure()

    # 保存路径
    output_dir = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/figures/thesis'

    # 保存为 PDF（高质量矢量格式）
    pdf_path = f'{output_dir}/fig_sliding_window.pdf'
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', format='pdf')
    print(f'已保存: {pdf_path}')

    # 保存为 PNG（预览用）
    png_path = f'{output_dir}/fig_sliding_window.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f'已保存: {png_path}')

    plt.close(fig)
    print('滑动窗口预测示意图绘制完成！')


if __name__ == '__main__':
    main()
