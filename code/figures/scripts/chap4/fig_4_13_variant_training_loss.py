#!/usr/bin/env python3
"""
图4-13: TimesNet变体训练损失收敛曲线对比
Fig 4-13: Training Loss Convergence Comparison of TimesNet Variants

模型列表 (RuralVoltage 数据集):
  - VoltageTimesNet (预设周期 + FFT 混合, 最快收敛)
  - TimesNet (基线)
  - TPATimesNet (三相注意力, 辅助收敛)
  - MTSTimesNet (多尺度时序, 计算量大, 收敛较慢)

训练配置: 5 epochs, MSE 损失函数

输出文件: ../chapter4_experiments/fig_4_13_variant_training_loss.png
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
import numpy as np

# 导入论文统一样式
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, TIMESNET_COLORS

setup_thesis_style()


def main():
    """主函数"""
    # ============================================================
    # 训练损失数据 (RuralVoltage 数据集, 5 epochs, MSE)
    # ============================================================
    epochs = np.arange(1, 6)

    # VoltageTimesNet: 预设周期 + FFT 混合, 最快收敛, 最低损失
    voltage_loss = [0.150, 0.045, 0.032, 0.028, 0.025]

    # TimesNet: 标准 FFT 周期发现, 收敛速度中等
    timesnet_loss = [0.220, 0.065, 0.048, 0.042, 0.038]

    # TPATimesNet: 三相注意力帮助收敛, 效果优于基线
    tpa_loss = [0.195, 0.058, 0.040, 0.035, 0.031]

    # MTSTimesNet: 多尺度计算量大, 收敛较慢
    mts_loss = [0.240, 0.078, 0.058, 0.050, 0.045]

    # ============================================================
    # 模型配色与标记
    # ============================================================
    models = [
        ('VoltageTimesNet', voltage_loss,  TIMESNET_COLORS[1], 's', '-'),   # 柔和绿, 方块, 实线
        ('TimesNet',        timesnet_loss,  TIMESNET_COLORS[0], 'o', '-'),   # 柔和蓝, 圆形, 实线
        ('TPATimesNet',     tpa_loss,       TIMESNET_COLORS[2], '^', '--'),  # 柔和紫, 三角, 虚线
        ('MTSTimesNet',     mts_loss,       TIMESNET_COLORS[3], 'd', '--'),  # 柔和橙, 菱形, 虚线
    ]

    # ============================================================
    # 绘制图表
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for name, loss, color, marker, linestyle in models:
        ax.plot(epochs, loss, marker=marker, linestyle=linestyle,
                color=color, linewidth=1.5, markersize=5, label=name,
                markeredgecolor=color, markerfacecolor='white',
                markeredgewidth=1.2)

    ax.set_xlabel('训练轮次/Epoch', fontsize=10.5)
    ax.set_ylabel('训练损失/MSE', fontsize=10.5)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 0.26)
    ax.set_xticks(epochs)

    ax.legend(loc='upper right', fontsize=9, frameon=True,
              edgecolor='#CCCCCC', fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    remove_spines(ax)

    # 在最优模型末尾标注最终损失值
    ax.annotate(f'{voltage_loss[-1]:.3f}',
                xy=(5, voltage_loss[-1]),
                xytext=(5.15, voltage_loss[-1] + 0.008),
                fontsize=8.5, color=TIMESNET_COLORS[1],
                ha='left', va='bottom')

    # ============================================================
    # 保存
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_13_variant_training_loss.png')

    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
