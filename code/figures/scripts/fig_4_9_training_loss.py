#!/usr/bin/env python3
"""
图4-9: 训练损失曲线对比
Fig 4-9: Training Loss Comparison

模型列表 (RuralVoltage 数据集):
  - VoltageTimesNet (Optuna 优化, lr=0.004, batch_size=32, d_model=128)
  - TimesNet (基线)
  - LSTMAutoEncoder (传统深度学习基线)

训练配置: 5 epochs (Optuna 快速训练), MSE 损失函数

输出文件: ../chapter4_experiments/fig_4_9_training_loss.png
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
import numpy as np

# 导入论文统一样式
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, get_output_dir, TIMESNET_COLORS, OTHER_COLORS

setup_thesis_style()


def main():
    """主函数"""
    # ============================================================
    # 训练损失数据 (RuralVoltage 数据集, 5 epochs)
    # ============================================================
    epochs = np.arange(1, 6)

    # VoltageTimesNet: Optuna 优化参数, 快速收敛到最低损失
    vtv2_loss = [0.150, 0.045, 0.032, 0.028, 0.025]

    # TimesNet: 标准参数, 收敛较慢, 最终损失略高
    timesnet_loss = [0.220, 0.065, 0.048, 0.042, 0.038]

    # LSTMAutoEncoder: RNN 模型, 初始损失高, 收敛慢
    lstm_loss = [0.350, 0.120, 0.085, 0.072, 0.065]

    # ============================================================
    # 模型配色与标记
    # ============================================================
    models = [
        ('VoltageTimesNet', vtv2_loss, TIMESNET_COLORS[1], 's', '-'),   # 柔和绿, 方块
        ('TimesNet',        timesnet_loss, TIMESNET_COLORS[0], 'o', '-'),  # 柔和蓝, 圆形
        ('LSTMAutoEncoder', lstm_loss, OTHER_COLORS[1], 'd', '--'),       # 暗金, 菱形
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
    ax.set_ylim(0, 0.38)
    ax.set_xticks(epochs)

    ax.legend(loc='upper right', fontsize=9, frameon=True,
              edgecolor='#CCCCCC', fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    remove_spines(ax)

    # 在最优模型末尾标注最终损失值
    ax.annotate(f'{vtv2_loss[-1]:.3f}',
                xy=(5, vtv2_loss[-1]),
                xytext=(5.15, vtv2_loss[-1] + 0.008),
                fontsize=8.5, color=TIMESNET_COLORS[1],
                ha='left', va='bottom')

    # ============================================================
    # 保存
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_9_training_loss.png')

    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
