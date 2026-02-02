#!/usr/bin/env python3
"""
图4-9: 训练损失曲线对比
Fig 4-9: Training Loss Comparison

输出文件: ../chapter4_experiments/fig_4_9_training_loss.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================================
# 论文格式配置：中文宋体 + 英文 Times New Roman，五号字 (10.5pt)
# ============================================================
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman']
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# 训练损失数据
epochs = np.arange(1, 11)

timesnet_train_loss = [0.2464719, 0.0499441, 0.0398960, 0.0368008, 0.0354152,
                       0.0346950, 0.0343230, 0.0340988, 0.0339781, 0.0338913]

voltage_timesnet_train_loss = [0.2316292, 0.0485549, 0.0390418, 0.0361695, 0.0348628,
                               0.0341867, 0.0337991, 0.0335669, 0.0334822, 0.0334200]

dlinear_train_loss = [0.1856, 0.0523, 0.0425, 0.0398, 0.0385,
                      0.0378, 0.0374, 0.0371, 0.0369, 0.0368]

tpatimesnet_train_loss = [0.2589, 0.0542, 0.0421, 0.0382, 0.0365,
                          0.0355, 0.0349, 0.0346, 0.0344, 0.0343]


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 柔和科研配色
    ax.plot(epochs, timesnet_train_loss, 'o-', color='#4878A8',
            linewidth=1.5, markersize=4, label='TimesNet')
    ax.plot(epochs, voltage_timesnet_train_loss, 's-', color='#72A86D',
            linewidth=1.5, markersize=4, label='VoltageTimesNet')
    ax.plot(epochs, dlinear_train_loss, '^-', color='#808080',
            linewidth=1.5, markersize=4, label='DLinear')
    ax.plot(epochs, tpatimesnet_train_loss, 'd-', color='#9B7BB8',
            linewidth=1.5, markersize=4, label='TPATimesNet')

    ax.set_xlabel('训练轮次/Epoch', fontsize=10.5)
    ax.set_ylabel('训练损失', fontsize=10.5)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.30)
    ax.set_xticks(epochs)
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_9_training_loss.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_9_training_loss.png")
    plt.close()


if __name__ == '__main__':
    main()
