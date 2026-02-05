#!/usr/bin/env python3
"""
图4-14: TPATimesNet三相注意力权重热力图
Fig 4-14: Three-Phase Attention Weight Heatmap of TPATimesNet

展示TPATimesNet学习到的三相电压 (Va, Vb, Vc) 之间的注意力权重矩阵。
对角线为自注意力 (1.000), 非对角线反映相间关联强度。

输出文件: ../chapter4_experiments/fig_4_14_phase_attention_heatmap.png
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
    # 注意力权重矩阵 (3x3, 学习后的值)
    # ============================================================
    attention_matrix = np.array([
        [1.000, 0.623, 0.587],   # Va -> Va, Vb, Vc
        [0.618, 1.000, 0.645],   # Vb -> Va, Vb, Vc
        [0.592, 0.651, 1.000],   # Vc -> Va, Vb, Vc
    ])

    phase_labels = ['$V_a$', '$V_b$', '$V_c$']

    # ============================================================
    # 绘制热力图
    # ============================================================
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(attention_matrix, cmap='YlOrRd',
                   vmin=0.5, vmax=1.0, aspect='equal')

    # 在每个格子内标注数值
    for i in range(3):
        for j in range(3):
            value = attention_matrix[i, j]
            # 高值用白色文字, 低值用深色文字
            text_color = 'white' if value > 0.85 else '#333333'
            ax.text(j, i, f'{value:.3f}',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color=text_color)

    # 坐标轴标签
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(phase_labels, fontsize=11)
    ax.set_yticklabels(phase_labels, fontsize=11)

    # 将X轴标签放在底部
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # 添加 colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('注意力权重', fontsize=10.5)

    # 移除边框 (热力图保留所有边框不合适, 直接隐藏刻度线)
    ax.tick_params(length=0)

    # ============================================================
    # 保存
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = get_output_dir(4)
    
    output_path = os.path.join(output_dir, 'fig_4_14_phase_attention_heatmap.png')

    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
