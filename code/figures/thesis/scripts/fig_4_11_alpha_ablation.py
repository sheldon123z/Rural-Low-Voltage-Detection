#!/usr/bin/env python3
"""
图4-11: Alpha参数消融实验
Fig 4-11: Alpha Parameter Ablation Study

输出文件: fig_4_11_alpha_ablation.pdf, fig_4_11_alpha_ablation.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 论文格式配置
try:
    matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
except:
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

# 使用中文字体路径
zh_font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
if os.path.exists(zh_font_path):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(zh_font_path)
    matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei']

# Alpha参数消融实验数据
alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
alpha_f1_scores = [0.9708, 0.9712, 0.9714, 0.9711, 0.9706]
baseline_f1 = 0.9704  # TimesNet (FFT only)


def main():
    """主函数"""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(alpha_values, alpha_f1_scores, 'o-', color='#d62728',
            linewidth=1.5, markersize=6, label='VoltageTimesNet')

    ax.axhline(y=baseline_f1, color='#1f77b4', linestyle='--',
               linewidth=1.5, label=f'TimesNet基线 ({baseline_f1:.4f})')

    # 标注最优alpha
    best_idx = np.argmax(alpha_f1_scores)
    best_alpha = alpha_values[best_idx]
    best_f1 = alpha_f1_scores[best_idx]
    ax.scatter([best_alpha], [best_f1], s=100, color='#d62728',
               marker='*', zorder=5, edgecolors='black', linewidths=0.5)
    ax.annotate(f'最优: α={best_alpha}, F1={best_f1:.4f}',
                xy=(best_alpha, best_f1),
                xytext=(best_alpha+0.08, best_f1+0.002),
                fontsize=9, color='#d62728')

    ax.set_xlabel('融合权重 α (FFT权重)', fontsize=11)
    ax.set_ylabel('F1分数 (F1-score)', fontsize=11)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.968, 0.974)
    ax.set_xticks(alpha_values)
    ax.legend(loc='lower left', fontsize=10, frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加说明文字
    ax.text(0.70, 0.9686, 'α: FFT周期发现权重\n1-α: 预设周期权重',
            fontsize=8, color='gray', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))

    plt.tight_layout()

    # 保存
    plt.savefig('fig_4_11_alpha_ablation.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('fig_4_11_alpha_ablation.png', dpi=300, bbox_inches='tight')
    print("已生成: fig_4_11_alpha_ablation.pdf/png")
    plt.close()


if __name__ == '__main__':
    main()
