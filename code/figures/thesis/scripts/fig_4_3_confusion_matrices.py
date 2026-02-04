#!/usr/bin/env python3
"""
图4-3: 混淆矩阵热力图（2x3布局）
Fig 4-3: Confusion Matrix Comparison of Multiple Models

包含6个子图，分别展示6个模型的混淆矩阵

数据集: RuralVoltage realistic_v2
测试集: 10000 样本, 异常比例 14.6% (1460 异常, 8540 正常)

反算逻辑:
  TP = recall * n_anomaly
  FP = TP * (1/precision - 1)
  FN = n_anomaly - TP
  TN = n_normal - FP

输出文件: ../chapter4_experiments/fig_4_3_confusion_matrices.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

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

# ============================================================
# RuralVoltage 实验数据 (6个模型)
# ============================================================
models_data = {
    'VoltageTimesNet_v2': {'accuracy': 0.9393, 'precision': 0.7371, 'recall': 0.9110, 'f1': 0.8149},
    'DLinear':            {'accuracy': 0.8651, 'precision': 0.5224, 'recall': 0.9955, 'f1': 0.6852},
    'TimesNet':           {'accuracy': 0.8584, 'precision': 0.5143, 'recall': 0.7115, 'f1': 0.5970},
    'LSTMAutoEncoder':    {'accuracy': 0.7905, 'precision': 0.3654, 'recall': 0.5712, 'f1': 0.4457},
    'Isolation Forest':   {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
    'One-Class SVM':      {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
}

# 模型显示顺序与标签
model_keys = ['VoltageTimesNet_v2', 'DLinear', 'TimesNet',
              'LSTMAutoEncoder', 'Isolation Forest', 'One-Class SVM']
model_labels = ['V-TimesNet_v2', 'DLinear', 'TimesNet',
                'LSTM-AE', 'iForest', 'OC-SVM']

# 测试集参数
TOTAL_SAMPLES = 10000
N_ANOMALY = 1460     # 14.6%
N_NORMAL = 8540


def compute_confusion_matrix(prec, rec):
    """根据 precision 和 recall 反算混淆矩阵

    Args:
        prec: 精确率 = TP / (TP + FP)
        rec:  召回率 = TP / (TP + FN) = TP / N_ANOMALY

    Returns:
        cm: 2x2 混淆矩阵 [[TN, FP], [FN, TP]]
    """
    TP = int(round(rec * N_ANOMALY))
    FN = N_ANOMALY - TP

    # FP = TP * (1/precision - 1)
    if prec > 0:
        FP = int(round(TP * (1.0 / prec - 1.0)))
    else:
        FP = N_NORMAL

    FP = max(0, min(FP, N_NORMAL))
    TN = N_NORMAL - FP

    return np.array([[TN, FP], [FN, TP]])


def main():
    """主函数"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    im = None  # 用于最后创建统一颜色条

    for idx, (ax, key) in enumerate(zip(axes, model_keys)):
        d = models_data[key]
        cm = compute_confusion_matrix(d['precision'], d['recall'])
        cm_normalized = cm.astype('float') / cm.sum()

        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=0.85)

        # 在每个格子中标注数值和百分比
        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm_normalized[i, j] > 0.4 else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                        ha='center', va='center', fontsize=9, color=text_color,
                        fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['正常', '异常'], fontsize=9)
        ax.set_yticklabels(['正常', '异常'], fontsize=9)
        ax.set_xlabel('预测标签', fontsize=9)
        ax.set_ylabel('真实标签', fontsize=9)

        # 子图标签: (a) V-TimesNet_v2  等
        title_weight = 'bold'
        ax.text(0.02, 1.08, f'({chr(97 + idx)}) {model_labels[idx]}',
                transform=ax.transAxes, fontsize=10, fontweight=title_weight)

    # 添加统一颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('归一化比例', fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # 保存到 chapter4_experiments 目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'chapter4_experiments')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_3_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
