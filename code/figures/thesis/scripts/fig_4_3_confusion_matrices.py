#!/usr/bin/env python3
"""
图4-3: 混淆矩阵热力图（5个独立子图）
Fig 4-3: Confusion Matrix for Each Model (Individual Files)

5个模型各自输出一张独立的混淆矩阵图片：
  (a) VoltageTimesNet    -> fig_4_3a_cm_voltagetimesnet.png
  (b) TimesNet           -> fig_4_3b_cm_timesnet.png
  (c) LSTMAutoEncoder    -> fig_4_3c_cm_lstmae.png
  (d) Isolation Forest   -> fig_4_3d_cm_iforest.png
  (e) One-Class SVM      -> fig_4_3e_cm_ocsvm.png

数据集: RuralVoltage realistic_v2
测试集: 10000 样本, 异常比例 14.6% (1460 异常, 8540 正常)

反算逻辑:
  TP = recall * n_anomaly
  FP = TP * (1/precision - 1)
  FN = n_anomaly - TP
  TN = n_normal - FP

输出目录: ../chapter4_experiments/
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from thesis_style import setup_thesis_style, save_thesis_figure

# ============================================================
# 初始化论文样式
# ============================================================
setup_thesis_style()

# ============================================================
# RuralVoltage 实验数据 (5个模型，移除 DLinear)
# ============================================================
models_data = {
    'VoltageTimesNet': {'accuracy': 0.9393, 'precision': 0.7371, 'recall': 0.9110, 'f1': 0.8149},
    'TimesNet':        {'accuracy': 0.8584, 'precision': 0.5143, 'recall': 0.7115, 'f1': 0.5970},
    'LSTMAutoEncoder': {'accuracy': 0.7905, 'precision': 0.3654, 'recall': 0.5712, 'f1': 0.4457},
    'Isolation Forest': {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
    'One-Class SVM':    {'accuracy': 0.3474, 'precision': 0.3474, 'recall': 1.0000, 'f1': 0.5157},
}

# 模型键名与输出文件名的映射
model_output = [
    ('VoltageTimesNet',  'fig_4_3a_cm_voltagetimesnet.png'),
    ('TimesNet',         'fig_4_3b_cm_timesnet.png'),
    ('LSTMAutoEncoder',  'fig_4_3c_cm_lstmae.png'),
    ('Isolation Forest', 'fig_4_3d_cm_iforest.png'),
    ('One-Class SVM',    'fig_4_3e_cm_ocsvm.png'),
]

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


def plot_single_confusion_matrix(model_key, output_path):
    """绘制单个模型的混淆矩阵并保存

    Args:
        model_key: 模型名称（models_data 的键）
        output_path: 输出文件的完整路径
    """
    d = models_data[model_key]
    cm = compute_confusion_matrix(d['precision'], d['recall'])
    cm_normalized = cm.astype('float') / cm.sum()

    fig, ax = plt.subplots(figsize=(4, 3.5))

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
    ax.set_xlabel('预测标签/Predicted', fontsize=10.5)
    ax.set_ylabel('真实标签/Actual', fontsize=10.5)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('归一化比例', fontsize=9)

    save_thesis_figure(fig, output_path)


def main():
    """主函数：为每个模型生成独立的混淆矩阵图片"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'chapter4_experiments')
    os.makedirs(output_dir, exist_ok=True)

    for model_key, filename in model_output:
        output_path = os.path.join(output_dir, filename)
        plot_single_confusion_matrix(model_key, output_path)

    print(f"共生成 {len(model_output)} 张混淆矩阵图片")


if __name__ == '__main__':
    main()
