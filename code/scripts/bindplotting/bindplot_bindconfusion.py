#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混淆矩阵图 / Confusion Matrix Plot
===================================

生成混淆矩阵图展示模型分类性能。
特点：
- 颜色深浅反映样本数量
- 百分比和数量双重标注
- 中英双语标签

Usage:
    python bindplot_bindconfusion.py --data results.json --output ./figures/

Author: Rural Low-Voltage Detection Project
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from bindstyle_config import (
    apply_bindstyle,
    COLORS,
    FONT_CONFIG,
    FIGURE_CONFIG,
    SAMPLE_DATA,
    create_bilingual_title,
    save_bindplot,
)


def generate_sample_confusion_matrix(precision, recall, n_samples=1000):
    """
    基于精确率和召回率生成示例混淆矩阵
    Generate sample confusion matrix based on precision and recall

    Args:
        precision: 精确率
        recall: 召回率
        n_samples: 总样本数

    Returns:
        np.ndarray: 2x2 混淆矩阵
    """
    # 假设正负样本比例为 1:1
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    # 基于召回率计算 TP 和 FN
    tp = int(n_positive * recall)
    fn = n_positive - tp

    # 基于精确率计算 FP
    # precision = TP / (TP + FP) => FP = TP * (1 - precision) / precision
    if precision > 0:
        fp = int(tp * (1 - precision) / precision)
    else:
        fp = n_negative

    fp = min(fp, n_negative)
    tn = n_negative - fp

    # 构建混淆矩阵
    confusion_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    return confusion_matrix


def load_data(data_path=None):
    """加载数据"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        # 使用第一个模型的数据生成混淆矩阵
        sample_data = SAMPLE_DATA.copy()
        sample_data['confusion_matrix'] = generate_sample_confusion_matrix(
            SAMPLE_DATA['precision'][0],
            SAMPLE_DATA['recall'][0],
            n_samples=2000
        ).tolist()
        sample_data['model_name'] = SAMPLE_DATA['models'][0]
        return sample_data


def plot_confusion_matrix(confusion_matrix, model_name=None, output_dir=None,
                          filename='confusion_matrix'):
    """
    绘制混淆矩阵图
    Plot confusion matrix

    Args:
        confusion_matrix: 混淆矩阵 (2x2 或更大)
        model_name: 模型名称（可选）
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 应用统一样式
    apply_bindstyle()

    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]

    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum() * 100

    # 类别标签（二分类）
    if n_classes == 2:
        class_labels = ['正常\nNormal', '异常\nAnomaly']
    else:
        class_labels = [f'类别 {i}\nClass {i}' for i in range(n_classes)]

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['square'])

    # 创建颜色映射
    colors_list = ['#fff5eb', '#fdd0a2', '#fd8d3c', '#d94801']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_orange', colors_list)

    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('样本数量 / Count', fontsize=FONT_CONFIG['label_size'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_size'])

    # 设置轴刻度
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels, fontsize=FONT_CONFIG['label_size'])
    ax.set_yticklabels(class_labels, fontsize=FONT_CONFIG['label_size'])

    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            percent = cm_percent[i, j]

            # 选择文字颜色
            text_color = 'white' if count > thresh else 'black'

            # 显示数量和百分比
            ax.text(
                j, i,
                f'{count}\n({percent:.1f}%)',
                ha='center',
                va='center',
                color=text_color,
                fontsize=FONT_CONFIG['annotation_size'] + 2,
                fontweight='bold'
            )

    # 设置标题
    if model_name:
        title = create_bilingual_title(
            f'{model_name} 混淆矩阵',
            f'{model_name} Confusion Matrix'
        )
    else:
        title = create_bilingual_title(
            '混淆矩阵',
            'Confusion Matrix'
        )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 设置轴标签
    ax.set_xlabel('预测标签 / Predicted Label', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('真实标签 / True Label', fontsize=FONT_CONFIG['label_size'])

    # 添加性能指标标注
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = (
            f'准确率/Acc: {accuracy:.3f}\n'
            f'精确率/Prec: {precision:.3f}\n'
            f'召回率/Rec: {recall:.3f}\n'
            f'F1: {f1:.3f}'
        )

        ax.text(
            1.02, 0.5, metrics_text,
            transform=ax.transAxes,
            fontsize=FONT_CONFIG['annotation_size'],
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        )

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if output_dir:
        save_bindplot(fig, filename, output_dir)
    else:
        save_bindplot(fig, filename)

    plt.close(fig)

    return fig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成混淆矩阵图 / Generate Confusion Matrix Plot'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='JSON 数据文件路径 / Path to JSON data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./figures',
        help='输出目录 / Output directory'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='confusion_matrix',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称 / Model name'
    )

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)

    # 获取混淆矩阵
    confusion_matrix = data.get('confusion_matrix')
    if confusion_matrix is None:
        # 如果没有混淆矩阵，基于精确率和召回率生成
        confusion_matrix = generate_sample_confusion_matrix(
            data.get('precision', SAMPLE_DATA['precision'])[0],
            data.get('recall', SAMPLE_DATA['recall'])[0]
        )

    model_name = args.model or data.get('model_name')

    # 绘制图表
    print("正在生成混淆矩阵图...")
    if model_name:
        print(f"模型: {model_name}")

    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        model_name=model_name,
        output_dir=args.output,
        filename=args.filename
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
