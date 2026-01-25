#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能热力图 / Performance Heatmap
================================

生成热力图展示各模型在不同指标上的性能表现。
特点：
- 颜色深浅直观反映性能高低
- 数值标注
- 中英双语标签

Usage:
    python bindplot_bindheatmap.py --data results.json --output ./figures/

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


def load_data(data_path=None):
    """加载数据"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        return SAMPLE_DATA


def plot_heatmap(models, metrics_dict, output_dir=None, filename='performance_heatmap'):
    """
    绘制性能热力图
    Plot performance heatmap

    Args:
        models: 模型名称列表
        metrics_dict: 指标字典 {指标名: [分数列表]}
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 应用统一样式
    apply_bindstyle()

    # 指标中英文映射
    metric_labels = {
        'f1_scores': 'F1',
        'precision': '精确率 / Prec',
        'recall': '召回率 / Rec',
        'accuracy': '准确率 / Acc',
    }

    # 准备数据矩阵
    metric_names = list(metrics_dict.keys())
    n_models = len(models)
    n_metrics = len(metric_names)

    # 创建数据矩阵 (模型 x 指标)
    data_matrix = np.array([metrics_dict[m] for m in metric_names]).T

    # 按 F1 分数排序
    if 'f1_scores' in metrics_dict:
        sorted_indices = np.argsort(metrics_dict['f1_scores'])[::-1]
    else:
        sorted_indices = np.arange(n_models)

    sorted_models = [models[i] for i in sorted_indices]
    sorted_matrix = data_matrix[sorted_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['wide'])

    # 创建自定义颜色映射（从浅到深的蓝绿色）
    colors_list = ['#f7fcf5', '#c7e9c0', '#74c476', '#31a354', '#006d2c']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_green', colors_list)

    # 绘制热力图
    im = ax.imshow(
        sorted_matrix,
        cmap=cmap,
        aspect='auto',
        vmin=0.7,
        vmax=1.0
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('分数 / Score', fontsize=FONT_CONFIG['label_size'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_size'])

    # 设置轴标签
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(
        [metric_labels.get(m, m) for m in metric_names],
        fontsize=FONT_CONFIG['label_size']
    )

    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(sorted_models, fontsize=FONT_CONFIG['tick_size'])

    # 添加数值标注
    for i in range(n_models):
        for j in range(n_metrics):
            value = sorted_matrix[i, j]
            # 根据背景色选择文字颜色
            text_color = 'white' if value > 0.85 else 'black'

            ax.text(
                j, i,
                f'{value:.3f}',
                ha='center',
                va='center',
                color=text_color,
                fontsize=FONT_CONFIG['annotation_size'],
                fontweight='bold' if value > 0.9 else 'normal'
            )

    # 标记最佳模型（第一行）
    for j in range(n_metrics):
        ax.add_patch(plt.Rectangle(
            (j - 0.5, -0.5), 1, 1,
            fill=False,
            edgecolor=COLORS['best'],
            linewidth=3
        ))

    # 设置标题
    title = create_bilingual_title(
        '模型性能热力图',
        'Model Performance Heatmap'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 设置轴标签
    ax.set_xlabel('性能指标 / Metrics', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('模型 / Model', fontsize=FONT_CONFIG['label_size'])

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
        description='生成性能热力图 / Generate Performance Heatmap'
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
        default='performance_heatmap',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)

    # 提取模型和指标
    models = data.get('models', SAMPLE_DATA['models'])
    metrics_dict = {
        'f1_scores': data.get('f1_scores', SAMPLE_DATA['f1_scores']),
        'precision': data.get('precision', SAMPLE_DATA['precision']),
        'recall': data.get('recall', SAMPLE_DATA['recall']),
        'accuracy': data.get('accuracy', SAMPLE_DATA['accuracy']),
    }

    # 绘制图表
    print("正在生成性能热力图...")
    print(f"模型数量: {len(models)}")
    print(f"指标数量: {len(metrics_dict)}")

    plot_heatmap(
        models=models,
        metrics_dict=metrics_dict,
        output_dir=args.output,
        filename=args.filename
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
