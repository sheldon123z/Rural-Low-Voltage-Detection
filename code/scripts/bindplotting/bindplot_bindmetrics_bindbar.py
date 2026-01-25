#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能指标分组柱状图 / Performance Metrics Grouped Bar Chart
=========================================================

生成分组柱状图展示各模型的多项性能指标对比。
特点：
- 多指标分组对比（F1, Precision, Recall, Accuracy）
- Okabe-Ito 配色方案
- 中英双语标签

Usage:
    python bindplot_bindmetrics_bindbar.py --data results.json --output ./figures/

Author: Rural Low-Voltage Detection Project
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from bindstyle_config import (
    apply_bindstyle,
    COLOR_PALETTE,
    FONT_CONFIG,
    FIGURE_CONFIG,
    SAMPLE_DATA,
    create_bilingual_title,
    save_bindplot,
)


def load_data(data_path=None):
    """
    加载数据
    Load data from JSON file or use sample data
    """
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        return SAMPLE_DATA


def plot_metrics_bar(models, metrics_dict, output_dir=None, filename='metrics_comparison'):
    """
    绘制性能指标分组柱状图
    Plot performance metrics as grouped bar chart

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
        'f1_scores': ('F1 分数', 'F1 Score'),
        'precision': ('精确率', 'Precision'),
        'recall': ('召回率', 'Recall'),
        'accuracy': ('准确率', 'Accuracy'),
    }

    # 准备数据
    metric_names = list(metrics_dict.keys())
    n_models = len(models)
    n_metrics = len(metric_names)

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['wide'])

    # 设置条形位置
    x = np.arange(n_models)
    bar_width = 0.8 / n_metrics
    offsets = np.linspace(-(n_metrics - 1) / 2 * bar_width,
                          (n_metrics - 1) / 2 * bar_width,
                          n_metrics)

    # 绘制每个指标的柱状图
    for i, (metric_key, scores) in enumerate(metrics_dict.items()):
        label_zh, label_en = metric_labels.get(metric_key, (metric_key, metric_key))
        label = f"{label_zh} / {label_en}"

        bars = ax.bar(
            x + offsets[i],
            scores,
            bar_width * 0.9,
            label=label,
            color=COLOR_PALETTE[i],
            edgecolor='white',
            linewidth=0.8,
            alpha=0.9
        )

    # 设置 X 轴
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=FONT_CONFIG['tick_size'], rotation=30, ha='right')
    ax.set_xlabel('模型 / Model', fontsize=FONT_CONFIG['label_size'])

    # 设置 Y 轴
    ax.set_ylabel('分数 / Score', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylim(0, 1.05)

    # 设置标题
    title = create_bilingual_title(
        '模型性能指标对比',
        'Model Performance Metrics Comparison'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # 添加图例
    ax.legend(
        loc='upper right',
        fontsize=FONT_CONFIG['legend_size'],
        framealpha=0.9,
        ncol=2
    )

    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
        description='生成性能指标分组柱状图 / Generate Performance Metrics Grouped Bar Chart'
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
        default='metrics_comparison',
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
    print("正在生成性能指标分组柱状图...")
    print(f"模型数量: {len(models)}")
    print(f"指标数量: {len(metrics_dict)}")

    plot_metrics_bar(
        models=models,
        metrics_dict=metrics_dict,
        output_dir=args.output,
        filename=args.filename
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
