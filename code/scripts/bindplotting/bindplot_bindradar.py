#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雷达图 / Radar Chart
====================

生成雷达图展示各模型在多维度指标上的表现。
特点：
- 多模型多维度对比
- 填充区域透明度区分
- 中英双语标签

Usage:
    python bindplot_bindradar.py --data results.json --output ./figures/

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


def plot_radar(models, metrics_dict, output_dir=None, filename='radar_chart',
               top_n=5):
    """
    绘制雷达图
    Plot radar chart for model comparison

    Args:
        models: 模型名称列表
        metrics_dict: 指标字典 {指标名: [分数列表]}
        output_dir: 输出目录
        filename: 输出文件名
        top_n: 显示前 n 个模型
    """
    # 应用统一样式
    apply_bindstyle()

    # 指标中英文映射
    metric_labels = {
        'f1_scores': 'F1',
        'precision': '精确率\nPrecision',
        'recall': '召回率\nRecall',
        'accuracy': '准确率\nAccuracy',
    }

    # 准备数据
    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)

    # 计算综合分数并排序
    avg_scores = []
    for i in range(len(models)):
        scores = [metrics_dict[m][i] for m in metric_names]
        avg_scores.append(np.mean(scores))

    sorted_indices = np.argsort(avg_scores)[::-1][:top_n]
    selected_models = [models[i] for i in sorted_indices]

    # 设置雷达图角度
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['square'], subplot_kw=dict(polar=True))

    # 绘制每个模型
    for idx, model_idx in enumerate(sorted_indices):
        values = [metrics_dict[m][model_idx] for m in metric_names]
        values += values[:1]  # 闭合图形

        model_name = models[model_idx]

        # 第一名使用特殊颜色
        if idx == 0:
            color = COLORS['best']
            linewidth = 2.5
            alpha_fill = 0.25
            alpha_line = 1.0
        else:
            color = COLOR_PALETTE[idx]
            linewidth = 1.8
            alpha_fill = 0.1
            alpha_line = 0.8

        # 绘制线条
        ax.plot(
            angles, values,
            'o-',
            linewidth=linewidth,
            label=model_name,
            color=color,
            alpha=alpha_line,
            markersize=6
        )

        # 填充区域
        ax.fill(angles, values, alpha=alpha_fill, color=color)

    # 设置角度标签
    labels = [metric_labels.get(m, m) for m in metric_names]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_CONFIG['label_size'])

    # 设置径向范围
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.0'],
                       fontsize=FONT_CONFIG['tick_size'] - 1)

    # 设置网格样式
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置标题
    title = create_bilingual_title(
        '模型性能雷达图',
        'Model Performance Radar Chart'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold',
                 pad=20, y=1.08)

    # 添加图例
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.0),
        fontsize=FONT_CONFIG['legend_size'],
        framealpha=0.9
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
        description='生成雷达图 / Generate Radar Chart'
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
        default='radar_chart',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='显示前 n 个模型 / Show top n models'
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
    print("正在生成雷达图...")
    print(f"显示前 {args.top_n} 个模型")

    plot_radar(
        models=models,
        metrics_dict=metrics_dict,
        output_dir=args.output,
        filename=args.filename,
        top_n=args.top_n
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
