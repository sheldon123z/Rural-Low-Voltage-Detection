#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F1分数排名图 / F1 Score Ranking Plot
=====================================

生成水平条形图展示各模型的F1分数排名。
特点：
- 渐变色展示排名（前三名用醒目颜色）
- 数值标签显示在条形内/外
- 中英双语标题和标签

Usage:
    python bindplot_f1_bindranking.py --data results.json --output ./figures/

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
    COLORS,
    FONT_CONFIG,
    FIGURE_CONFIG,
    SAMPLE_DATA,
    create_bilingual_title,
    save_bindplot,
    get_ranked_colors,
)


def load_data(data_path=None):
    """
    加载数据
    Load data from JSON file or use sample data

    Args:
        data_path: JSON 文件路径

    Returns:
        dict: 包含 models 和 f1_scores 的字典
    """
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        return SAMPLE_DATA


def plot_f1_ranking(models, f1_scores, output_dir=None, filename='f1_score_ranking'):
    """
    绘制 F1 分数排名水平条形图
    Plot F1 score ranking as horizontal bar chart

    Args:
        models: 模型名称列表
        f1_scores: F1 分数列表
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 应用统一样式
    apply_bindstyle()

    # 按 F1 分数排序（降序）
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [f1_scores[i] for i in sorted_indices]

    # 获取排名颜色
    n_models = len(sorted_models)
    colors = get_ranked_colors(n_models, highlight_top=3)

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['wide'])

    # 绘制水平条形图
    y_pos = np.arange(n_models)
    bars = ax.barh(
        y_pos,
        sorted_scores,
        color=colors,
        edgecolor='white',
        linewidth=1.5,
        height=0.7,
        alpha=0.9
    )

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        width = bar.get_width()

        # 根据条形长度决定标签位置
        if width > 0.5:
            # 标签在条形内部
            label_x = width - 0.02
            ha = 'right'
            color = 'white'
            fontweight = 'bold'
        else:
            # 标签在条形外部
            label_x = width + 0.01
            ha = 'left'
            color = '#333333'
            fontweight = 'normal'

        # 添加排名标签
        rank_label = f"#{i+1}" if i < 3 else ""

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f'{score:.4f} {rank_label}',
            ha=ha,
            va='center',
            color=color,
            fontsize=FONT_CONFIG['annotation_size'],
            fontweight=fontweight
        )

    # 设置 Y 轴标签（模型名称）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=FONT_CONFIG['label_size'])

    # 设置 X 轴
    ax.set_xlim(0, max(sorted_scores) * 1.08)
    ax.set_xlabel('F1 分数 / F1 Score', fontsize=FONT_CONFIG['label_size'])

    # 设置标题
    title = create_bilingual_title(
        '模型 F1 分数排名对比',
        'Model F1 Score Ranking Comparison'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 添加网格（仅 X 轴）
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # 添加图例说明
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['best'], edgecolor='white', label='最佳 / Best'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['second'], edgecolor='white', label='次佳 / 2nd'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['third'], edgecolor='white', label='第三 / 3rd'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['gray'], edgecolor='white', label='其他 / Others'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=FONT_CONFIG['legend_size'],
        framealpha=0.9
    )

    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 反转 Y 轴（最高分在顶部）
    ax.invert_yaxis()

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
        description='生成 F1 分数排名图 / Generate F1 Score Ranking Plot'
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
        default='f1_score_ranking',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)

    # 提取模型和分数
    models = data.get('models', SAMPLE_DATA['models'])
    f1_scores = data.get('f1_scores', SAMPLE_DATA['f1_scores'])

    # 绘制图表
    print("正在生成 F1 分数排名图...")
    print(f"模型数量: {len(models)}")
    print(f"最高 F1 分数: {max(f1_scores):.4f}")

    plot_f1_ranking(
        models=models,
        f1_scores=f1_scores,
        output_dir=args.output,
        filename=args.filename
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
