#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PR 曲线图 / Precision-Recall Curve Plot
========================================

生成 PR (Precision-Recall) 曲线图。
特点：
- 多模型对比
- AP (Average Precision) 值标注
- 中英双语标签

Usage:
    python bindplot_bindpr.py --data results.json --output ./figures/

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
    MARKERS,
    create_bilingual_title,
    save_bindplot,
)


def generate_sample_pr_data(models, precision_list, recall_list):
    """
    生成示例 PR 数据
    Generate sample PR data

    Args:
        models: 模型列表
        precision_list: 精确率列表
        recall_list: 召回率列表

    Returns:
        dict: 包含各模型 PR 数据的字典
    """
    pr_data = {}

    for i, model in enumerate(models):
        prec = precision_list[i]
        rec = recall_list[i]

        # 生成 PR 曲线点
        n_points = 100
        recall = np.linspace(0, 1, n_points)

        # 基于模型性能生成曲线
        # 使用指数衰减函数
        ap = (prec + rec) / 2  # 近似 AP
        decay_factor = 1 + (1 - ap) * 3

        precision = prec * np.exp(-decay_factor * (recall - rec) ** 2)

        # 确保曲线单调递减
        precision = np.minimum.accumulate(precision[::-1])[::-1]

        # 添加随机扰动
        np.random.seed(hash(model) % (2**31))
        noise = np.random.normal(0, 0.015, n_points)
        precision = np.clip(precision + noise, 0, 1)

        # 计算 AP (平均精确率)
        ap_value = np.trapz(precision, recall)

        pr_data[model] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'ap': float(ap_value)
        }

    return pr_data


def load_data(data_path=None):
    """加载数据"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        sample_data = SAMPLE_DATA.copy()
        sample_data['pr_data'] = generate_sample_pr_data(
            SAMPLE_DATA['models'],
            SAMPLE_DATA['precision'],
            SAMPLE_DATA['recall']
        )
        return sample_data


def plot_pr(pr_data, output_dir=None, filename='pr_curve', top_n=6):
    """
    绘制 PR 曲线图
    Plot Precision-Recall curve

    Args:
        pr_data: PR 数据字典 {模型名: {precision, recall, ap}}
        output_dir: 输出目录
        filename: 输出文件名
        top_n: 显示前 n 个模型
    """
    # 应用统一样式
    apply_bindstyle()

    # 按 AP 排序
    sorted_models = sorted(
        pr_data.keys(),
        key=lambda x: pr_data[x]['ap'],
        reverse=True
    )[:top_n]

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['square'])

    # 绘制每个模型的 PR 曲线
    for idx, model in enumerate(sorted_models):
        data = pr_data[model]
        precision = np.array(data['precision'])
        recall = np.array(data['recall'])
        ap = data['ap']

        # 第一名使用特殊样式
        if idx == 0:
            color = COLORS['best']
            linewidth = 2.8
            alpha = 1.0
            zorder = 10
        else:
            color = COLOR_PALETTE[idx]
            linewidth = 2.0
            alpha = 0.85
            zorder = 5 - idx

        ax.plot(
            recall, precision,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=f'{model} (AP={ap:.3f})',
            marker=MARKERS[idx],
            markevery=15,
            markersize=6,
            zorder=zorder
        )

    # 设置轴范围
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # 设置轴标签
    ax.set_xlabel('召回率 / Recall', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('精确率 / Precision', fontsize=FONT_CONFIG['label_size'])

    # 设置标题
    title = create_bilingual_title(
        'PR 曲线对比',
        'Precision-Recall Curve Comparison'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.4)

    # 添加图例
    ax.legend(
        loc='lower left',
        fontsize=FONT_CONFIG['legend_size'] - 1,
        framealpha=0.9
    )

    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加 F1 等值线
    f1_values = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_values:
        x = np.linspace(0.01, 1, 100)
        y = f1 * x / (2 * x - f1)
        y = np.where((y > 0) & (y <= 1), y, np.nan)
        ax.plot(x, y, '--', color='gray', alpha=0.3, linewidth=1)
        # 添加 F1 标签
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) > 0:
            label_idx = valid_idx[len(valid_idx) // 2]
            ax.annotate(
                f'F1={f1}',
                (x[label_idx], y[label_idx]),
                fontsize=7,
                color='gray',
                alpha=0.6
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
        description='生成 PR 曲线图 / Generate Precision-Recall Curve Plot'
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
        default='pr_curve',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=6,
        help='显示前 n 个模型 / Show top n models'
    )

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)

    # 获取 PR 数据
    if 'pr_data' in data:
        pr_data = data['pr_data']
    else:
        pr_data = generate_sample_pr_data(
            data.get('models', SAMPLE_DATA['models']),
            data.get('precision', SAMPLE_DATA['precision']),
            data.get('recall', SAMPLE_DATA['recall'])
        )

    # 绘制图表
    print("正在生成 PR 曲线图...")
    print(f"显示前 {args.top_n} 个模型")

    plot_pr(
        pr_data=pr_data,
        output_dir=args.output,
        filename=args.filename,
        top_n=args.top_n
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
