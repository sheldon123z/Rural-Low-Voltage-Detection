#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练曲线图 / Training Curve Plot
================================

生成训练曲线图展示模型训练过程中的损失变化。
特点：
- 多模型对比
- 早停点标注
- 训练/验证损失对比
- 中英双语标签

Usage:
    python bindplot_bindtraining.py --data results.json --output ./figures/

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


def generate_sample_training_data(models, f1_scores, n_epochs=50):
    """
    生成示例训练数据
    Generate sample training data

    Args:
        models: 模型列表
        f1_scores: F1 分数列表（用于生成合理的损失曲线）
        n_epochs: 训练轮数

    Returns:
        dict: 训练数据字典
    """
    training_data = {}

    for model, f1 in zip(models, f1_scores):
        np.random.seed(hash(model) % (2**31))

        # 基于 F1 分数确定收敛速度和最终损失
        final_loss = 0.1 + (1 - f1) * 0.5  # F1 越高，最终损失越低
        convergence_speed = 5 + (f1 - 0.8) * 20  # F1 越高，收敛越快

        # 生成训练损失
        epochs = np.arange(1, n_epochs + 1)
        train_loss = final_loss + (1.0 - final_loss) * np.exp(-epochs / convergence_speed)

        # 添加噪声
        noise = np.random.normal(0, 0.02, n_epochs)
        train_loss = train_loss + noise
        train_loss = np.clip(train_loss, 0.01, 2.0)

        # 生成验证损失（略高于训练损失）
        val_noise = np.random.normal(0, 0.03, n_epochs)
        val_loss = train_loss * 1.05 + val_noise + 0.02
        val_loss = np.clip(val_loss, 0.01, 2.0)

        # 确定早停点
        best_val_idx = np.argmin(val_loss)
        early_stop_epoch = best_val_idx + 1 + np.random.randint(3, 8)
        early_stop_epoch = min(early_stop_epoch, n_epochs)

        training_data[model] = {
            'epochs': epochs.tolist(),
            'train_loss': train_loss.tolist(),
            'val_loss': val_loss.tolist(),
            'early_stop_epoch': int(early_stop_epoch),
            'best_val_loss': float(val_loss[best_val_idx]),
            'best_epoch': int(best_val_idx + 1)
        }

    return training_data


def load_data(data_path=None):
    """加载数据"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print("使用示例数据 / Using sample data")
        sample_data = SAMPLE_DATA.copy()
        sample_data['training_data'] = generate_sample_training_data(
            SAMPLE_DATA['models'],
            SAMPLE_DATA['f1_scores']
        )
        return sample_data


def plot_training_curve(training_data, output_dir=None, filename='training_curve',
                        top_n=5, show_val=True):
    """
    绘制训练曲线图
    Plot training curve

    Args:
        training_data: 训练数据字典 {模型名: {epochs, train_loss, val_loss, ...}}
        output_dir: 输出目录
        filename: 输出文件名
        top_n: 显示前 n 个模型
        show_val: 是否显示验证损失
    """
    # 应用统一样式
    apply_bindstyle()

    # 按最终训练损失排序
    sorted_models = sorted(
        training_data.keys(),
        key=lambda x: training_data[x]['train_loss'][-1]
    )[:top_n]

    # 创建图表
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['wide'])

    # 绘制每个模型的训练曲线
    for idx, model in enumerate(sorted_models):
        data = training_data[model]
        epochs = np.array(data['epochs'])
        train_loss = np.array(data['train_loss'])

        # 第一名使用特殊样式
        if idx == 0:
            color = COLORS['best']
            linewidth = 2.5
            alpha = 1.0
            zorder = 10
        else:
            color = COLOR_PALETTE[idx]
            linewidth = 1.8
            alpha = 0.8
            zorder = 5 - idx

        # 绘制训练损失
        ax.plot(
            epochs, train_loss,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=f'{model}',
            zorder=zorder
        )

        # 绘制验证损失（虚线）
        if show_val and 'val_loss' in data:
            val_loss = np.array(data['val_loss'])
            ax.plot(
                epochs, val_loss,
                color=color,
                linewidth=linewidth * 0.7,
                alpha=alpha * 0.6,
                linestyle='--',
                zorder=zorder - 0.5
            )

        # 标记早停点
        if 'early_stop_epoch' in data and data['early_stop_epoch'] < len(epochs):
            es_epoch = data['early_stop_epoch']
            es_loss = train_loss[es_epoch - 1]
            ax.scatter(
                [es_epoch], [es_loss],
                color=color,
                marker='*',
                s=150,
                zorder=zorder + 1,
                edgecolor='white',
                linewidth=1
            )

        # 标记最佳点
        if 'best_epoch' in data and idx == 0:
            best_epoch = data['best_epoch']
            best_loss = data['best_val_loss']
            ax.annotate(
                f'最佳: {best_loss:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 5, best_loss + 0.1),
                fontsize=FONT_CONFIG['annotation_size'],
                arrowprops=dict(arrowstyle='->', color=COLORS['best'], lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    # 设置轴标签
    ax.set_xlabel('训练轮次 / Epoch', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('损失 / Loss', fontsize=FONT_CONFIG['label_size'])

    # 设置标题
    title = create_bilingual_title(
        '模型训练损失曲线',
        'Model Training Loss Curve'
    )
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.4)

    # 设置 Y 轴为对数刻度（如果损失范围较大）
    all_losses = []
    for model in sorted_models:
        all_losses.extend(training_data[model]['train_loss'])
    if max(all_losses) / min(all_losses) > 10:
        ax.set_yscale('log')

    # 添加图例
    legend = ax.legend(
        loc='upper right',
        fontsize=FONT_CONFIG['legend_size'],
        framealpha=0.9
    )

    # 添加说明
    if show_val:
        ax.text(
            0.02, 0.02,
            '实线: 训练损失 / Solid: Train\n虚线: 验证损失 / Dashed: Val\n★: 早停点 / Early Stop',
            transform=ax.transAxes,
            fontsize=FONT_CONFIG['annotation_size'] - 1,
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
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
        description='生成训练曲线图 / Generate Training Curve Plot'
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
        default='training_curve',
        help='输出文件名（不含扩展名）/ Output filename (without extension)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='显示前 n 个模型 / Show top n models'
    )
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='不显示验证损失 / Do not show validation loss'
    )

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)

    # 获取训练数据
    if 'training_data' in data:
        training_data = data['training_data']
    else:
        training_data = generate_sample_training_data(
            data.get('models', SAMPLE_DATA['models']),
            data.get('f1_scores', SAMPLE_DATA['f1_scores'])
        )

    # 绘制图表
    print("正在生成训练曲线图...")
    print(f"显示前 {args.top_n} 个模型")

    plot_training_curve(
        training_data=training_data,
        output_dir=args.output,
        filename=args.filename,
        top_n=args.top_n,
        show_val=not args.no_val
    )

    print("图表生成完成!")


if __name__ == '__main__':
    main()
