#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一样式配置模块 / Unified Style Configuration Module
=====================================================

提供科学绘图的统一样式配置，包括：
- Okabe-Ito 色盲友好配色方案
- 中文字体配置 (WenQuanYi Micro Hei)
- 图表尺寸和DPI设置
- 标签和标题样式

Usage:
    from bindstyle_config import apply_bindstyle, COLORS, FONT_CONFIG
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# =============================================================================
# Okabe-Ito 色盲友好配色方案
# Color-blind friendly palette by Okabe and Ito (2008)
# =============================================================================
COLORS = {
    # 主要颜色 (Primary colors)
    'orange': '#E69F00',      # 橙色 - 最佳/突出
    'sky_blue': '#56B4E9',    # 天蓝色
    'green': '#009E73',       # 青绿色
    'yellow': '#F0E442',      # 黄色
    'blue': '#0072B2',        # 深蓝色
    'vermillion': '#D55E00',  # 朱红色 - 警告/次佳
    'purple': '#CC79A7',      # 紫红色
    'black': '#000000',       # 黑色
    'gray': '#999999',        # 灰色

    # 语义化颜色 (Semantic colors)
    'best': '#E69F00',        # 最佳模型 - 橙色
    'second': '#56B4E9',      # 次佳模型 - 天蓝色
    'third': '#009E73',       # 第三名 - 青绿色
    'baseline': '#999999',    # 基线模型 - 灰色
    'highlight': '#D55E00',   # 高亮 - 朱红色
    'positive': '#009E73',    # 正面 - 青绿色
    'negative': '#D55E00',    # 负面 - 朱红色
}

# 配色列表 (用于多模型对比)
COLOR_PALETTE = [
    COLORS['orange'],
    COLORS['sky_blue'],
    COLORS['green'],
    COLORS['blue'],
    COLORS['vermillion'],
    COLORS['purple'],
    COLORS['yellow'],
    COLORS['black'],
    COLORS['gray'],
]

# 渐变色配置
GRADIENT_COLORS = {
    'warm': ['#FFF7BC', '#FEC44F', '#D95F0E'],  # 暖色渐变
    'cool': ['#EDF8FB', '#66C2A4', '#006D2C'],  # 冷色渐变
    'blue': ['#EFF3FF', '#6BAED6', '#08519C'],  # 蓝色渐变
    'performance': ['#FEE0D2', '#FC9272', '#DE2D26'],  # 性能热图
}

# =============================================================================
# 字体配置
# Font Configuration
# =============================================================================
FONT_CONFIG = {
    'family': 'WenQuanYi Micro Hei',
    'zh_font': 'WenQuanYi Micro Hei',
    'en_font': 'DejaVu Sans',
    'title_size': 14,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'annotation_size': 9,
}

# =============================================================================
# 图表尺寸配置
# Figure Size Configuration
# =============================================================================
FIGURE_CONFIG = {
    # 单图尺寸 (Single figure)
    'single': (8, 6),
    'wide': (10, 6),
    'square': (7, 7),
    'tall': (6, 8),

    # DPI 设置
    'dpi': 300,
    'save_dpi': 300,

    # 边距
    'tight_layout': True,
    'pad_inches': 0.1,
}

# =============================================================================
# 线条和标记配置
# Line and Marker Configuration
# =============================================================================
LINE_CONFIG = {
    'linewidth': 2.0,
    'marker_size': 8,
    'alpha': 0.8,
    'grid_alpha': 0.3,
}

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# =============================================================================
# 应用样式函数
# Style Application Functions
# =============================================================================
def apply_bindstyle():
    """
    应用统一的绘图样式配置
    Apply unified plotting style configuration
    """
    # 设置中文字体
    plt.rcParams['font.family'] = FONT_CONFIG['family']
    plt.rcParams['font.sans-serif'] = [
        FONT_CONFIG['zh_font'],
        FONT_CONFIG['en_font'],
        'SimHei',
        'Arial'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # 字体大小
    plt.rcParams['font.size'] = FONT_CONFIG['tick_size']
    plt.rcParams['axes.titlesize'] = FONT_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = FONT_CONFIG['label_size']
    plt.rcParams['xtick.labelsize'] = FONT_CONFIG['tick_size']
    plt.rcParams['ytick.labelsize'] = FONT_CONFIG['tick_size']
    plt.rcParams['legend.fontsize'] = FONT_CONFIG['legend_size']

    # 图表样式
    plt.rcParams['figure.dpi'] = FIGURE_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = FIGURE_CONFIG['save_dpi']
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # 网格
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = LINE_CONFIG['grid_alpha']
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = '#CCCCCC'

    # 线条
    plt.rcParams['lines.linewidth'] = LINE_CONFIG['linewidth']
    plt.rcParams['lines.markersize'] = LINE_CONFIG['marker_size']

    # 图例
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'

    # 紧凑布局
    plt.rcParams['figure.autolayout'] = FIGURE_CONFIG['tight_layout']


def get_color_gradient(n, cmap_name='viridis'):
    """
    获取 n 个渐变色
    Get n gradient colors

    Args:
        n: 颜色数量
        cmap_name: matplotlib colormap 名称

    Returns:
        颜色列表
    """
    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]


def get_ranked_colors(n, highlight_top=3):
    """
    获取排名颜色（前几名用醒目颜色）
    Get ranked colors with top performers highlighted

    Args:
        n: 总数量
        highlight_top: 需要高亮的前几名

    Returns:
        颜色列表
    """
    highlight_colors = [COLORS['best'], COLORS['second'], COLORS['third']]
    colors = []

    for i in range(n):
        if i < min(highlight_top, len(highlight_colors)):
            colors.append(highlight_colors[i])
        else:
            colors.append(COLORS['gray'])

    return colors


def create_bilingual_title(zh_title, en_title):
    """
    创建中英双语标题
    Create bilingual title (Chinese and English)

    Args:
        zh_title: 中文标题
        en_title: 英文标题

    Returns:
        格式化的双语标题
    """
    return f"{zh_title}\n{en_title}"


def save_bindplot(fig, filename, output_dir=None, formats=('png',)):
    """
    保存图表到指定目录
    Save figure to specified directory

    Args:
        fig: matplotlib figure 对象
        filename: 文件名（不含扩展名）
        output_dir: 输出目录，默认为当前目录
        formats: 保存格式列表，默认 ('png',)
    """
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            filepath,
            dpi=FIGURE_CONFIG['save_dpi'],
            bbox_inches='tight',
            pad_inches=FIGURE_CONFIG['pad_inches'],
            facecolor='white',
            edgecolor='none'
        )
        print(f"已保存 / Saved: {filepath}")


# =============================================================================
# 示例数据（用于测试）
# Sample Data (for testing)
# =============================================================================
SAMPLE_DATA = {
    'models': [
        'VoltageTimesNet', 'TimesNet', 'Transformer', 'DLinear',
        'PatchTST', 'iTransformer', 'Autoformer', 'Informer'
    ],
    'f1_scores': [0.9234, 0.9012, 0.8845, 0.8721, 0.8654, 0.8543, 0.8321, 0.8102],
    'precision': [0.9156, 0.8934, 0.8812, 0.8698, 0.8623, 0.8512, 0.8287, 0.8056],
    'recall': [0.9314, 0.9092, 0.8879, 0.8745, 0.8686, 0.8575, 0.8356, 0.8149],
    'accuracy': [0.9456, 0.9234, 0.9123, 0.9012, 0.8934, 0.8856, 0.8723, 0.8601],
}


# =============================================================================
# 模块初始化
# Module Initialization
# =============================================================================
if __name__ == '__main__':
    # 测试样式配置
    apply_bindstyle()

    print("=" * 60)
    print("bindstyle_config.py - 统一样式配置模块")
    print("=" * 60)
    print("\n可用配色:")
    for name, color in COLORS.items():
        print(f"  {name}: {color}")

    print("\n字体配置:")
    for key, value in FONT_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n图表尺寸配置:")
    for key, value in FIGURE_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n样式已应用成功!")
