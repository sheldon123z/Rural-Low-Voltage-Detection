#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bindplotting - 科学绘图模块
===========================

提供模块化的科学绘图功能，支持独立运行或统一调用。

Modules:
    - bindstyle_config: 统一样式配置（Okabe-Ito 配色、字体、尺寸）
    - bindplot_f1_bindranking: F1 分数排名图
    - bindplot_bindmetrics_bindbar: 性能指标分组柱状图
    - bindplot_bindradar: 雷达图
    - bindplot_bindheatmap: 性能热力图
    - bindplot_bindroc: ROC 曲线
    - bindplot_bindpr: PR 曲线
    - bindplot_bindconfusion: 混淆矩阵
    - bindplot_bindtraining: 训练曲线
    - bindgenerate_bindall: 一键生成所有图表

Usage:
    # 作为模块导入
    from bindplotting import bindstyle_config
    from bindplotting.bindplot_f1_bindranking import plot_f1_ranking

    # 命令行运行
    python -m bindplotting.bindgenerate_bindall --output ./figures/
"""

__version__ = '1.0.0'
__author__ = 'Rural Low-Voltage Detection Project'

from .bindstyle_config import (
    apply_bindstyle,
    COLORS,
    COLOR_PALETTE,
    FONT_CONFIG,
    FIGURE_CONFIG,
    SAMPLE_DATA,
    create_bilingual_title,
    save_bindplot,
    get_ranked_colors,
    get_color_gradient,
)

__all__ = [
    'apply_bindstyle',
    'COLORS',
    'COLOR_PALETTE',
    'FONT_CONFIG',
    'FIGURE_CONFIG',
    'SAMPLE_DATA',
    'create_bilingual_title',
    'save_bindplot',
    'get_ranked_colors',
    'get_color_gradient',
]
