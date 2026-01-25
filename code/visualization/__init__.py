"""
Visualization Module for Voltage Anomaly Detection

This module provides thesis-compliant plotting utilities for
rural power grid voltage anomaly detection research.

Components:
    - ThesisPlotter: Main plotting class with thesis formatting
    - PlotFactory: Unified interface for all plot types
    - InteractivePlotter: Plotly-based interactive visualizations
    - THESIS_CONFIG: Configuration for thesis-compliant plots

Usage:
    from visualization import ThesisPlotter, PlotFactory

    # 使用 ThesisPlotter（传统方式）
    plotter = ThesisPlotter(output_dir='thesis/figures/chap3', chapter=3)
    plotter.plot_voltage_timeseries(data, labels)

    # 使用 PlotFactory（推荐方式）
    factory = PlotFactory(output_dir='./figures', chapter=4)
    factory.training_loss_curve(training_history)
    factory.metrics_bar_chart(metrics)
"""

from .thesis_plots import (
    THESIS_CONFIG,
    ThesisPlotter,
    plot_anomaly_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_detection_results,
    plot_fft_spectrum,
    plot_model_comparison,
    plot_reconstruction_error,
    plot_roc_pr_curves,
    plot_training_curves,
    plot_tsne_pca,
    plot_voltage_radar,
    plot_voltage_timeseries,
    setup_thesis_style,
)

from .plot_factory import PlotFactory, create_factory

# 尝试导入交互式绘图模块
try:
    from .interactive_plots import InteractivePlotter, PLOTLY_AVAILABLE
except ImportError:
    PLOTLY_AVAILABLE = False
    InteractivePlotter = None

__all__ = [
    # 原有导出
    "ThesisPlotter",
    "THESIS_CONFIG",
    "setup_thesis_style",
    "plot_voltage_timeseries",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_roc_pr_curves",
    "plot_reconstruction_error",
    "plot_tsne_pca",
    "plot_training_curves",
    "plot_correlation_heatmap",
    "plot_fft_spectrum",
    "plot_anomaly_distribution",
    "plot_detection_results",
    "plot_voltage_radar",
    # 新增导出
    "PlotFactory",
    "create_factory",
    "InteractivePlotter",
    "PLOTLY_AVAILABLE",
]

__version__ = "2.0.0"
__author__ = "Rural Voltage Detection Project"
