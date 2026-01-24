"""
Visualization Module for Voltage Anomaly Detection

This module provides thesis-compliant plotting utilities for
rural power grid voltage anomaly detection research.

Components:
    - ThesisPlotter: Main plotting class with thesis formatting
    - THESIS_CONFIG: Configuration for thesis-compliant plots
    - plot_*: Individual plotting functions

Usage:
    from visualization import ThesisPlotter

    plotter = ThesisPlotter(output_dir='thesis/figures/chap3', chapter=3)
    plotter.plot_voltage_timeseries(data, labels)
    plotter.save_all()
"""

from .thesis_plots import (
    ThesisPlotter,
    THESIS_CONFIG,
    setup_thesis_style,
    plot_voltage_timeseries,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_reconstruction_error,
    plot_tsne_pca,
    plot_training_curves,
    plot_correlation_heatmap,
    plot_fft_spectrum,
    plot_anomaly_distribution,
    plot_detection_results,
    plot_voltage_radar,
)

__all__ = [
    'ThesisPlotter',
    'THESIS_CONFIG',
    'setup_thesis_style',
    'plot_voltage_timeseries',
    'plot_model_comparison',
    'plot_confusion_matrix',
    'plot_roc_pr_curves',
    'plot_reconstruction_error',
    'plot_tsne_pca',
    'plot_training_curves',
    'plot_correlation_heatmap',
    'plot_fft_spectrum',
    'plot_anomaly_distribution',
    'plot_detection_results',
    'plot_voltage_radar',
]

__version__ = '1.0.0'
__author__ = 'Rural Voltage Detection Project'
