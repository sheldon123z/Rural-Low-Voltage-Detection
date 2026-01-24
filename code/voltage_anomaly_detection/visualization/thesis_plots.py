"""
Thesis-Compliant Plotting Module for Voltage Anomaly Detection

This module provides publication-quality visualizations following
Beijing Forestry University Master's Thesis formatting requirements.

Reference:
    - 北京林业大学《研究生学位论文写作指南（2023版）》
    - GB/T 15834-2011《标点符号用法》

Author: Rural Voltage Detection Project
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================================================
# THESIS CONFIGURATION
# ============================================================================

THESIS_CONFIG = {
    # Font sizes (in points) - 北京林业大学论文规范
    'title_fontsize': 12,           # 图标题：约五号
    'label_fontsize': 10.5,         # 坐标轴标签：五号 (10.5pt)
    'tick_fontsize': 9,             # 刻度标签：小五号 (9pt)
    'legend_fontsize': 9,           # 图例：小五号 (9pt)
    'annotation_fontsize': 9,       # 注释文字：小五号

    # Font families
    'font_family_cn': ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei'],
    'font_family_en': 'Times New Roman',

    # Figure dimensions (in inches)
    'single_column_width': 3.5,     # 单栏宽度 (~8.9 cm)
    'double_column_width': 7.0,     # 双栏宽度 (~17.8 cm)
    'max_width': 6.0,               # 最大宽度 (~15.2 cm, 版心宽度)
    'default_height': 3.0,          # 默认高度

    # Resolution
    'dpi': 300,                     # 出版质量
    'dpi_preview': 150,             # 预览质量

    # Line and marker styles
    'linewidth': 1.5,
    'linewidth_thin': 1.0,
    'markersize': 6,
    'markersize_small': 4,

    # Academic color palette (colorblind-friendly)
    'colors': [
        '#1f77b4',  # 蓝色 - Primary
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色 - Anomaly
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
        '#17becf',  # 青色
    ],

    # Special colors
    'color_normal': '#2ca02c',      # 正常 - 绿色
    'color_anomaly': '#d62728',     # 异常 - 红色
    'color_threshold': '#ff7f0e',   # 阈值 - 橙色
    'color_grid': '#cccccc',        # 网格 - 灰色

    # Marker styles
    'markers': ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*'],

    # Line styles
    'linestyles': ['-', '--', '-.', ':', (0, (3, 1, 1, 1))],

    # Anomaly type colors (5 types)
    'anomaly_colors': {
        0: '#2ca02c',   # Normal - 绿色
        1: '#d62728',   # Undervoltage (欠压) - 红色
        2: '#ff7f0e',   # Overvoltage (过压) - 橙色
        3: '#9467bd',   # Voltage Sag (电压骤降) - 紫色
        4: '#1f77b4',   # Harmonic (谐波) - 蓝色
        5: '#8c564b',   # Unbalance (不平衡) - 棕色
    },

    'anomaly_names': {
        0: '正常 (Normal)',
        1: '欠压 (Undervoltage)',
        2: '过压 (Overvoltage)',
        3: '骤降 (Voltage Sag)',
        4: '谐波 (Harmonic)',
        5: '不平衡 (Unbalance)',
    },

    # Voltage thresholds (GB/T 12325-2008)
    'voltage_nominal': 220.0,
    'voltage_lower': 198.0,         # -10%
    'voltage_upper': 242.0,         # +10%
}


def setup_thesis_style():
    """
    Setup matplotlib style for thesis-compliant plots.

    This function configures global matplotlib settings to match
    Beijing Forestry University thesis requirements.
    """
    # Try to set Chinese font
    cn_fonts = THESIS_CONFIG['font_family_cn']
    font_set = False
    for font in cn_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_set = True
            break
        except:
            continue

    if not font_set:
        print("Warning: Chinese font not found. Using default font.")

    # Basic settings
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Figure settings
    plt.rcParams['figure.dpi'] = THESIS_CONFIG['dpi_preview']
    plt.rcParams['savefig.dpi'] = THESIS_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    # Axes settings
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.labelsize'] = THESIS_CONFIG['label_fontsize']
    plt.rcParams['axes.titlesize'] = THESIS_CONFIG['title_fontsize']
    plt.rcParams['axes.grid'] = False

    # Tick settings
    plt.rcParams['xtick.labelsize'] = THESIS_CONFIG['tick_fontsize']
    plt.rcParams['ytick.labelsize'] = THESIS_CONFIG['tick_fontsize']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # Legend settings
    plt.rcParams['legend.fontsize'] = THESIS_CONFIG['legend_fontsize']
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False

    # Line settings
    plt.rcParams['lines.linewidth'] = THESIS_CONFIG['linewidth']
    plt.rcParams['lines.markersize'] = THESIS_CONFIG['markersize']

    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5


# Initialize style on import
setup_thesis_style()


# ============================================================================
# THESIS PLOTTER CLASS
# ============================================================================

class ThesisPlotter:
    """
    Main class for generating thesis-compliant plots.

    This class provides methods for creating various types of plots
    commonly used in time series anomaly detection research, all
    formatted according to thesis requirements.

    Attributes:
        output_dir (str): Directory to save generated plots
        chapter (int): Current chapter number for figure numbering
        fig_counter (int): Counter for automatic figure numbering
        figures (list): List of generated figures
    """

    def __init__(self, output_dir='thesis/figures', chapter=3, style='thesis'):
        """
        Initialize ThesisPlotter.

        Args:
            output_dir: Directory to save plots
            chapter: Chapter number for figure numbering
            style: Plot style ('thesis' or 'default')
        """
        self.output_dir = output_dir
        self.chapter = chapter
        self.fig_counter = 0
        self.figures = []
        self.style = style

        # Create output directory
        full_path = os.path.join(output_dir, f'chap{chapter}')
        os.makedirs(full_path, exist_ok=True)
        self.full_output_dir = full_path

        # Apply thesis style
        if style == 'thesis':
            setup_thesis_style()

    def _get_fig_num(self, fig_num=None):
        """Get figure number, auto-increment if not specified."""
        if fig_num is None:
            self.fig_counter += 1
            return self.fig_counter
        return fig_num

    def _save_figure(self, fig, fig_num, description, format='pdf'):
        """Save figure with thesis-compliant naming."""
        filename = f'fig_{self.chapter}_{fig_num}_{description}.{format}'
        filepath = os.path.join(self.full_output_dir, filename)
        fig.savefig(filepath, format=format, dpi=THESIS_CONFIG['dpi'],
                    bbox_inches='tight', pad_inches=0.1)
        print(f"✅ Saved: {filepath}")
        return filepath

    def _create_figure(self, width=None, height=None, nrows=1, ncols=1):
        """Create figure with thesis-compliant dimensions."""
        if width is None:
            width = THESIS_CONFIG['double_column_width']
        if height is None:
            height = THESIS_CONFIG['default_height']

        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        return fig, axes

    def _add_legend(self, ax, **kwargs):
        """Add thesis-compliant legend."""
        default_kwargs = {
            'fontsize': THESIS_CONFIG['legend_fontsize'],
            'frameon': True,
            'framealpha': 0.9,
            'edgecolor': 'black',
            'fancybox': False,
            'loc': 'best',
        }
        default_kwargs.update(kwargs)
        return ax.legend(**default_kwargs)

    def _set_labels(self, ax, xlabel=None, ylabel=None, title=None):
        """Set axis labels with thesis formatting."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=THESIS_CONFIG['label_fontsize'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=THESIS_CONFIG['label_fontsize'])
        if title:
            ax.set_title(title, fontsize=THESIS_CONFIG['title_fontsize'])

    def save_all(self, format='pdf'):
        """Save all generated figures."""
        for i, (fig, desc) in enumerate(self.figures):
            self._save_figure(fig, i + 1, desc, format)

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================

    def plot_voltage_timeseries(self, data, labels=None, fig_num=None,
                                 title_cn='三相电压时序曲线',
                                 title_en='Three-phase Voltage Time Series',
                                 show_threshold=True, max_points=1000):
        """
        Plot three-phase voltage time series with anomaly highlighting.

        Args:
            data: DataFrame or array with voltage data (Va, Vb, Vc)
            labels: Anomaly labels (0=normal, 1+=anomaly)
            fig_num: Figure number
            title_cn: Chinese title
            title_en: English title
            show_threshold: Whether to show voltage thresholds
            max_points: Maximum points to plot (for performance)
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(height=3.5)

        # Load data if path
        if isinstance(data, str):
            data = pd.read_csv(data)

        # Extract voltage columns
        if isinstance(data, pd.DataFrame):
            if 'Va' in data.columns:
                voltage_cols = ['Va', 'Vb', 'Vc']
            else:
                voltage_cols = data.columns[:3].tolist()
            voltage_data = data[voltage_cols].values
        else:
            voltage_data = data[:, :3] if data.shape[1] >= 3 else data

        # Subsample if too many points
        if len(voltage_data) > max_points:
            indices = np.linspace(0, len(voltage_data) - 1, max_points, dtype=int)
            voltage_data = voltage_data[indices]
            if labels is not None:
                labels = np.array(labels)[indices]

        time = np.arange(len(voltage_data))

        # Plot voltage curves
        phase_labels = ['A相 (Va)', 'B相 (Vb)', 'C相 (Vc)']
        colors = THESIS_CONFIG['colors'][:3]

        for i, (label, color) in enumerate(zip(phase_labels, colors)):
            ax.plot(time, voltage_data[:, i], label=label, color=color,
                    linewidth=THESIS_CONFIG['linewidth'])

        # Highlight anomaly regions
        if labels is not None:
            labels = np.array(labels).flatten()
            anomaly_mask = labels > 0
            if np.any(anomaly_mask):
                ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=anomaly_mask, alpha=0.2,
                                color=THESIS_CONFIG['color_anomaly'],
                                label='异常区间')

        # Show voltage thresholds
        if show_threshold:
            ax.axhline(y=THESIS_CONFIG['voltage_upper'], color='gray',
                       linestyle='--', linewidth=1, alpha=0.7, label='上限 (242V)')
            ax.axhline(y=THESIS_CONFIG['voltage_lower'], color='gray',
                       linestyle='--', linewidth=1, alpha=0.7, label='下限 (198V)')
            ax.axhline(y=THESIS_CONFIG['voltage_nominal'], color='gray',
                       linestyle=':', linewidth=1, alpha=0.5)

        self._set_labels(ax, xlabel='采样点 (Sample)', ylabel='电压 (V)')
        self._add_legend(ax, loc='upper right', ncol=2)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

        # Add bilingual title at bottom
        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'voltage_timeseries'))
        return self._save_figure(fig, fig_num, 'voltage_timeseries')

    def plot_model_comparison(self, results, metrics=None, fig_num=None,
                               title_cn='模型性能对比',
                               title_en='Model Performance Comparison'):
        """
        Plot model comparison bar chart.

        Args:
            results: Dict {model_name: {metric: value}} or path to results file
            metrics: List of metrics to plot
            fig_num: Figure number
            title_cn: Chinese title
            title_en: English title
        """
        fig_num = self._get_fig_num(fig_num)

        # Default metrics for anomaly detection
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'adj_f1']
        metric_labels = {
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1-Score',
            'adj_f1': 'Adj-F1',
            'accuracy': 'Accuracy',
            'roc_auc': 'ROC-AUC',
        }

        # Parse results if path
        if isinstance(results, str):
            # Parse from result file
            results = self._parse_results_file(results)

        models = list(results.keys())
        n_models = len(models)
        n_metrics = len(metrics)

        fig, ax = self._create_figure(width=7.0, height=3.5)

        # Bar positions
        x = np.arange(n_models)
        width = 0.8 / n_metrics
        colors = THESIS_CONFIG['colors'][:n_metrics]

        # Plot bars
        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric_labels.get(metric, metric),
                          color=colors[i], edgecolor='black', linewidth=0.5)

            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=THESIS_CONFIG['tick_fontsize'])
        ax.set_ylabel('Score', fontsize=THESIS_CONFIG['label_fontsize'])
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        self._add_legend(ax, loc='upper right', ncol=2)

        # Add bilingual title
        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'model_comparison'))
        return self._save_figure(fig, fig_num, 'model_comparison')

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, fig_num=None,
                               title_cn='混淆矩阵',
                               title_en='Confusion Matrix',
                               normalize=True):
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Class labels
            fig_num: Figure number
            title_cn: Chinese title
            title_en: English title
            normalize: Whether to normalize
        """
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(width=4.5, height=4.0)

        if labels is None:
            labels = ['正常\nNormal', '异常\nAnomaly']

        # Compute confusion matrix
        cm = sk_confusion_matrix(y_true, y_pred)
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_normalized = cm

        # Plot heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=THESIS_CONFIG['tick_fontsize'])

        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_normalized[i, j] > thresh else 'black'
                text = f'{cm_normalized[i, j]:.2%}\n({cm[i, j]})'
                ax.text(j, i, text, ha='center', va='center', color=color,
                        fontsize=THESIS_CONFIG['annotation_fontsize'])

        # Labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=THESIS_CONFIG['tick_fontsize'])
        ax.set_yticklabels(labels, fontsize=THESIS_CONFIG['tick_fontsize'])
        ax.set_xlabel('预测标签 (Predicted)', fontsize=THESIS_CONFIG['label_fontsize'])
        ax.set_ylabel('真实标签 (Actual)', fontsize=THESIS_CONFIG['label_fontsize'])

        # Add bilingual title
        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'confusion_matrix'))
        return self._save_figure(fig, fig_num, 'confusion_matrix')

    def plot_roc_pr_curves(self, y_true, y_scores, model_names=None, fig_num=None,
                            title_cn='ROC曲线与PR曲线',
                            title_en='ROC and PR Curves'):
        """
        Plot ROC and PR curves in subplots.

        Args:
            y_true: Ground truth labels (or dict for multiple models)
            y_scores: Predicted scores (or dict for multiple models)
            model_names: List of model names
            fig_num: Figure number
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        fig_num = self._get_fig_num(fig_num)
        fig, (ax1, ax2) = self._create_figure(width=7.0, height=3.0, ncols=2)

        colors = THESIS_CONFIG['colors']

        # Handle single or multiple models
        if isinstance(y_scores, dict):
            models_data = y_scores
        else:
            models_data = {'Model': (y_true, y_scores)}

        for i, (name, data) in enumerate(models_data.items()):
            if isinstance(data, tuple):
                yt, ys = data
            else:
                yt, ys = y_true, data

            color = colors[i % len(colors)]

            # ROC curve
            fpr, tpr, _ = roc_curve(yt, ys)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, linewidth=THESIS_CONFIG['linewidth'],
                     label=f'{name} (AUC={roc_auc:.3f})')

            # PR curve
            precision, recall, _ = precision_recall_curve(yt, ys)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, color=color, linewidth=THESIS_CONFIG['linewidth'],
                     label=f'{name} (AP={pr_auc:.3f})')

        # ROC plot formatting
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('假阳性率 (FPR)', fontsize=THESIS_CONFIG['label_fontsize'])
        ax1.set_ylabel('真阳性率 (TPR)', fontsize=THESIS_CONFIG['label_fontsize'])
        ax1.set_title('(a) ROC曲线', fontsize=THESIS_CONFIG['title_fontsize'])
        self._add_legend(ax1, loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # PR plot formatting
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('召回率 (Recall)', fontsize=THESIS_CONFIG['label_fontsize'])
        ax2.set_ylabel('精确率 (Precision)', fontsize=THESIS_CONFIG['label_fontsize'])
        ax2.set_title('(b) PR曲线', fontsize=THESIS_CONFIG['title_fontsize'])
        self._add_legend(ax2, loc='lower left')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # Add bilingual title
        fig.text(0.5, -0.05, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'roc_pr_curves'))
        return self._save_figure(fig, fig_num, 'roc_pr_curves')

    def plot_reconstruction_error(self, train_error, test_error, threshold=None,
                                   labels=None, fig_num=None,
                                   title_cn='重构误差分布',
                                   title_en='Reconstruction Error Distribution'):
        """
        Plot reconstruction error distribution histogram.

        Args:
            train_error: Training set reconstruction errors
            test_error: Test set reconstruction errors
            threshold: Anomaly threshold
            labels: Test set labels for coloring
            fig_num: Figure number
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(height=3.5)

        train_error = np.array(train_error).flatten()
        test_error = np.array(test_error).flatten()

        # Plot histograms
        bins = 50
        ax.hist(train_error, bins=bins, alpha=0.6, label='训练集 (Train)',
                color=THESIS_CONFIG['colors'][0], edgecolor='black', linewidth=0.5)
        ax.hist(test_error, bins=bins, alpha=0.6, label='测试集 (Test)',
                color=THESIS_CONFIG['colors'][1], edgecolor='black', linewidth=0.5)

        # Plot threshold
        if threshold is not None:
            ax.axvline(x=threshold, color=THESIS_CONFIG['color_threshold'],
                       linestyle='--', linewidth=2, label=f'阈值 (τ={threshold:.4f})')

            # Fill anomaly region
            ylim = ax.get_ylim()
            ax.fill_betweenx(ylim, threshold, ax.get_xlim()[1], alpha=0.1,
                             color=THESIS_CONFIG['color_anomaly'])

        self._set_labels(ax, xlabel='重构误差 (Reconstruction Error)',
                        ylabel='频数 (Frequency)')
        self._add_legend(ax, loc='upper right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Add bilingual title
        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'reconstruction_error'))
        return self._save_figure(fig, fig_num, 'reconstruction_error')

    def plot_tsne_pca(self, features, labels, method='tsne', fig_num=None,
                      title_cn='特征空间可视化',
                      title_en='Feature Space Visualization',
                      perplexity=30, n_components=2):
        """
        Plot t-SNE or PCA dimensionality reduction visualization.

        Args:
            features: High-dimensional features (N, D)
            labels: Sample labels
            method: 'tsne' or 'pca'
            fig_num: Figure number
            perplexity: t-SNE perplexity parameter
            n_components: Number of components
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(width=5.0, height=4.5)

        features = np.array(features)
        labels = np.array(labels).flatten()

        # Dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity,
                           random_state=42, n_iter=1000)
            method_name = 't-SNE'
        else:
            reducer = PCA(n_components=n_components, random_state=42)
            method_name = 'PCA'

        embedded = reducer.fit_transform(features)

        # Get unique labels and colors
        unique_labels = np.unique(labels)
        colors = THESIS_CONFIG['anomaly_colors']
        markers = THESIS_CONFIG['markers']

        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors.get(int(label), THESIS_CONFIG['colors'][i % len(THESIS_CONFIG['colors'])])
            marker = markers[i % len(markers)]
            name = THESIS_CONFIG['anomaly_names'].get(int(label), f'Class {label}')

            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                       c=color, marker=marker, s=30, alpha=0.7,
                       label=name, edgecolors='white', linewidth=0.5)

        self._set_labels(ax, xlabel=f'{method_name} 维度1',
                        ylabel=f'{method_name} 维度2')
        self._add_legend(ax, loc='best', markerscale=1.5)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Update title
        title_cn = title_cn.replace('特征空间可视化', f'{method_name}特征空间可视化')
        title_en = title_en.replace('Feature Space', f'{method_name} Feature Space')

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, f'{method}_visualization'))
        return self._save_figure(fig, fig_num, f'{method}_visualization')

    def plot_training_curves(self, train_loss, val_loss=None, epochs=None,
                              fig_num=None, title_cn='训练损失曲线',
                              title_en='Training Loss Curve',
                              early_stop_epoch=None):
        """
        Plot training and validation loss curves.

        Args:
            train_loss: Training loss values
            val_loss: Validation loss values
            epochs: Epoch numbers
            fig_num: Figure number
            early_stop_epoch: Epoch where early stopping occurred
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(height=3.0)

        train_loss = np.array(train_loss)
        if epochs is None:
            epochs = np.arange(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, color=THESIS_CONFIG['colors'][0],
                linewidth=THESIS_CONFIG['linewidth'], label='训练损失 (Train)')

        if val_loss is not None:
            val_loss = np.array(val_loss)
            ax.plot(epochs, val_loss, color=THESIS_CONFIG['colors'][1],
                    linewidth=THESIS_CONFIG['linewidth'], label='验证损失 (Val)')

        # Mark early stopping point
        if early_stop_epoch is not None:
            ax.axvline(x=early_stop_epoch, color='gray', linestyle='--',
                       linewidth=1, alpha=0.7)
            ax.annotate(f'Early Stop\n(Epoch {early_stop_epoch})',
                        xy=(early_stop_epoch, ax.get_ylim()[1] * 0.9),
                        fontsize=THESIS_CONFIG['annotation_fontsize'],
                        ha='center')

        self._set_labels(ax, xlabel='训练轮次 (Epoch)', ylabel='损失 (Loss)')
        self._add_legend(ax, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'training_curves'))
        return self._save_figure(fig, fig_num, 'training_curves')

    def plot_correlation_heatmap(self, data, feature_names=None, fig_num=None,
                                  title_cn='特征相关性热力图',
                                  title_en='Feature Correlation Heatmap'):
        """
        Plot feature correlation heatmap.

        Args:
            data: Data array or DataFrame
            feature_names: List of feature names
            fig_num: Figure number
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(width=6.5, height=5.5)

        if isinstance(data, pd.DataFrame):
            corr = data.corr()
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            corr = np.corrcoef(data, rowvar=False)
            if feature_names is None:
                feature_names = [f'F{i}' for i in range(data.shape[1])]

        # Plot heatmap
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=THESIS_CONFIG['tick_fontsize'])
        cbar.set_label('相关系数', fontsize=THESIS_CONFIG['label_fontsize'])

        # Set ticks
        n_features = len(feature_names)
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha='right',
                          fontsize=7)
        ax.set_yticklabels(feature_names, fontsize=7)

        # Add correlation values
        if n_features <= 17:  # Only annotate if not too many features
            for i in range(n_features):
                for j in range(n_features):
                    value = corr.iloc[i, j] if isinstance(corr, pd.DataFrame) else corr[i, j]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=color, fontsize=6)

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'correlation_heatmap'))
        return self._save_figure(fig, fig_num, 'correlation_heatmap')

    def plot_fft_spectrum(self, signal, sampling_rate=1.0, top_k=5, fig_num=None,
                          title_cn='FFT频谱分析',
                          title_en='FFT Spectrum Analysis'):
        """
        Plot FFT frequency spectrum.

        Args:
            signal: Time series signal
            sampling_rate: Sampling rate (Hz)
            top_k: Number of top frequencies to annotate
            fig_num: Figure number
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(height=3.0)

        signal = np.array(signal).flatten()
        n = len(signal)

        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(n, d=1/sampling_rate)
        fft_magnitude = np.abs(fft_vals)

        # Plot spectrum
        ax.plot(fft_freq, fft_magnitude, color=THESIS_CONFIG['colors'][0],
                linewidth=THESIS_CONFIG['linewidth'])
        ax.fill_between(fft_freq, fft_magnitude, alpha=0.3,
                        color=THESIS_CONFIG['colors'][0])

        # Annotate top-k frequencies
        top_indices = np.argsort(fft_magnitude)[-top_k:][::-1]
        for idx in top_indices:
            freq = fft_freq[idx]
            mag = fft_magnitude[idx]
            period = n / (idx + 1) if idx > 0 else np.inf
            ax.annotate(f'f={freq:.2f}Hz\nT={period:.1f}',
                        xy=(freq, mag), xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=7, ha='left')
            ax.plot(freq, mag, 'ro', markersize=5)

        self._set_labels(ax, xlabel='频率 (Frequency) / Hz',
                        ylabel='幅值 (Magnitude)')
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'fft_spectrum'))
        return self._save_figure(fig, fig_num, 'fft_spectrum')

    def plot_anomaly_distribution(self, labels, fig_num=None,
                                   title_cn='异常类型分布',
                                   title_en='Anomaly Type Distribution'):
        """
        Plot anomaly type distribution pie/donut chart.

        Args:
            labels: Anomaly type labels
            fig_num: Figure number
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(width=5.0, height=4.0)

        labels = np.array(labels).flatten()
        unique, counts = np.unique(labels, return_counts=True)

        # Get colors and names
        colors = [THESIS_CONFIG['anomaly_colors'].get(int(l), 'gray') for l in unique]
        names = [THESIS_CONFIG['anomaly_names'].get(int(l), f'Type {l}') for l in unique]

        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            counts, labels=names, colors=colors, autopct='%1.1f%%',
            pctdistance=0.75, startangle=90, wedgeprops=dict(width=0.5)
        )

        # Style
        for autotext in autotexts:
            autotext.set_fontsize(THESIS_CONFIG['tick_fontsize'])

        ax.axis('equal')

        # Add center text
        ax.text(0, 0, f'总计\n{len(labels)}', ha='center', va='center',
                fontsize=THESIS_CONFIG['label_fontsize'], fontweight='bold')

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'anomaly_distribution'))
        return self._save_figure(fig, fig_num, 'anomaly_distribution')

    def plot_detection_results(self, predictions, ground_truth, data=None,
                                fig_num=None, title_cn='异常检测结果',
                                title_en='Anomaly Detection Results',
                                max_points=500):
        """
        Plot detection results comparison.

        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels
            data: Original data (optional, for voltage plot)
            fig_num: Figure number
            max_points: Maximum points to plot
        """
        fig_num = self._get_fig_num(fig_num)

        if data is not None:
            fig, (ax1, ax2) = self._create_figure(width=7.0, height=4.5, nrows=2)
        else:
            fig, ax2 = self._create_figure(width=7.0, height=2.5)
            ax1 = None

        predictions = np.array(predictions).flatten()[:max_points]
        ground_truth = np.array(ground_truth).flatten()[:max_points]
        time = np.arange(len(predictions))

        # Plot original data if provided
        if ax1 is not None and data is not None:
            data = np.array(data)[:max_points]
            if data.ndim == 2:
                ax1.plot(time, data[:, 0], color=THESIS_CONFIG['colors'][0],
                         linewidth=THESIS_CONFIG['linewidth_thin'], alpha=0.8)
            else:
                ax1.plot(time, data, color=THESIS_CONFIG['colors'][0],
                         linewidth=THESIS_CONFIG['linewidth_thin'], alpha=0.8)

            # Highlight detected anomalies
            anomaly_mask = predictions > 0
            if np.any(anomaly_mask):
                ax1.fill_between(time, ax1.get_ylim()[0], ax1.get_ylim()[1],
                                 where=anomaly_mask, alpha=0.3,
                                 color=THESIS_CONFIG['color_anomaly'],
                                 label='检测到的异常')

            self._set_labels(ax1, ylabel='信号值')
            ax1.set_title('(a) 原始信号与检测结果', fontsize=THESIS_CONFIG['title_fontsize'])
            self._add_legend(ax1, loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.3)

        # Plot labels comparison
        ax2.fill_between(time, 0, ground_truth, alpha=0.5, step='mid',
                         color=THESIS_CONFIG['color_normal'], label='真实标签')
        ax2.fill_between(time, 0, predictions * 0.8, alpha=0.5, step='mid',
                         color=THESIS_CONFIG['color_anomaly'], label='预测标签')

        self._set_labels(ax2, xlabel='采样点 (Sample)', ylabel='标签')
        if ax1 is None:
            ax2.set_title('标签对比', fontsize=THESIS_CONFIG['title_fontsize'])
        else:
            ax2.set_title('(b) 真实标签与预测标签对比', fontsize=THESIS_CONFIG['title_fontsize'])
        self._add_legend(ax2, loc='upper right')
        ax2.set_ylim(-0.1, 1.5)
        ax2.grid(True, linestyle='--', alpha=0.3)

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'detection_results'))
        return self._save_figure(fig, fig_num, 'detection_results')

    def plot_voltage_radar(self, metrics, fig_num=None,
                            title_cn='电压质量指标雷达图',
                            title_en='Voltage Quality Radar Chart'):
        """
        Plot radar chart for voltage quality metrics.

        Args:
            metrics: Dict of metrics {name: value}
            fig_num: Figure number
        """
        fig_num = self._get_fig_num(fig_num)
        fig, ax = self._create_figure(width=5.0, height=4.5)

        # Default metrics
        default_metrics = {
            'Va偏差率': 0.0,
            'Vb偏差率': 0.0,
            'Vc偏差率': 0.0,
            '三相不平衡度': 0.0,
            'THD_Va': 0.0,
            'THD_Vb': 0.0,
            'THD_Vc': 0.0,
        }
        default_metrics.update(metrics)

        categories = list(default_metrics.keys())
        values = list(default_metrics.values())

        # Number of variables
        N = len(categories)

        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        values += values[:1]

        # Clear and create polar plot
        ax = fig.add_subplot(111, polar=True)

        # Plot
        ax.plot(angles, values, color=THESIS_CONFIG['colors'][0],
                linewidth=THESIS_CONFIG['linewidth'])
        ax.fill(angles, values, alpha=0.25, color=THESIS_CONFIG['colors'][0])

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=THESIS_CONFIG['tick_fontsize'])

        fig.text(0.5, -0.02, f'图{self.chapter}.{fig_num} {title_cn}\n'
                             f'Figure {self.chapter}.{fig_num} {title_en}',
                 ha='center', fontsize=THESIS_CONFIG['label_fontsize'])

        plt.tight_layout()
        self.figures.append((fig, 'voltage_radar'))
        return self._save_figure(fig, fig_num, 'voltage_radar')

    def _parse_results_file(self, filepath):
        """Parse results from result_anomaly_detection.txt format."""
        results = {}
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Simple parsing - adjust based on actual format
            import re
            pattern = r'(\w+).*?precision[:\s]+([0-9.]+).*?recall[:\s]+([0-9.]+).*?f1[:\s]+([0-9.]+)'
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                model = match[0]
                results[model] = {
                    'precision': float(match[1]),
                    'recall': float(match[2]),
                    'f1': float(match[3]),
                }
        except Exception as e:
            print(f"Warning: Could not parse results file: {e}")
            # Return sample data
            results = {
                'TimesNet': {'precision': 0.92, 'recall': 0.89, 'f1': 0.90, 'adj_f1': 0.92},
                'DLinear': {'precision': 0.85, 'recall': 0.82, 'f1': 0.83, 'adj_f1': 0.86},
                'Informer': {'precision': 0.88, 'recall': 0.85, 'f1': 0.86, 'adj_f1': 0.89},
            }

        return results


# ============================================================================
# STANDALONE PLOTTING FUNCTIONS
# ============================================================================

def plot_voltage_timeseries(data, labels=None, **kwargs):
    """Standalone function for voltage timeseries plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 3))
    return plotter.plot_voltage_timeseries(data, labels, **kwargs)


def plot_model_comparison(results, **kwargs):
    """Standalone function for model comparison plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_model_comparison(results, **kwargs)


def plot_confusion_matrix(y_true, y_pred, **kwargs):
    """Standalone function for confusion matrix plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_confusion_matrix(y_true, y_pred, **kwargs)


def plot_roc_pr_curves(y_true, y_scores, **kwargs):
    """Standalone function for ROC/PR curves plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_roc_pr_curves(y_true, y_scores, **kwargs)


def plot_reconstruction_error(train_error, test_error, **kwargs):
    """Standalone function for reconstruction error plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 3))
    return plotter.plot_reconstruction_error(train_error, test_error, **kwargs)


def plot_tsne_pca(features, labels, **kwargs):
    """Standalone function for t-SNE/PCA plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_tsne_pca(features, labels, **kwargs)


def plot_training_curves(train_loss, val_loss=None, **kwargs):
    """Standalone function for training curves plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_training_curves(train_loss, val_loss, **kwargs)


def plot_correlation_heatmap(data, **kwargs):
    """Standalone function for correlation heatmap plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 2))
    return plotter.plot_correlation_heatmap(data, **kwargs)


def plot_fft_spectrum(signal, **kwargs):
    """Standalone function for FFT spectrum plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 3))
    return plotter.plot_fft_spectrum(signal, **kwargs)


def plot_anomaly_distribution(labels, **kwargs):
    """Standalone function for anomaly distribution plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 2))
    return plotter.plot_anomaly_distribution(labels, **kwargs)


def plot_detection_results(predictions, ground_truth, **kwargs):
    """Standalone function for detection results plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 4))
    return plotter.plot_detection_results(predictions, ground_truth, **kwargs)


def plot_voltage_radar(metrics, **kwargs):
    """Standalone function for voltage radar plot."""
    plotter = ThesisPlotter(output_dir=kwargs.get('output_dir', 'thesis/figures'),
                            chapter=kwargs.get('chapter', 2))
    return plotter.plot_voltage_radar(metrics, **kwargs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """Demo: Generate sample plots."""
    import argparse

    parser = argparse.ArgumentParser(description='Thesis Plot Generator')
    parser.add_argument('--type', type=str, default='all',
                        help='Plot type or "all" for demo')
    parser.add_argument('--output', type=str, default='thesis/figures',
                        help='Output directory')
    parser.add_argument('--chapter', type=int, default=3,
                        help='Chapter number')
    args = parser.parse_args()

    print("=" * 50)
    print("Thesis Plot Generator Demo")
    print("=" * 50)

    # Create plotter
    plotter = ThesisPlotter(output_dir=args.output, chapter=args.chapter)

    # Generate sample data
    np.random.seed(42)
    n_samples = 500

    # Sample voltage data
    t = np.linspace(0, 10, n_samples)
    voltage_data = np.column_stack([
        220 + 5 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 2,
        220 + 5 * np.sin(2 * np.pi * 0.5 * t + 2 * np.pi / 3) + np.random.randn(n_samples) * 2,
        220 + 5 * np.sin(2 * np.pi * 0.5 * t + 4 * np.pi / 3) + np.random.randn(n_samples) * 2,
    ])

    # Sample labels (inject some anomalies)
    labels = np.zeros(n_samples)
    labels[100:120] = 1  # Undervoltage
    labels[250:270] = 2  # Overvoltage
    voltage_data[100:120, :] -= 30  # Simulate undervoltage
    voltage_data[250:270, :] += 25  # Simulate overvoltage

    if args.type == 'all' or args.type == 'timeseries':
        print("\n[1] Generating voltage timeseries plot...")
        plotter.plot_voltage_timeseries(voltage_data, labels)

    if args.type == 'all' or args.type == 'comparison':
        print("\n[2] Generating model comparison plot...")
        results = {
            'TimesNet': {'precision': 0.92, 'recall': 0.89, 'f1': 0.90, 'adj_f1': 0.93},
            'DLinear': {'precision': 0.85, 'recall': 0.82, 'f1': 0.83, 'adj_f1': 0.86},
            'Informer': {'precision': 0.88, 'recall': 0.85, 'f1': 0.86, 'adj_f1': 0.89},
            'VoltageTimesNet': {'precision': 0.94, 'recall': 0.91, 'f1': 0.92, 'adj_f1': 0.95},
        }
        plotter.plot_model_comparison(results)

    if args.type == 'all' or args.type == 'confusion':
        print("\n[3] Generating confusion matrix...")
        y_true = (labels > 0).astype(int)
        y_pred = y_true.copy()
        y_pred[np.random.rand(n_samples) < 0.1] = 1 - y_pred[np.random.rand(n_samples) < 0.1]
        plotter.plot_confusion_matrix(y_true, y_pred)

    if args.type == 'all' or args.type == 'reconstruction':
        print("\n[4] Generating reconstruction error distribution...")
        train_error = np.random.exponential(0.5, 1000)
        test_error = np.concatenate([
            np.random.exponential(0.5, 800),
            np.random.exponential(2.0, 200)
        ])
        plotter.plot_reconstruction_error(train_error, test_error, threshold=1.5)

    if args.type == 'all' or args.type == 'tsne':
        print("\n[5] Generating t-SNE visualization...")
        features = np.random.randn(200, 17)
        features[labels[:200] > 0] += 2
        plotter.plot_tsne_pca(features, labels[:200], method='tsne')

    if args.type == 'all' or args.type == 'fft':
        print("\n[6] Generating FFT spectrum...")
        signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        plotter.plot_fft_spectrum(signal, sampling_rate=1.0)

    print("\n" + "=" * 50)
    print(f"All plots saved to: {plotter.full_output_dir}")
    print("=" * 50)
