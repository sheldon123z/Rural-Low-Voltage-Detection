"""
图表工厂 - 统一接口

本模块提供统一的图表生成接口，支持 matplotlib 静态图表和 Plotly 交互式图表。

支持的图表类型（18种）：
A. 训练过程可视化：训练损失曲线、学习率变化曲线
B. 模型性能对比：性能指标柱状图、雷达图、热力图、F1排名图
C. 分类评估：ROC曲线、PR曲线、混淆矩阵
D. 异常检测专用：重构误差分布、检测结果时序图、阈值敏感性分析
E. 特征分析：t-SNE可视化、特征相关性热力图
F. 频谱分析：FFT频谱图、周期性分析图
G. 统计分析：箱线图、置信区间图

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 导入字体配置
from utils.font_config import (
    setup_matplotlib_thesis_style,
    ACADEMIC_COLORS, MARKERS, LINE_STYLES,
    ANOMALY_COLORS, ANOMALY_NAMES,
    get_color, get_marker, get_linestyle,
)


class PlotFactory:
    """
    图表生成工厂

    统一接口生成各类科研图表，支持论文格式规范。

    Attributes:
        output_dir: 输出目录
        chapter: 章节号
        fig_counter: 图表计数器
    """

    def __init__(
        self,
        output_dir: str = "./figures",
        chapter: int = 4,
        style: str = "thesis",
        dpi: int = 300
    ):
        """
        初始化图表工厂

        Args:
            output_dir: 输出目录
            chapter: 章节号，用于图表编号
            style: 绘图样式
            dpi: 输出分辨率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chapter = chapter
        self.fig_counter = 0
        self.dpi = dpi

        # 初始化样式
        setup_matplotlib_thesis_style(chapter=chapter, verbose=False)

    def _get_fig_num(self) -> str:
        """获取图表编号"""
        self.fig_counter += 1
        return f"{self.chapter}.{self.fig_counter}"

    def _save_figure(self, fig: plt.Figure, filename: str, tight: bool = True) -> str:
        """保存图表"""
        path = self.output_dir / f"{filename}.png"
        if tight:
            fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight", pad_inches=0.1)
        else:
            fig.savefig(str(path), dpi=self.dpi)
        plt.close(fig)
        return str(path)

    def _add_bilingual_title(self, ax, title_cn: str, title_en: str, fig_num: str = None):
        """添加中英文双语标题"""
        if fig_num:
            full_title = f"图{fig_num} {title_cn}\nFigure {fig_num} {title_en}"
        else:
            full_title = f"{title_cn}\n{title_en}"
        ax.set_title(full_title, fontsize=12, pad=10)

    # ========================================================================
    # A. 训练过程可视化
    # ========================================================================

    def training_loss_curve(
        self,
        training_history: Dict[str, List[Dict]],
        filename: str = "训练曲线对比",
        figsize: Tuple[float, float] = (10, 8)
    ) -> str:
        """
        绘制训练损失曲线

        Args:
            training_history: {模型名: [{'epoch', 'train_loss', 'vali_loss', 'test_loss'}, ...]}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        loss_types = [
            ("train_loss", "训练损失", "Training Loss"),
            ("vali_loss", "验证损失", "Validation Loss"),
            ("test_loss", "测试损失", "Test Loss"),
        ]

        for idx, (loss_key, title_cn, title_en) in enumerate(loss_types):
            ax = axes[idx // 2, idx % 2]

            for i, (model, history) in enumerate(training_history.items()):
                epochs = [h["epoch"] for h in history]
                loss = [h[loss_key] for h in history]

                ax.plot(epochs, loss, label=model,
                       color=get_color(i), marker=get_marker(i),
                       markevery=2, linewidth=1.5, markersize=5)

            ax.set_xlabel("训练轮次 (Epoch)")
            ax.set_ylabel("损失 (Loss)")
            ax.set_title(f"{title_cn}\n{title_en}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

        # 第四个子图：最终损失对比柱状图
        ax = axes[1, 1]
        models = list(training_history.keys())
        x = np.arange(len(models))
        width = 0.25

        final_train = [training_history[m][-1]["train_loss"] for m in models]
        final_vali = [training_history[m][-1]["vali_loss"] for m in models]
        final_test = [training_history[m][-1]["test_loss"] for m in models]

        ax.bar(x - width, final_train, width, label="训练", color=get_color(0))
        ax.bar(x, final_vali, width, label="验证", color=get_color(1))
        ax.bar(x + width, final_test, width, label="测试", color=get_color(2))

        ax.set_xlabel("模型 (Model)")
        ax.set_ylabel("最终损失 (Final Loss)")
        ax.set_title("最终损失对比\nFinal Loss Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        fig_num = self._get_fig_num()
        fig.suptitle(f"图{fig_num} 模型训练曲线对比\nFigure {fig_num} Model Training Curves Comparison",
                    fontsize=14, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # B. 模型性能对比
    # ========================================================================

    def metrics_bar_chart(
        self,
        metrics: Dict[str, Dict],
        filename: str = "性能指标对比",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制性能指标柱状图

        Args:
            metrics: {模型名: {'accuracy', 'precision', 'recall', 'f1_score'}}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        models = list(metrics.keys())
        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        metric_labels = ["准确率\nAccuracy", "精确率\nPrecision", "召回率\nRecall", "F1分数\nF1-Score"]

        x = np.arange(len(models))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            values = [metrics[m].get(metric, 0) for m in models]
            bars = ax.bar(x + i * width - 1.5 * width, values, width,
                         label=label, color=get_color(i))

            # 在柱顶添加数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xlabel("模型 (Model)")
        ax.set_ylabel("指标值 (Metric Value)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 模型性能指标对比\nFigure {fig_num} Model Performance Comparison",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def radar_chart(
        self,
        metrics: Dict[str, Dict],
        filename: str = "雷达图对比",
        figsize: Tuple[float, float] = (8, 8)
    ) -> str:
        """
        绘制性能雷达图

        Args:
            metrics: 模型指标字典
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

        categories = ["准确率", "精确率", "召回率", "F1分数"]
        num_vars = len(categories)

        # 角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        for i, (model, m) in enumerate(metrics.items()):
            values = [
                m.get("accuracy", 0),
                m.get("precision", 0),
                m.get("recall", 0),
                m.get("f1_score", 0),
            ]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=model,
                   color=get_color(i), markersize=6)
            ax.fill(angles, values, alpha=0.15, color=get_color(i))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 模型性能雷达图\nFigure {fig_num} Model Performance Radar Chart",
                    fontsize=12, y=1.08)

        return self._save_figure(fig, filename)

    def performance_heatmap(
        self,
        metrics: Dict[str, Dict],
        filename: str = "性能热力图",
        figsize: Tuple[float, float] = (10, 8)
    ) -> str:
        """
        绘制性能热力图

        Args:
            metrics: 模型指标字典
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        models = list(metrics.keys())
        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        metric_labels = ["准确率", "精确率", "召回率", "F1分数"]

        # 构建数据矩阵
        data = np.array([
            [metrics[m].get(metric, 0) for metric in metric_names]
            for m in models
        ])

        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.7, vmax=1.0)

        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("指标值", fontsize=10)

        # 设置刻度
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_yticklabels(models, fontsize=10)

        # 添加数值标注
        for i in range(len(models)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f"{data[i, j]:.4f}",
                              ha="center", va="center", fontsize=9,
                              color="white" if data[i, j] < 0.85 else "black")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 模型性能热力图\nFigure {fig_num} Model Performance Heatmap",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def f1_ranking_chart(
        self,
        metrics: Dict[str, Dict],
        filename: str = "F1分数对比",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制 F1 分数排名图

        Args:
            metrics: 模型指标字典
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 按 F1 分数排序
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1].get("f1_score", 0), reverse=True)
        models = [m for m, _ in sorted_metrics]
        f1_scores = [m.get("f1_score", 0) for _, m in sorted_metrics]

        # 颜色渐变
        colors = [plt.cm.RdYlGn(0.3 + 0.6 * (i / len(models))) for i in range(len(models))]
        colors = colors[::-1]

        y = np.arange(len(models))
        bars = ax.barh(y, f1_scores, color=colors, edgecolor="black", linewidth=0.5)

        # 添加数值标注
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                   f"{score:.4f}", va="center", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("F1分数 (F1-Score)")
        ax.set_xlim(0, max(f1_scores) * 1.1)
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 模型F1分数排名\nFigure {fig_num} Model F1-Score Ranking",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # C. 分类评估
    # ========================================================================

    def roc_curve(
        self,
        predictions: Dict[str, Dict],
        filename: str = "ROC曲线",
        figsize: Tuple[float, float] = (8, 8)
    ) -> str:
        """
        绘制 ROC 曲线

        Args:
            predictions: {模型名: {'gt': array, 'test_energy': array, 'threshold': float}}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=figsize)

        for i, (model, data) in enumerate(predictions.items()):
            if "gt" not in data or "test_energy" not in data:
                continue

            gt = data["gt"]
            scores = data["test_energy"]

            # 确保长度一致
            min_len = min(len(gt), len(scores))
            gt = gt[:min_len]
            scores = scores[:min_len]

            fpr, tpr, _ = roc_curve(gt, scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{model} (AUC={roc_auc:.4f})",
                   color=get_color(i), linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机分类器")
        ax.set_xlabel("假阳性率 (False Positive Rate)")
        ax.set_ylabel("真阳性率 (True Positive Rate)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} ROC曲线对比\nFigure {fig_num} ROC Curve Comparison",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def pr_curve(
        self,
        predictions: Dict[str, Dict],
        filename: str = "PR曲线",
        figsize: Tuple[float, float] = (8, 8)
    ) -> str:
        """
        绘制 PR 曲线

        Args:
            predictions: {模型名: {'gt': array, 'test_energy': array}}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        fig, ax = plt.subplots(figsize=figsize)

        for i, (model, data) in enumerate(predictions.items()):
            if "gt" not in data or "test_energy" not in data:
                continue

            gt = data["gt"]
            scores = data["test_energy"]

            min_len = min(len(gt), len(scores))
            gt = gt[:min_len]
            scores = scores[:min_len]

            precision, recall, _ = precision_recall_curve(gt, scores)
            ap = average_precision_score(gt, scores)

            ax.plot(recall, precision, label=f"{model} (AP={ap:.4f})",
                   color=get_color(i), linewidth=2)

        ax.set_xlabel("召回率 (Recall)")
        ax.set_ylabel("精确率 (Precision)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} PR曲线对比\nFigure {fig_num} Precision-Recall Curve Comparison",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        filename: str = "混淆矩阵",
        figsize: Tuple[float, float] = (8, 6)
    ) -> str:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        from sklearn.metrics import confusion_matrix as cm

        fig, ax = plt.subplots(figsize=figsize)

        # 计算混淆矩阵
        matrix = cm(y_true, y_pred)
        matrix_percent = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis] * 100

        im = ax.imshow(matrix_percent, cmap="Blues")

        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("百分比 (%)", fontsize=10)

        # 设置标签
        labels = ["正常", "异常"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("预测标签 (Predicted)")
        ax.set_ylabel("真实标签 (Actual)")

        # 添加数值标注
        for i in range(2):
            for j in range(2):
                text = f"{matrix_percent[i, j]:.1f}%\n({matrix[i, j]})"
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                       color="white" if matrix_percent[i, j] > 50 else "black")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} {model_name}混淆矩阵\nFigure {fig_num} {model_name} Confusion Matrix",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # D. 异常检测专用
    # ========================================================================

    def reconstruction_error_distribution(
        self,
        train_energy: np.ndarray,
        test_energy: np.ndarray,
        threshold: float,
        test_labels: np.ndarray = None,
        filename: str = "重构误差分布",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制重构误差分布图

        Args:
            train_energy: 训练集误差
            test_energy: 测试集误差
            threshold: 异常阈值
            test_labels: 测试集标签
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制直方图
        ax.hist(train_energy, bins=50, alpha=0.5, label="训练集 (Train)",
               color=get_color(0), density=True)
        ax.hist(test_energy, bins=50, alpha=0.5, label="测试集 (Test)",
               color=get_color(1), density=True)

        # 绘制阈值线
        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2,
                  label=f"阈值 (Threshold): {threshold:.4f}")

        # 填充异常区域
        ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.2, color="red", label="异常区域")

        ax.set_xlabel("重构误差 (Reconstruction Error)")
        ax.set_ylabel("密度 (Density)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 重构误差分布\nFigure {fig_num} Reconstruction Error Distribution",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def threshold_sensitivity(
        self,
        energy: np.ndarray,
        labels: np.ndarray,
        filename: str = "阈值敏感性分析",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制阈值敏感性分析图

        Args:
            energy: 异常分数
            labels: 真实标签
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        fig, ax = plt.subplots(figsize=figsize)

        # 计算不同阈值下的指标
        percentiles = np.arange(90, 100, 0.5)
        precisions, recalls, f1s = [], [], []

        for p in percentiles:
            thresh = np.percentile(energy, p)
            pred = (energy > thresh).astype(int)

            min_len = min(len(labels), len(pred))
            precisions.append(precision_score(labels[:min_len], pred[:min_len], zero_division=0))
            recalls.append(recall_score(labels[:min_len], pred[:min_len], zero_division=0))
            f1s.append(f1_score(labels[:min_len], pred[:min_len], zero_division=0))

        ax.plot(percentiles, precisions, "o-", label="精确率 (Precision)",
               color=get_color(0), linewidth=2, markersize=6)
        ax.plot(percentiles, recalls, "s-", label="召回率 (Recall)",
               color=get_color(1), linewidth=2, markersize=6)
        ax.plot(percentiles, f1s, "^-", label="F1分数 (F1-Score)",
               color=get_color(2), linewidth=2, markersize=6)

        # 标注最优F1点
        best_idx = np.argmax(f1s)
        ax.axvline(x=percentiles[best_idx], color="gray", linestyle="--", alpha=0.7)
        ax.scatter([percentiles[best_idx]], [f1s[best_idx]], s=100, c="red", zorder=5,
                  marker="*", label=f"最优F1: {f1s[best_idx]:.4f} @ {percentiles[best_idx]:.1f}%")

        ax.set_xlabel("阈值百分位 (Threshold Percentile)")
        ax.set_ylabel("指标值 (Metric Value)")
        ax.set_xlim([90, 100])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 阈值敏感性分析\nFigure {fig_num} Threshold Sensitivity Analysis",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # E. 特征分析
    # ========================================================================

    def tsne_visualization(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        filename: str = "tSNE可视化",
        figsize: Tuple[float, float] = (10, 8),
        perplexity: int = 30,
        n_iter: int = 1000
    ) -> str:
        """
        绘制 t-SNE 降维可视化

        Args:
            features: 特征数据 (N, D)
            labels: 标签 (N,)
            filename: 输出文件名
            figsize: 图形大小
            perplexity: t-SNE困惑度
            n_iter: 迭代次数

        Returns:
            str: 保存的文件路径
        """
        from sklearn.manifold import TSNE

        # 采样（如果数据太大）
        max_samples = 5000
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embedded = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=figsize)

        # 分类绘制
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            color = ANOMALY_COLORS.get(int(label), get_color(int(label)))
            name = ANOMALY_NAMES.get(int(label), f"类别 {label}")

            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=color, label=name, alpha=0.6, s=20)

        ax.set_xlabel("t-SNE 维度 1")
        ax.set_ylabel("t-SNE 维度 2")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} t-SNE特征空间可视化\nFigure {fig_num} t-SNE Feature Space Visualization",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def correlation_heatmap(
        self,
        data: np.ndarray,
        feature_names: List[str] = None,
        filename: str = "特征相关性热力图",
        figsize: Tuple[float, float] = (10, 8)
    ) -> str:
        """
        绘制特征相关性热力图

        Args:
            data: 特征数据 (N, D)
            feature_names: 特征名称列表
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 计算相关性矩阵
        corr = np.corrcoef(data.T)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("相关系数", fontsize=10)

        # 设置刻度
        n_features = data.shape[1]
        if feature_names is None:
            feature_names = [f"F{i+1}" for i in range(n_features)]

        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=8)

        # 如果特征数较少，添加数值标注
        if n_features <= 17:
            for i in range(n_features):
                for j in range(n_features):
                    text = ax.text(j, i, f"{corr[i, j]:.2f}",
                                  ha="center", va="center", fontsize=7,
                                  color="white" if abs(corr[i, j]) > 0.5 else "black")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 特征相关性热力图\nFigure {fig_num} Feature Correlation Heatmap",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # F. 频谱分析
    # ========================================================================

    def fft_spectrum(
        self,
        signal: np.ndarray,
        sampling_rate: float = 1.0,
        top_k: int = 5,
        filename: str = "FFT频谱分析",
        figsize: Tuple[float, float] = (12, 5)
    ) -> str:
        """
        绘制 FFT 频谱图

        Args:
            signal: 输入信号 (T,) 或 (T, C)
            sampling_rate: 采样率
            top_k: 标注前k个主要频率
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 如果是多通道，取第一个通道
        if signal.ndim > 1:
            signal = signal[:, 0]

        n = len(signal)

        # FFT
        fft_result = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
        magnitude = np.abs(fft_result) / n

        # 时域图
        ax1 = axes[0]
        t = np.arange(n) / sampling_rate
        ax1.plot(t[:min(1000, n)], signal[:min(1000, n)], color=get_color(0), linewidth=0.5)
        ax1.set_xlabel("时间 (Time)")
        ax1.set_ylabel("幅值 (Amplitude)")
        ax1.set_title("时域信号\nTime Domain Signal")
        ax1.grid(True, alpha=0.3, linestyle="--")

        # 频域图
        ax2 = axes[1]
        ax2.fill_between(freqs[1:], magnitude[1:], alpha=0.5, color=get_color(0))
        ax2.plot(freqs[1:], magnitude[1:], color=get_color(0), linewidth=1)

        # 标注主要频率
        top_indices = np.argsort(magnitude[1:])[-top_k:][::-1] + 1
        for idx in top_indices:
            period = 1 / freqs[idx] if freqs[idx] > 0 else float("inf")
            ax2.annotate(f"f={freqs[idx]:.3f}\nT={period:.1f}",
                        xy=(freqs[idx], magnitude[idx]),
                        xytext=(freqs[idx] + 0.01, magnitude[idx] * 1.1),
                        fontsize=8, arrowprops=dict(arrowstyle="->", color="red"))

        ax2.set_xlabel("频率 (Frequency)")
        ax2.set_ylabel("幅值 (Magnitude)")
        ax2.set_title("频域分析\nFrequency Domain Analysis")
        ax2.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        fig.suptitle(f"图{fig_num} FFT频谱分析\nFigure {fig_num} FFT Spectrum Analysis",
                    fontsize=12, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    # ========================================================================
    # G. 统计分析
    # ========================================================================

    def boxplot(
        self,
        data: Dict[str, List[float]],
        filename: str = "多次实验分布",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制箱线图

        Args:
            data: {模型名: [多次实验值]}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        models = list(data.keys())
        values = [data[m] for m in models]

        bp = ax.boxplot(values, labels=models, patch_artist=True)

        # 设置颜色
        for i, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
            patch.set_facecolor(get_color(i))
            patch.set_alpha(0.6)
            median.set_color("black")
            median.set_linewidth(2)

        # 添加均值点
        means = [np.mean(v) for v in values]
        ax.scatter(range(1, len(models) + 1), means, marker="D", color="red",
                  s=50, zorder=5, label="均值")

        ax.set_xlabel("模型 (Model)")
        ax.set_ylabel("F1分数 (F1-Score)")
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 多次实验结果分布\nFigure {fig_num} Multiple Experiments Distribution",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def confidence_interval(
        self,
        data: Dict[str, Tuple[float, float]],
        filename: str = "置信区间对比",
        figsize: Tuple[float, float] = (10, 6)
    ) -> str:
        """
        绘制置信区间图

        Args:
            data: {模型名: (均值, 标准差)}
            filename: 输出文件名
            figsize: 图形大小

        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        models = list(data.keys())
        means = [data[m][0] for m in models]
        stds = [data[m][1] for m in models]

        # 95% 置信区间 (约 1.96 * std)
        ci = [1.96 * s for s in stds]

        x = np.arange(len(models))
        ax.errorbar(x, means, yerr=ci, fmt="o", capsize=5, capthick=2,
                   markersize=10, color=get_color(0), ecolor="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("模型 (Model)")
        ax.set_ylabel("F1分数 (F1-Score)")
        ax.grid(True, alpha=0.3, linestyle="--")

        fig_num = self._get_fig_num()
        ax.set_title(f"图{fig_num} 模型性能95%置信区间\nFigure {fig_num} Model Performance 95% Confidence Interval",
                    fontsize=12, pad=15)
        plt.tight_layout()

        return self._save_figure(fig, filename)


# ============================================================================
# 便捷函数
# ============================================================================

def create_factory(output_dir: str = "./figures", chapter: int = 4) -> PlotFactory:
    """创建图表工厂实例"""
    return PlotFactory(output_dir=output_dir, chapter=chapter)
