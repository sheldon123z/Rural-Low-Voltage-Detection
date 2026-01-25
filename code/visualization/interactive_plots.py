"""
Plotly 交互式可视化模块

本模块提供基于 Plotly 的交互式图表生成功能。

特点：
1. 支持缩放、平移、悬停提示
2. 支持导出为 HTML（离线查看）
3. 支持导出为静态 PNG
4. 中文字体配置

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    pio = None
    warnings.warn("Plotly 未安装，交互式图表功能不可用")


# ============================================================================
# 配置
# ============================================================================

# 中文字体配置
PLOTLY_FONT = "WenQuanYi Micro Hei, Noto Sans CJK SC, Microsoft YaHei, sans-serif"

# 配色方案
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# 异常类型颜色
ANOMALY_COLORS = {
    0: "#2ca02c",  # 正常 - 绿色
    1: "#d62728",  # 欠压 - 红色
    2: "#ff7f0e",  # 过压 - 橙色
    3: "#9467bd",  # 骤降 - 紫色
    4: "#1f77b4",  # 谐波 - 蓝色
    5: "#8c564b",  # 不平衡 - 棕色
}


class InteractivePlotter:
    """
    交互式图表生成器

    使用 Plotly 生成可交互的 HTML 图表。
    """

    def __init__(
        self,
        output_dir: str = "./interactive",
        theme: str = "plotly_white",
        width: int = 1000,
        height: int = 600
    ):
        """
        初始化交互式绘图器

        Args:
            output_dir: 输出目录
            theme: Plotly 主题
            width: 默认图宽度
            height: 默认图高度
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly 未安装，请运行: pip install plotly kaleido")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme
        self.width = width
        self.height = height

        # 设置默认模板
        pio.templates.default = theme

    def _get_layout(self, title: str = "", title_en: str = "") -> Dict:
        """获取标准布局配置"""
        full_title = f"{title}<br><sup>{title_en}</sup>" if title_en else title

        return {
            "title": {
                "text": full_title,
                "font": {"family": PLOTLY_FONT, "size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            "font": {"family": PLOTLY_FONT, "size": 12},
            "xaxis": {
                "title": {"font": {"size": 12}},
                "tickfont": {"size": 10},
                "gridcolor": "#e0e0e0",
                "linecolor": "black",
                "mirror": True,
            },
            "yaxis": {
                "title": {"font": {"size": 12}},
                "tickfont": {"size": 10},
                "gridcolor": "#e0e0e0",
                "linecolor": "black",
                "mirror": True,
            },
            "legend": {
                "font": {"size": 10},
                "bordercolor": "black",
                "borderwidth": 1,
            },
            "margin": {"l": 60, "r": 30, "t": 80, "b": 60},
            "width": self.width,
            "height": self.height,
        }

    # ========================================================================
    # 训练过程可视化
    # ========================================================================

    def training_dashboard(
        self,
        training_history: Dict[str, List[Dict]],
        filename: str = "训练仪表盘"
    ) -> str:
        """
        生成训练过程仪表盘

        Args:
            training_history: {模型名: [{'epoch', 'train_loss', 'vali_loss', 'test_loss'}, ...]}
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "训练损失 (Training Loss)",
                "验证损失 (Validation Loss)",
                "测试损失 (Test Loss)",
                "损失对比 (Loss Comparison)"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for i, (model, history) in enumerate(training_history.items()):
            color = COLORS[i % len(COLORS)]
            epochs = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            vali_loss = [h["vali_loss"] for h in history]
            test_loss = [h["test_loss"] for h in history]

            # 训练损失
            fig.add_trace(
                go.Scatter(x=epochs, y=train_loss, name=model, mode="lines+markers",
                          line=dict(color=color), marker=dict(size=6),
                          legendgroup=model, showlegend=True),
                row=1, col=1
            )

            # 验证损失
            fig.add_trace(
                go.Scatter(x=epochs, y=vali_loss, name=model, mode="lines+markers",
                          line=dict(color=color), marker=dict(size=6),
                          legendgroup=model, showlegend=False),
                row=1, col=2
            )

            # 测试损失
            fig.add_trace(
                go.Scatter(x=epochs, y=test_loss, name=model, mode="lines+markers",
                          line=dict(color=color), marker=dict(size=6),
                          legendgroup=model, showlegend=False),
                row=2, col=1
            )

            # 最终损失对比
            final_train = train_loss[-1] if train_loss else 0
            final_vali = vali_loss[-1] if vali_loss else 0
            final_test = test_loss[-1] if test_loss else 0

        # 添加最终损失柱状图
        models = list(training_history.keys())
        final_losses = {
            "训练": [training_history[m][-1]["train_loss"] for m in models],
            "验证": [training_history[m][-1]["vali_loss"] for m in models],
            "测试": [training_history[m][-1]["test_loss"] for m in models],
        }

        for j, (loss_type, values) in enumerate(final_losses.items()):
            fig.add_trace(
                go.Bar(x=models, y=values, name=loss_type,
                      marker_color=COLORS[j]),
                row=2, col=2
            )

        fig.update_layout(
            title={
                "text": "模型训练过程仪表盘<br><sup>Training Dashboard</sup>",
                "font": {"family": PLOTLY_FONT, "size": 18},
                "x": 0.5,
            },
            font={"family": PLOTLY_FONT},
            height=800,
            width=1200,
            barmode="group",
        )

        fig.update_xaxes(title_text="训练轮次 (Epoch)", row=1, col=1)
        fig.update_xaxes(title_text="训练轮次 (Epoch)", row=1, col=2)
        fig.update_xaxes(title_text="训练轮次 (Epoch)", row=2, col=1)
        fig.update_xaxes(title_text="模型 (Model)", row=2, col=2)

        fig.update_yaxes(title_text="损失 (Loss)", row=1, col=1)
        fig.update_yaxes(title_text="损失 (Loss)", row=1, col=2)
        fig.update_yaxes(title_text="损失 (Loss)", row=2, col=1)
        fig.update_yaxes(title_text="最终损失 (Final Loss)", row=2, col=2)

        return self._save_figure(fig, filename)

    def loss_comparison(
        self,
        training_history: Dict[str, List[Dict]],
        loss_type: str = "vali_loss",
        filename: str = "损失曲线对比"
    ) -> str:
        """
        生成损失曲线对比图

        Args:
            training_history: 训练历史数据
            loss_type: 损失类型 ("train_loss", "vali_loss", "test_loss")
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        loss_names = {
            "train_loss": "训练损失 (Training Loss)",
            "vali_loss": "验证损失 (Validation Loss)",
            "test_loss": "测试损失 (Test Loss)",
        }

        fig = go.Figure()

        for i, (model, history) in enumerate(training_history.items()):
            color = COLORS[i % len(COLORS)]
            epochs = [h["epoch"] for h in history]
            loss = [h[loss_type] for h in history]

            fig.add_trace(go.Scatter(
                x=epochs, y=loss, name=model,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        layout = self._get_layout(
            title=f"模型{loss_names.get(loss_type, loss_type)}对比",
            title_en=f"Model {loss_type.replace('_', ' ').title()} Comparison"
        )
        layout["xaxis"]["title"] = "训练轮次 (Epoch)"
        layout["yaxis"]["title"] = "损失 (Loss)"

        fig.update_layout(**layout)

        return self._save_figure(fig, filename)

    # ========================================================================
    # 性能对比可视化
    # ========================================================================

    def metrics_radar(
        self,
        metrics: Dict[str, Dict],
        filename: str = "性能雷达图"
    ) -> str:
        """
        生成性能雷达图

        Args:
            metrics: {模型名: {'accuracy', 'precision', 'recall', 'f1_score'}}
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        categories = ["准确率", "精确率", "召回率", "F1分数"]
        categories_en = ["Accuracy", "Precision", "Recall", "F1-Score"]

        fig = go.Figure()

        for i, (model, m) in enumerate(metrics.items()):
            values = [
                m.get("accuracy", 0),
                m.get("precision", 0),
                m.get("recall", 0),
                m.get("f1_score", 0),
            ]
            values.append(values[0])  # 闭合多边形

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=COLORS[i % len(COLORS)],
                opacity=0.3,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                name=model,
            ))

        layout = self._get_layout(
            title="模型性能雷达图对比",
            title_en="Model Performance Radar Chart"
        )
        layout["polar"] = {
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "tickvals": [0.2, 0.4, 0.6, 0.8, 1.0],
            }
        }

        fig.update_layout(**layout)

        return self._save_figure(fig, filename)

    def metrics_parallel_coordinates(
        self,
        metrics: Dict[str, Dict],
        filename: str = "平行坐标图"
    ) -> str:
        """
        生成平行坐标图

        Args:
            metrics: 模型指标字典
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        import pandas as pd

        # 准备数据
        data = []
        for model, m in metrics.items():
            data.append({
                "模型": model,
                "准确率": m.get("accuracy", 0),
                "精确率": m.get("precision", 0),
                "召回率": m.get("recall", 0),
                "F1分数": m.get("f1_score", 0),
            })

        df = pd.DataFrame(data)

        # 添加模型编号用于着色
        df["模型编号"] = range(len(df))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df["模型编号"],
                colorscale="Viridis",
                showscale=True,
            ),
            dimensions=[
                dict(label="准确率", values=df["准确率"], range=[0.8, 1.0]),
                dict(label="精确率", values=df["精确率"], range=[0.8, 1.0]),
                dict(label="召回率", values=df["召回率"], range=[0.7, 1.0]),
                dict(label="F1分数", values=df["F1分数"], range=[0.8, 1.0]),
            ],
        ))

        layout = self._get_layout(
            title="模型性能平行坐标图",
            title_en="Model Performance Parallel Coordinates"
        )
        fig.update_layout(**layout)

        return self._save_figure(fig, filename)

    # ========================================================================
    # 异常检测可视化
    # ========================================================================

    def anomaly_timeline(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        energy: np.ndarray = None,
        threshold: float = None,
        max_points: int = 5000,
        filename: str = "异常检测时序图"
    ) -> str:
        """
        生成异常检测时序交互图

        Args:
            data: 原始数据 (T, features) 或 (T,)
            predictions: 预测标签 (T,)
            ground_truth: 真实标签 (T,)
            energy: 异常分数 (T,)
            threshold: 异常阈值
            max_points: 最大显示点数
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        # 数据采样
        if len(predictions) > max_points:
            step = len(predictions) // max_points
            indices = np.arange(0, len(predictions), step)
            predictions = predictions[indices]
            ground_truth = ground_truth[indices]
            if energy is not None:
                energy = energy[indices]
            if data is not None and len(data) == len(indices) * step:
                data = data[indices]

        n_rows = 3 if energy is not None else 2
        subplot_titles = ["原始信号 (Signal)", "检测结果 (Detection Results)"]
        if energy is not None:
            subplot_titles.append("异常分数 (Anomaly Score)")

        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3] if n_rows == 3 else [0.5, 0.5],
        )

        x = np.arange(len(predictions))

        # 原始信号（如果提供）
        if data is not None:
            if data.ndim == 2:
                for i in range(min(data.shape[1], 3)):
                    fig.add_trace(
                        go.Scatter(x=x, y=data[:, i], name=f"特征 {i+1}",
                                  mode="lines", line=dict(width=1)),
                        row=1, col=1
                    )
            else:
                fig.add_trace(
                    go.Scatter(x=x, y=data, name="信号", mode="lines"),
                    row=1, col=1
                )

        # 检测结果
        fig.add_trace(
            go.Scatter(x=x, y=ground_truth, name="真实标签",
                      mode="lines", fill="tozeroy",
                      line=dict(color=ANOMALY_COLORS[1], width=0),
                      fillcolor="rgba(214, 39, 40, 0.3)"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=predictions * 0.8, name="预测标签",
                      mode="lines", line=dict(color="#1f77b4", width=2)),
            row=2, col=1
        )

        # 异常分数
        if energy is not None:
            fig.add_trace(
                go.Scatter(x=x, y=energy, name="异常分数",
                          mode="lines", line=dict(color="#ff7f0e", width=1)),
                row=3, col=1
            )
            if threshold is not None:
                fig.add_hline(y=threshold, line_dash="dash",
                             line_color="red", row=3, col=1,
                             annotation_text=f"阈值: {threshold:.4f}")

        layout = self._get_layout(
            title="异常检测结果时序图",
            title_en="Anomaly Detection Timeline"
        )
        layout["height"] = 800
        layout["xaxis3" if n_rows == 3 else "xaxis2"]["title"] = "时间步 (Time Step)"

        fig.update_layout(**layout)

        return self._save_figure(fig, filename)

    def error_distribution_3d(
        self,
        features: np.ndarray,
        error: np.ndarray,
        labels: np.ndarray,
        filename: str = "3D误差分布图"
    ) -> str:
        """
        生成 3D 误差分布图

        Args:
            features: 特征数据 (N, 2) 或 (N, 3)
            error: 误差分数 (N,)
            labels: 标签 (N,)
            filename: 输出文件名

        Returns:
            str: 保存的文件路径
        """
        # 如果特征维度不是2或3，使用PCA降维
        if features.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            features = pca.fit_transform(features)
        elif features.shape[1] == 2:
            features = np.column_stack([features, error])

        fig = go.Figure()

        # 分别绘制正常和异常点
        for label, name, color in [(0, "正常", ANOMALY_COLORS[0]), (1, "异常", ANOMALY_COLORS[1])]:
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=features[mask, 0],
                y=features[mask, 1],
                z=features[mask, 2] if features.shape[1] > 2 else error[mask],
                mode="markers",
                name=name,
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.6,
                ),
            ))

        layout = self._get_layout(
            title="特征空间异常分布 3D 图",
            title_en="3D Anomaly Distribution in Feature Space"
        )
        layout["scene"] = {
            "xaxis_title": "特征 1",
            "yaxis_title": "特征 2",
            "zaxis_title": "误差分数" if features.shape[1] == 2 else "特征 3",
        }

        fig.update_layout(**layout)

        return self._save_figure(fig, filename)

    # ========================================================================
    # 保存函数
    # ========================================================================

    def _save_figure(self, fig, filename: str) -> str:
        """
        保存图表

        Args:
            fig: Plotly 图表对象
            filename: 文件名（不含扩展名）

        Returns:
            str: 保存的文件路径
        """
        # 保存 HTML
        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")

        # 尝试保存静态图片
        try:
            png_path = self.output_dir / f"{filename}.png"
            fig.write_image(str(png_path), width=self.width, height=self.height, scale=2)
        except Exception as e:
            print(f"警告: 无法保存静态图片 ({e})，请安装 kaleido: pip install kaleido")

        return str(html_path)

    def save_html(self, fig, filename: str) -> str:
        """导出为 HTML"""
        path = self.output_dir / f"{filename}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        return str(path)

    def save_static(self, fig, filename: str, format: str = "png") -> str:
        """导出为静态图片"""
        path = self.output_dir / f"{filename}.{format}"
        fig.write_image(str(path), width=self.width, height=self.height, scale=2)
        return str(path)
