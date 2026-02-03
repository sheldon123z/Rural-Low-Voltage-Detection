"""
检测结果可视化模块
农村低压配电网电压异常检测项目

本模块提供基于 Plotly 的交互式检测结果可视化功能。

Functions:
    create_detection_timeline: 检测结果时间线图
    create_anomaly_heatmap: 异常热力图
    create_score_distribution: 异常分数分布图
    create_detection_summary: 检测结果摘要图

Author: Rural Voltage Detection Project
Date: 2026
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None


# ============================================================================
# 配色方案
# ============================================================================

THESIS_COLORS = {
    "anomaly": "#E74C3C",      # 异常红
    "normal": "#2ECC71",       # 正常绿
    "primary": "#4878A8",      # 柔和蓝
    "secondary": "#72A86D",    # 柔和绿
    "accent": "#C4785C",       # 柔和橙
    "warning": "#D4A84C",      # 柔和黄
    "neutral": "#808080",      # 中性灰
    "light_gray": "#B0B0B0",   # 浅灰
    "threshold": "#9B59B6",    # 阈值紫
}

# 中文字体配置
PLOTLY_FONT = "WenQuanYi Micro Hei, Noto Sans CJK SC, Microsoft YaHei, SimSun, sans-serif"


# ============================================================================
# 检测结果时间线图
# ============================================================================

def create_detection_timeline(
    data: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    feature_names: Optional[List[str]] = None,
    max_features: int = 3,
    title: str = "异常检测结果时间线",
) -> "go.Figure":
    """
    创建检测结果时间线图

    Args:
        data: 原始数据，形状为 (T, C) 或 (T,)
        scores: 异常分数，形状为 (T,)
        labels: 预测标签，形状为 (T,)，1=异常，0=正常
        threshold: 异常判定阈值
        feature_names: 特征名称列表
        max_features: 最多显示的特征数量
        title: 图表标题

    Returns:
        go.Figure: Plotly 图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly 未安装，请运行: pip install plotly")

    # 数据预处理
    data = np.asarray(data)
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, C = data.shape
    time_steps = np.arange(T)

    # 限制特征数量
    n_features = min(C, max_features)

    # 默认特征名称
    if feature_names is None:
        feature_names = [f"特征 {i+1}" for i in range(n_features)]
    else:
        feature_names = feature_names[:n_features]

    # 创建子图：时序数据 + 异常分数
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=["时序数据", "异常分数"],
    )

    # 找到异常区域
    anomaly_regions = _find_anomaly_regions(labels)

    # 绘制时序数据
    colors = [THESIS_COLORS["primary"], THESIS_COLORS["secondary"], THESIS_COLORS["accent"]]
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=data[:, i],
                mode="lines",
                name=feature_names[i],
                line=dict(color=colors[i % len(colors)], width=1.5),
                hovertemplate=f"{feature_names[i]}<br>时间步: %{{x}}<br>值: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1
        )

    # 高亮异常区域（红色背景）
    for start, end in anomaly_regions:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=THESIS_COLORS["anomaly"],
            opacity=0.2,
            layer="below",
            line_width=0,
            row=1, col=1,
        )
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=THESIS_COLORS["anomaly"],
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2, col=1,
        )

    # 绘制异常分数曲线
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=scores,
            mode="lines",
            name="异常分数",
            line=dict(color=THESIS_COLORS["accent"], width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(THESIS_COLORS['accent'][1:3], 16)}, "
                      f"{int(THESIS_COLORS['accent'][3:5], 16)}, "
                      f"{int(THESIS_COLORS['accent'][5:7], 16)}, 0.3)",
            hovertemplate="时间步: %{x}<br>异常分数: %{y:.4f}<extra></extra>",
        ),
        row=2, col=1
    )

    # 绘制阈值线
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=THESIS_COLORS["threshold"],
        line_width=2,
        annotation_text=f"阈值: {threshold:.4f}",
        annotation_position="top right",
        annotation_font=dict(color=THESIS_COLORS["threshold"], size=11),
        row=2, col=1,
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=PLOTLY_FONT, size=16),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family=PLOTLY_FONT, size=12),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="时间步",
        row=2, col=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    fig.update_yaxes(
        title_text="数值",
        row=1, col=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    fig.update_yaxes(
        title_text="异常分数",
        row=2, col=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    return fig


# ============================================================================
# 异常热力图
# ============================================================================

def create_anomaly_heatmap(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "多通道异常分布热力图",
) -> "go.Figure":
    """
    创建异常热力图

    显示多通道数据的异常分布情况。

    Args:
        data: 原始数据，形状为 (T, C)
        labels: 预测标签，形状为 (T,)，1=异常，0=正常
        feature_names: 特征名称列表
        title: 图表标题

    Returns:
        go.Figure: Plotly 图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly 未安装，请运行: pip install plotly")

    # 数据预处理
    data = np.asarray(data)
    labels = np.asarray(labels).flatten()

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, C = data.shape

    # 默认特征名称
    if feature_names is None:
        feature_names = [f"通道 {i+1}" for i in range(C)]

    # 标准化数据用于热力图显示
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.85, 0.15],
        subplot_titles=["数据热力图", "异常标签"],
    )

    # 绘制热力图
    fig.add_trace(
        go.Heatmap(
            z=data_normalized.T,
            x=np.arange(T),
            y=feature_names,
            colorscale=[
                [0.0, THESIS_COLORS["normal"]],
                [0.5, THESIS_COLORS["warning"]],
                [1.0, THESIS_COLORS["anomaly"]],
            ],
            colorbar=dict(
                title="归一化值",
                titleside="right",
                tickfont=dict(size=10),
            ),
            hovertemplate="时间步: %{x}<br>通道: %{y}<br>归一化值: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1
    )

    # 绘制异常标签条
    label_colors = np.where(labels == 1, 1, 0).reshape(1, -1)
    fig.add_trace(
        go.Heatmap(
            z=label_colors,
            x=np.arange(T),
            y=["标签"],
            colorscale=[
                [0.0, THESIS_COLORS["normal"]],
                [1.0, THESIS_COLORS["anomaly"]],
            ],
            showscale=False,
            hovertemplate="时间步: %{x}<br>状态: %{customdata}<extra></extra>",
            customdata=[["异常" if l == 1 else "正常" for l in labels]],
        ),
        row=2, col=1
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=PLOTLY_FONT, size=16),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family=PLOTLY_FONT, size=12),
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="时间步",
        row=2, col=1,
        showgrid=False,
        linecolor="black",
    )

    fig.update_yaxes(
        showgrid=False,
        linecolor="black",
    )

    return fig


# ============================================================================
# 异常分数分布图
# ============================================================================

def create_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    bins: int = 50,
    title: str = "异常分数分布",
) -> "go.Figure":
    """
    创建异常分数分布图

    区分正常和异常样本的分数分布，并显示阈值位置。

    Args:
        scores: 异常分数，形状为 (N,)
        labels: 真实标签，形状为 (N,)，1=异常，0=正常
        threshold: 异常判定阈值，如果为 None 则自动计算
        bins: 直方图的 bin 数量
        title: 图表标题

    Returns:
        go.Figure: Plotly 图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly 未安装，请运行: pip install plotly")

    # 数据预处理
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    # 分离正常和异常样本的分数
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    # 自动计算阈值（如果未提供）
    if threshold is None:
        threshold = np.percentile(normal_scores, 95) if len(normal_scores) > 0 else np.median(scores)

    # 计算统计信息
    n_normal = len(normal_scores)
    n_anomaly = len(anomaly_scores)

    # 创建图表
    fig = go.Figure()

    # 确定 bin 范围
    score_min = min(scores.min(), 0)
    score_max = scores.max() * 1.1
    bin_edges = np.linspace(score_min, score_max, bins + 1)

    # 绘制正常样本直方图
    if len(normal_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=normal_scores,
                name=f"正常样本 (n={n_normal})",
                xbins=dict(start=score_min, end=score_max, size=(score_max - score_min) / bins),
                marker_color=THESIS_COLORS["normal"],
                opacity=0.7,
                hovertemplate="分数区间: %{x}<br>样本数: %{y}<extra>正常样本</extra>",
            )
        )

    # 绘制异常样本直方图
    if len(anomaly_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                name=f"异常样本 (n={n_anomaly})",
                xbins=dict(start=score_min, end=score_max, size=(score_max - score_min) / bins),
                marker_color=THESIS_COLORS["anomaly"],
                opacity=0.7,
                hovertemplate="分数区间: %{x}<br>样本数: %{y}<extra>异常样本</extra>",
            )
        )

    # 绘制阈值线
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=THESIS_COLORS["threshold"],
        line_width=2,
        annotation_text=f"阈值: {threshold:.4f}",
        annotation_position="top",
        annotation_font=dict(color=THESIS_COLORS["threshold"], size=11),
    )

    # 添加分类区域标注
    fig.add_annotation(
        x=threshold / 2,
        y=0.95,
        yref="paper",
        text="预测正常",
        showarrow=False,
        font=dict(size=12, color=THESIS_COLORS["normal"]),
    )

    fig.add_annotation(
        x=(threshold + score_max) / 2,
        y=0.95,
        yref="paper",
        text="预测异常",
        showarrow=False,
        font=dict(size=12, color=THESIS_COLORS["anomaly"]),
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=PLOTLY_FONT, size=16),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family=PLOTLY_FONT, size=12),
        xaxis_title="异常分数",
        yaxis_title="样本数量",
        barmode="overlay",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    return fig


# ============================================================================
# 检测结果摘要图
# ============================================================================

def create_detection_summary(
    results: Dict[str, Any],
    title: str = "检测结果摘要",
) -> "go.Figure":
    """
    创建检测结果摘要图

    包含关键指标的综合展示。

    Args:
        results: 检测结果字典，包含以下字段：
            - accuracy: 准确率
            - precision: 精确率
            - recall: 召回率
            - f1_score: F1 分数
            - tp: 真阳性数量 (可选)
            - fp: 假阳性数量 (可选)
            - tn: 真阴性数量 (可选)
            - fn: 假阴性数量 (可选)
            - threshold: 使用的阈值 (可选)
            - total_samples: 总样本数 (可选)
            - anomaly_ratio: 异常比例 (可选)
        title: 图表标题

    Returns:
        go.Figure: Plotly 图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly 未安装，请运行: pip install plotly")

    # 提取指标
    accuracy = results.get("accuracy", 0)
    precision = results.get("precision", 0)
    recall = results.get("recall", 0)
    f1_score = results.get("f1_score", 0)

    # 混淆矩阵数据
    tp = results.get("tp", 0)
    fp = results.get("fp", 0)
    tn = results.get("tn", 0)
    fn = results.get("fn", 0)
    has_confusion = any([tp, fp, tn, fn])

    # 创建子图
    if has_confusion:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["性能指标", "混淆矩阵"],
            specs=[[{"type": "bar"}, {"type": "heatmap"}]],
            column_widths=[0.5, 0.5],
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["性能指标"],
            specs=[[{"type": "bar"}]],
        )

    # 绘制性能指标柱状图
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    values = [accuracy, precision, recall, f1_score]
    colors = [THESIS_COLORS["primary"], THESIS_COLORS["secondary"],
              THESIS_COLORS["accent"], THESIS_COLORS["warning"]]

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1
    )

    # 绘制混淆矩阵
    if has_confusion:
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        labels_text = [["TN", "FP"], ["FN", "TP"]]

        # 创建带数值的热力图
        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                x=["预测正常", "预测异常"],
                y=["实际正常", "实际异常"],
                colorscale=[
                    [0.0, "#FFFFFF"],
                    [0.5, THESIS_COLORS["warning"]],
                    [1.0, THESIS_COLORS["anomaly"]],
                ],
                showscale=False,
                hovertemplate="预测: %{x}<br>实际: %{y}<br>数量: %{z}<extra></extra>",
            ),
            row=1, col=2
        )

        # 添加数值标注
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(
                    dict(
                        x=["预测正常", "预测异常"][j],
                        y=["实际正常", "实际异常"][i],
                        text=f"{labels_text[i][j]}<br>{confusion_matrix[i, j]}",
                        showarrow=False,
                        font=dict(size=14, color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white"),
                        xref="x2",
                        yref="y2",
                    )
                )
        fig.update_layout(annotations=annotations)

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=PLOTLY_FONT, size=16),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family=PLOTLY_FONT, size=12),
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_yaxes(
        range=[0, 1.1],
        row=1, col=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        linecolor="black",
    )

    fig.update_xaxes(
        linecolor="black",
        row=1, col=1,
    )

    # 添加额外信息注释
    extra_info = []
    if "threshold" in results:
        extra_info.append(f"阈值: {results['threshold']:.4f}")
    if "total_samples" in results:
        extra_info.append(f"总样本: {results['total_samples']}")
    if "anomaly_ratio" in results:
        extra_info.append(f"异常比例: {results['anomaly_ratio']:.2%}")

    if extra_info:
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<br>".join(extra_info),
            showarrow=False,
            font=dict(size=10, color=THESIS_COLORS["neutral"]),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=THESIS_COLORS["light_gray"],
            borderwidth=1,
        )

    return fig


# ============================================================================
# 辅助函数
# ============================================================================

def _find_anomaly_regions(labels: np.ndarray) -> List[tuple]:
    """
    找到连续的异常区域

    Args:
        labels: 标签数组，1=异常，0=正常

    Returns:
        List[tuple]: 异常区域列表，每个元素为 (start, end)
    """
    labels = np.asarray(labels).flatten()
    regions = []

    # 找到变化点
    changes = np.diff(labels.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # 处理边界情况
    if labels[0] == 1:
        starts = np.insert(starts, 0, 0)
    if labels[-1] == 1:
        ends = np.append(ends, len(labels))

    for s, e in zip(starts, ends):
        regions.append((s, e))

    return regions


# ============================================================================
# 导出接口
# ============================================================================

__all__ = [
    "create_detection_timeline",
    "create_anomaly_heatmap",
    "create_score_distribution",
    "create_detection_summary",
    "THESIS_COLORS",
]
