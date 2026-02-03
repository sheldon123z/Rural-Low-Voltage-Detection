"""
模型对比可视化模块

提供基于 Plotly 的交互式模型对比图表，包括雷达图、柱状图和指标表格。

Author: Rural Voltage Detection Project
Date: 2026
"""

from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    warnings.warn("Plotly 未安装，交互式图表功能不可用。请运行: pip install plotly")


# ============================================================================
# 配置常量
# ============================================================================

# 中文字体配置
PLOTLY_FONT = "WenQuanYi Micro Hei, Noto Sans CJK SC, Microsoft YaHei, sans-serif"

# 模型配色方案（柔和科研风格）
MODEL_COLORS = {
    "VoltageTimesNet_v2": "#4878A8",  # 柔和蓝
    "VoltageTimesNet": "#72A86D",     # 柔和绿
    "TimesNet": "#C4785C",            # 柔和橙
    "TPATimesNet": "#D4A84C",         # 柔和黄
    "MTSTimesNet": "#9B59B6",         # 柔和紫
    "DLinear": "#808080",             # 中性灰
}

# 默认配色（用于未知模型）
DEFAULT_COLORS = [
    "#4878A8", "#72A86D", "#C4785C", "#D4A84C",
    "#9B59B6", "#808080", "#17becf", "#e377c2"
]

# 指标名称映射（英文到中文）
METRIC_NAMES = {
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1分数",
    "f1_score": "F1分数",
    "accuracy": "准确率",
    "auc": "AUC",
    "roc_auc": "AUC",
}

# 指标顺序（雷达图显示顺序）
DEFAULT_METRICS = ["precision", "recall", "f1", "accuracy", "auc"]


# ============================================================================
# 辅助函数
# ============================================================================

def _check_plotly():
    """检查 Plotly 是否可用"""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly 未安装，请运行: pip install plotly kaleido\n"
            "如需导出静态图片，还需要安装 kaleido: pip install kaleido"
        )


def _get_model_color(model_name: str, index: int = 0) -> str:
    """
    获取模型对应的颜色

    Args:
        model_name: 模型名称
        index: 如果模型不在预定义配色中，使用索引选择默认颜色

    Returns:
        str: 颜色十六进制值
    """
    return MODEL_COLORS.get(model_name, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def _get_metric_chinese_name(metric: str) -> str:
    """
    获取指标的中文名称

    Args:
        metric: 英文指标名

    Returns:
        str: 中文名称
    """
    return METRIC_NAMES.get(metric.lower(), metric)


def _normalize_metrics(metrics_dict: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    标准化指标字典，将所有指标名转为小写，处理常见别名

    Args:
        metrics_dict: 原始指标字典

    Returns:
        Dict: 标准化后的指标字典
    """
    normalized = {}
    alias_map = {
        "f1_score": "f1",
        "f1-score": "f1",
        "roc_auc": "auc",
        "auroc": "auc",
    }

    for model, metrics in metrics_dict.items():
        normalized[model] = {}
        for key, value in metrics.items():
            key_lower = key.lower()
            key_normalized = alias_map.get(key_lower, key_lower)
            normalized[model][key_normalized] = value

    return normalized


def _get_common_layout(title: str = "", title_en: str = "") -> Dict:
    """
    获取通用布局配置

    Args:
        title: 中文标题
        title_en: 英文标题（可选）

    Returns:
        Dict: 布局配置
    """
    full_title = f"{title}<br><sup>{title_en}</sup>" if title_en else title

    return {
        "title": {
            "text": full_title,
            "font": {"family": PLOTLY_FONT, "size": 16},
            "x": 0.5,
            "xanchor": "center",
        },
        "font": {"family": PLOTLY_FONT, "size": 12},
        "legend": {
            "font": {"size": 11},
            "bordercolor": "black",
            "borderwidth": 1,
            "bgcolor": "rgba(255,255,255,0.9)",
        },
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "margin": {"l": 60, "r": 60, "t": 80, "b": 60},
    }


# ============================================================================
# 主要可视化函数
# ============================================================================

def create_radar_chart(
    metrics_dict: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    title: str = "模型性能雷达图对比",
    title_en: str = "Model Performance Radar Chart",
    fill_opacity: float = 0.25,
    line_width: float = 2.5,
    height: int = 500,
    width: int = 600,
) -> go.Figure:
    """
    创建雷达图对比多个模型的性能指标

    Args:
        metrics_dict: 模型指标字典
            格式: {"model_name": {"precision": 0.9, "recall": 0.85, ...}, ...}
        metrics: 要显示的指标列表，默认 ["precision", "recall", "f1", "accuracy", "auc"]
        title: 中文标题
        title_en: 英文标题
        fill_opacity: 填充透明度
        line_width: 线条宽度
        height: 图表高度
        width: 图表宽度

    Returns:
        go.Figure: Plotly 图表对象

    Example:
        >>> metrics = {
        ...     "VoltageTimesNet_v2": {"precision": 0.76, "recall": 0.59, "f1": 0.66, "accuracy": 0.91, "auc": 0.85},
        ...     "TimesNet": {"precision": 0.76, "recall": 0.57, "f1": 0.65, "accuracy": 0.91, "auc": 0.83},
        ... }
        >>> fig = create_radar_chart(metrics)
        >>> fig.show()
    """
    _check_plotly()

    # 标准化指标名
    metrics_dict = _normalize_metrics(metrics_dict)

    # 确定要显示的指标
    if metrics is None:
        # 自动检测共有指标
        all_metrics = set()
        for model_metrics in metrics_dict.values():
            all_metrics.update(model_metrics.keys())
        # 按默认顺序排列
        metrics = [m for m in DEFAULT_METRICS if m in all_metrics]
        # 添加其他指标
        for m in all_metrics:
            if m not in metrics:
                metrics.append(m)

    if not metrics:
        raise ValueError("未找到有效指标")

    # 中文指标名
    categories = [_get_metric_chinese_name(m) for m in metrics]
    # 闭合多边形
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    for i, (model, model_metrics) in enumerate(metrics_dict.items()):
        # 获取指标值
        values = [model_metrics.get(m, 0) for m in metrics]
        values_closed = values + [values[0]]  # 闭合多边形

        color = _get_model_color(model, i)

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor=color,
            opacity=fill_opacity,
            line=dict(color=color, width=line_width),
            name=model,
            hovertemplate=(
                f"<b>{model}</b><br>"
                "%{theta}: %{r:.4f}<extra></extra>"
            ),
        ))

    # 布局配置
    layout = _get_common_layout(title, title_en)
    layout.update({
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "tickvals": [0.2, 0.4, 0.6, 0.8, 1.0],
                "tickfont": {"size": 10},
                "gridcolor": "#e0e0e0",
                "linecolor": "#808080",
            },
            "angularaxis": {
                "tickfont": {"size": 11, "family": PLOTLY_FONT},
                "gridcolor": "#e0e0e0",
                "linecolor": "#808080",
            },
            "bgcolor": "white",
        },
        "height": height,
        "width": width,
    })

    fig.update_layout(**layout)

    return fig


def create_bar_chart(
    metrics_dict: Dict[str, Dict],
    metric_name: str = "f1",
    sort: bool = True,
    ascending: bool = False,
    title: Optional[str] = None,
    title_en: Optional[str] = None,
    show_values: bool = True,
    height: int = 400,
    width: int = 600,
    orientation: str = "v",
) -> go.Figure:
    """
    创建柱状图对比指定指标

    Args:
        metrics_dict: 模型指标字典
            格式: {"model_name": {"precision": 0.9, "recall": 0.85, ...}, ...}
        metric_name: 要对比的指标名（如 "f1", "recall", "precision"）
        sort: 是否按值排序
        ascending: 是否升序排列
        title: 中文标题（默认自动生成）
        title_en: 英文标题（默认自动生成）
        show_values: 是否在柱子上显示数值
        height: 图表高度
        width: 图表宽度
        orientation: 柱状图方向，"v" 垂直或 "h" 水平

    Returns:
        go.Figure: Plotly 图表对象

    Example:
        >>> metrics = {
        ...     "VoltageTimesNet_v2": {"f1": 0.66},
        ...     "TimesNet": {"f1": 0.65},
        ... }
        >>> fig = create_bar_chart(metrics, metric_name="f1")
        >>> fig.show()
    """
    _check_plotly()

    # 标准化指标名
    metrics_dict = _normalize_metrics(metrics_dict)
    metric_name = metric_name.lower()

    # 别名处理
    alias_map = {"f1_score": "f1", "roc_auc": "auc"}
    metric_name = alias_map.get(metric_name, metric_name)

    # 提取数据
    data = []
    for model, model_metrics in metrics_dict.items():
        value = model_metrics.get(metric_name, None)
        if value is not None:
            data.append({"model": model, "value": value})

    if not data:
        raise ValueError(f"未找到指标 '{metric_name}'")

    df = pd.DataFrame(data)

    # 排序
    if sort:
        df = df.sort_values("value", ascending=ascending)

    # 获取颜色
    colors = [_get_model_color(model, i) for i, model in enumerate(df["model"])]

    # 默认标题
    metric_cn = _get_metric_chinese_name(metric_name)
    if title is None:
        title = f"模型{metric_cn}对比"
    if title_en is None:
        title_en = f"Model {metric_name.upper()} Comparison"

    # 创建图表
    if orientation == "v":
        fig = go.Figure(go.Bar(
            x=df["model"],
            y=df["value"],
            marker_color=colors,
            text=df["value"].apply(lambda x: f"{x:.4f}") if show_values else None,
            textposition="outside" if show_values else None,
            hovertemplate="<b>%{x}</b><br>" + f"{metric_cn}: " + "%{y:.4f}<extra></extra>",
        ))
        xaxis_title = "模型 (Model)"
        yaxis_title = f"{metric_cn} ({metric_name.upper()})"
    else:
        fig = go.Figure(go.Bar(
            y=df["model"],
            x=df["value"],
            orientation="h",
            marker_color=colors,
            text=df["value"].apply(lambda x: f"{x:.4f}") if show_values else None,
            textposition="outside" if show_values else None,
            hovertemplate="<b>%{y}</b><br>" + f"{metric_cn}: " + "%{x:.4f}<extra></extra>",
        ))
        xaxis_title = f"{metric_cn} ({metric_name.upper()})"
        yaxis_title = "模型 (Model)"

    # 布局
    layout = _get_common_layout(title, title_en)
    layout.update({
        "xaxis": {
            "title": xaxis_title,
            "title_font": {"size": 12},
            "tickfont": {"size": 10, "family": PLOTLY_FONT},
            "gridcolor": "#e0e0e0",
            "linecolor": "black",
            "mirror": True,
        },
        "yaxis": {
            "title": yaxis_title,
            "title_font": {"size": 12},
            "tickfont": {"size": 10, "family": PLOTLY_FONT},
            "gridcolor": "#e0e0e0",
            "linecolor": "black",
            "mirror": True,
            "range": [0, max(df["value"]) * 1.15] if orientation == "v" else None,
        },
        "height": height,
        "width": width,
        "showlegend": False,
    })

    fig.update_layout(**layout)

    return fig


def create_metrics_table(
    metrics_dict: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    sort_by: Optional[str] = "f1",
    ascending: bool = False,
    precision: int = 4,
) -> pd.DataFrame:
    """
    创建详细指标表格

    Args:
        metrics_dict: 模型指标字典
            格式: {"model_name": {"precision": 0.9, "recall": 0.85, ...}, ...}
        metrics: 要显示的指标列表（默认自动检测）
        sort_by: 按哪个指标排序（默认 "f1"）
        ascending: 是否升序排列
        precision: 小数位数

    Returns:
        pd.DataFrame: 指标表格

    Example:
        >>> metrics = {
        ...     "VoltageTimesNet_v2": {"precision": 0.76, "recall": 0.59, "f1": 0.66},
        ...     "TimesNet": {"precision": 0.76, "recall": 0.57, "f1": 0.65},
        ... }
        >>> df = create_metrics_table(metrics)
        >>> print(df)
    """
    # 标准化指标名
    metrics_dict = _normalize_metrics(metrics_dict)

    # 确定要显示的指标
    if metrics is None:
        all_metrics = set()
        for model_metrics in metrics_dict.values():
            all_metrics.update(model_metrics.keys())
        metrics = [m for m in DEFAULT_METRICS if m in all_metrics]
        for m in all_metrics:
            if m not in metrics:
                metrics.append(m)

    # 构建数据
    rows = []
    for model, model_metrics in metrics_dict.items():
        row = {"模型": model}
        for m in metrics:
            col_name = _get_metric_chinese_name(m)
            value = model_metrics.get(m, None)
            if value is not None:
                row[col_name] = round(value, precision)
            else:
                row[col_name] = None
        rows.append(row)

    df = pd.DataFrame(rows)

    # 排序
    if sort_by:
        sort_col = _get_metric_chinese_name(sort_by.lower())
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=ascending)

    # 重置索引
    df = df.reset_index(drop=True)

    return df


def create_model_comparison_plot(
    model_results: Dict[str, Dict],
    include_radar: bool = True,
    include_bar: bool = True,
    bar_metric: str = "f1",
    height: int = 500,
    width: int = 1000,
) -> go.Figure:
    """
    创建综合模型对比图（包含多个子图）

    Args:
        model_results: 模型结果字典
            格式: {"model_name": {"precision": 0.9, "recall": 0.85, ...}, ...}
        include_radar: 是否包含雷达图
        include_bar: 是否包含柱状图
        bar_metric: 柱状图显示的指标
        height: 图表高度
        width: 图表宽度

    Returns:
        go.Figure: Plotly 图表对象

    Example:
        >>> results = {
        ...     "VoltageTimesNet_v2": {"precision": 0.76, "recall": 0.59, "f1": 0.66, "accuracy": 0.91},
        ...     "TimesNet": {"precision": 0.76, "recall": 0.57, "f1": 0.65, "accuracy": 0.91},
        ... }
        >>> fig = create_model_comparison_plot(results)
        >>> fig.show()
    """
    _check_plotly()

    # 标准化指标名
    model_results = _normalize_metrics(model_results)

    # 确定子图数量
    n_subplots = sum([include_radar, include_bar])
    if n_subplots == 0:
        raise ValueError("至少需要包含一个子图")

    # 创建子图
    specs = []
    subplot_titles = []

    if include_radar:
        specs.append([{"type": "polar"}])
        subplot_titles.append("性能雷达图 (Performance Radar)")

    if include_bar:
        specs.append([{"type": "xy"}])
        metric_cn = _get_metric_chinese_name(bar_metric)
        subplot_titles.append(f"{metric_cn}对比 ({bar_metric.upper()} Comparison)")

    fig = make_subplots(
        rows=n_subplots, cols=1,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        row_heights=[1.0 / n_subplots] * n_subplots,
    )

    row_idx = 1

    # 雷达图
    if include_radar:
        # 获取共有指标
        all_metrics = set()
        for model_metrics in model_results.values():
            all_metrics.update(model_metrics.keys())
        metrics = [m for m in DEFAULT_METRICS if m in all_metrics]

        categories = [_get_metric_chinese_name(m) for m in metrics]
        categories_closed = categories + [categories[0]]

        for i, (model, model_metrics) in enumerate(model_results.items()):
            values = [model_metrics.get(m, 0) for m in metrics]
            values_closed = values + [values[0]]

            color = _get_model_color(model, i)

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    fillcolor=color,
                    opacity=0.25,
                    line=dict(color=color, width=2.5),
                    name=model,
                    legendgroup=model,
                    showlegend=True,
                ),
                row=row_idx, col=1
            )

        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont={"size": 9},
                gridcolor="#e0e0e0",
            ),
            angularaxis=dict(
                tickfont={"size": 10, "family": PLOTLY_FONT},
                gridcolor="#e0e0e0",
            ),
            bgcolor="white",
        )
        row_idx += 1

    # 柱状图
    if include_bar:
        bar_metric_lower = bar_metric.lower()
        alias_map = {"f1_score": "f1", "roc_auc": "auc"}
        bar_metric_lower = alias_map.get(bar_metric_lower, bar_metric_lower)

        # 提取数据并排序
        bar_data = []
        for model, model_metrics in model_results.items():
            value = model_metrics.get(bar_metric_lower, None)
            if value is not None:
                bar_data.append({"model": model, "value": value})

        bar_data.sort(key=lambda x: x["value"], reverse=True)

        models = [d["model"] for d in bar_data]
        values = [d["value"] for d in bar_data]
        colors = [_get_model_color(m, i) for i, m in enumerate(models)]

        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
                showlegend=False,
            ),
            row=row_idx, col=1
        )

        metric_cn = _get_metric_chinese_name(bar_metric)
        fig.update_xaxes(
            title_text="模型 (Model)",
            tickfont={"size": 10, "family": PLOTLY_FONT},
            row=row_idx, col=1
        )
        fig.update_yaxes(
            title_text=f"{metric_cn} ({bar_metric.upper()})",
            tickfont={"size": 10},
            range=[0, max(values) * 1.15] if values else [0, 1],
            row=row_idx, col=1
        )

    # 全局布局
    layout = _get_common_layout(
        "模型综合性能对比",
        "Comprehensive Model Performance Comparison"
    )
    layout.update({
        "height": height,
        "width": width,
        "legend": {
            "font": {"size": 10, "family": PLOTLY_FONT},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    })

    fig.update_layout(**layout)

    return fig


# ============================================================================
# 额外工具函数
# ============================================================================

def create_grouped_bar_chart(
    metrics_dict: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    title: str = "多指标模型对比",
    title_en: str = "Multi-Metric Model Comparison",
    height: int = 400,
    width: int = 800,
) -> go.Figure:
    """
    创建分组柱状图，同时对比多个指标

    Args:
        metrics_dict: 模型指标字典
        metrics: 要显示的指标列表
        title: 中文标题
        title_en: 英文标题
        height: 图表高度
        width: 图表宽度

    Returns:
        go.Figure: Plotly 图表对象
    """
    _check_plotly()

    # 标准化指标名
    metrics_dict = _normalize_metrics(metrics_dict)

    # 确定指标
    if metrics is None:
        all_metrics = set()
        for model_metrics in metrics_dict.values():
            all_metrics.update(model_metrics.keys())
        metrics = [m for m in DEFAULT_METRICS if m in all_metrics]

    models = list(metrics_dict.keys())

    fig = go.Figure()

    # 指标颜色
    metric_colors = DEFAULT_COLORS[:len(metrics)]

    for i, metric in enumerate(metrics):
        values = [metrics_dict[m].get(metric, 0) for m in models]
        metric_cn = _get_metric_chinese_name(metric)

        fig.add_trace(go.Bar(
            name=metric_cn,
            x=models,
            y=values,
            marker_color=metric_colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    layout = _get_common_layout(title, title_en)
    layout.update({
        "barmode": "group",
        "xaxis": {
            "title": "模型 (Model)",
            "tickfont": {"size": 10, "family": PLOTLY_FONT},
        },
        "yaxis": {
            "title": "指标值 (Metric Value)",
            "tickfont": {"size": 10},
            "range": [0, 1.1],
        },
        "height": height,
        "width": width,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
    })

    fig.update_layout(**layout)

    return fig


def export_figure(
    fig: go.Figure,
    filepath: str,
    format: str = "html",
    width: int = None,
    height: int = None,
    scale: float = 2,
) -> str:
    """
    导出图表到文件

    Args:
        fig: Plotly 图表对象
        filepath: 保存路径（不含扩展名）
        format: 格式 ("html", "png", "pdf", "svg")
        width: 图片宽度（仅静态图片）
        height: 图片高度（仅静态图片）
        scale: 缩放比例（仅静态图片）

    Returns:
        str: 保存的文件路径
    """
    _check_plotly()

    if format == "html":
        output_path = f"{filepath}.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
    else:
        output_path = f"{filepath}.{format}"
        try:
            fig.write_image(
                output_path,
                width=width,
                height=height,
                scale=scale,
            )
        except Exception as e:
            raise RuntimeError(
                f"导出静态图片失败: {e}\n"
                "请确保已安装 kaleido: pip install kaleido"
            )

    return output_path


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    # 测试数据
    test_metrics = {
        "VoltageTimesNet_v2": {
            "precision": 0.7614,
            "recall": 0.5858,
            "f1": 0.6622,
            "accuracy": 0.9119,
            "auc": 0.8523,
        },
        "VoltageTimesNet": {
            "precision": 0.7541,
            "recall": 0.5726,
            "f1": 0.6509,
            "accuracy": 0.9094,
            "auc": 0.8412,
        },
        "TimesNet": {
            "precision": 0.7606,
            "recall": 0.5705,
            "f1": 0.6520,
            "accuracy": 0.9102,
            "auc": 0.8389,
        },
        "TPATimesNet": {
            "precision": 0.7524,
            "recall": 0.5710,
            "f1": 0.6493,
            "accuracy": 0.9090,
            "auc": 0.8356,
        },
        "DLinear": {
            "precision": 0.7201,
            "recall": 0.5123,
            "f1": 0.5989,
            "accuracy": 0.8956,
            "auc": 0.7923,
        },
    }

    print("测试 create_radar_chart...")
    fig1 = create_radar_chart(test_metrics)
    print("  雷达图创建成功")

    print("\n测试 create_bar_chart...")
    fig2 = create_bar_chart(test_metrics, metric_name="f1")
    print("  柱状图创建成功")

    print("\n测试 create_metrics_table...")
    df = create_metrics_table(test_metrics)
    print(df.to_string())

    print("\n测试 create_model_comparison_plot...")
    fig3 = create_model_comparison_plot(test_metrics)
    print("  综合对比图创建成功")

    print("\n测试 create_grouped_bar_chart...")
    fig4 = create_grouped_bar_chart(test_metrics, metrics=["precision", "recall", "f1"])
    print("  分组柱状图创建成功")

    print("\n所有测试通过!")
