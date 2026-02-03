"""
创新对比标签页

对比 TimesNet 和 VoltageTimesNet 的差异:
1. 预设周期机制: VoltageTimesNet 使用电力系统先验知识
2. 混合机制: 预设周期 + FFT 发现周期的加权融合
3. 性能对比: F1、召回率等指标提升

Author: Rural Voltage Detection Project
Date: 2026
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# 配置常量
# ============================================================================

# Plotly 中文字体配置
FONT_FAMILY = "Noto Serif CJK JP, SimSun, Microsoft YaHei, sans-serif"
FONT_SIZE = 12

# 配色方案 (柔和科研风格)
THESIS_COLORS = {
    "primary": "#4878A8",      # 柔和蓝 (VoltageTimesNet)
    "secondary": "#C4785C",    # 柔和橙 (TimesNet)
    "accent": "#72A86D",       # 柔和绿
    "warning": "#D4A84C",      # 柔和黄
    "neutral": "#808080",      # 中性灰
    "light_gray": "#B0B0B0",   # 浅灰
    "preset": "#9B59B6",       # 紫色 (预设周期)
    "fft": "#E74C3C",          # 红色 (FFT周期)
}

# 性能数据 (基于实际实验结果)
PERFORMANCE_DATA = {
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
}

# 预设周期配置 (基于电力系统先验知识)
PRESET_PERIODS_INFO = {
    "1min": {"value": 60, "desc": "短期波动周期"},
    "5min": {"value": 300, "desc": "暂态事件周期"},
    "15min": {"value": 900, "desc": "负荷变化周期"},
    "1h": {"value": 3600, "desc": "小时负荷模式"},
    "日周期": {"value": 288, "desc": "日负荷周期 (5分钟采样)"},
}


# ============================================================================
# 辅助函数
# ============================================================================

def _get_font(size: int = None) -> dict:
    """获取字体配置字典"""
    return {"family": FONT_FAMILY, "size": size or FONT_SIZE}


def _simulate_fft_periods(seq_len: int = 100, seed: int = 42) -> list:
    """
    模拟 FFT 发现的周期
    用于演示目的，实际周期取决于输入数据
    """
    np.random.seed(seed)
    # 模拟 FFT 发现的主要周期
    base_periods = [seq_len // 5, seq_len // 10, seq_len // 3, seq_len // 7, seq_len // 15]
    return [max(2, p) for p in base_periods]


def _simulate_preset_periods(seq_len: int = 100) -> list:
    """
    获取适用于当前序列长度的预设周期
    """
    preset_candidates = [5, 10, 25, 50]  # 相对于序列长度的预设
    return [p for p in preset_candidates if 2 <= p <= seq_len // 2]


# ============================================================================
# 可视化函数
# ============================================================================

def create_preset_period_comparison(seq_len: int = 100) -> go.Figure:
    """
    创建预设周期对比图

    对比 TimesNet (纯 FFT) 和 VoltageTimesNet (FFT + 预设周期)
    """
    # 模拟 FFT 发现的周期
    fft_periods = _simulate_fft_periods(seq_len)
    # 预设周期
    preset_periods = _simulate_preset_periods(seq_len)

    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "TimesNet: 纯 FFT 周期发现",
            "VoltageTimesNet: FFT + 预设周期"
        ),
        horizontal_spacing=0.15
    )

    # 左图: TimesNet 纯 FFT
    fig.add_trace(
        go.Bar(
            x=[f"T={p}" for p in fft_periods],
            y=np.random.uniform(0.5, 1.0, len(fft_periods)),  # 模拟权重
            name="FFT 发现的周期",
            marker_color=THESIS_COLORS["secondary"],
            text=[f"T={p}" for p in fft_periods],
            textposition="outside",
            hovertemplate="周期: %{x}<br>权重: %{y:.3f}<extra></extra>"
        ),
        row=1, col=1
    )

    # 右图: VoltageTimesNet 混合周期
    # FFT 周期 (部分)
    n_fft = max(1, len(fft_periods) - len(preset_periods))
    combined_periods = fft_periods[:n_fft] + preset_periods
    combined_weights = list(np.random.uniform(0.5, 1.0, n_fft)) + list(np.random.uniform(0.6, 1.0, len(preset_periods)))
    combined_colors = [THESIS_COLORS["fft"]] * n_fft + [THESIS_COLORS["preset"]] * len(preset_periods)
    combined_labels = [f"FFT: T={p}" for p in fft_periods[:n_fft]] + [f"预设: T={p}" for p in preset_periods]

    fig.add_trace(
        go.Bar(
            x=combined_labels,
            y=combined_weights,
            name="混合周期",
            marker_color=combined_colors,
            text=combined_labels,
            textposition="outside",
            hovertemplate="类型: %{x}<br>权重: %{y:.3f}<extra></extra>"
        ),
        row=1, col=2
    )

    # 添加图例说明
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name="FFT 周期",
            marker=dict(size=15, color=THESIS_COLORS["fft"], symbol="square"),
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name="预设周期",
            marker=dict(size=15, color=THESIS_COLORS["preset"], symbol="square"),
            showlegend=True
        )
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="周期发现机制对比: TimesNet vs VoltageTimesNet",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        ),
        height=450,
        template="plotly_white",
        barmode="group"
    )

    # 更新坐标轴
    fig.update_yaxes(title_text="权重", range=[0, 1.2], row=1, col=1)
    fig.update_yaxes(title_text="权重", range=[0, 1.2], row=1, col=2)
    fig.update_xaxes(title_text="周期", row=1, col=1)
    fig.update_xaxes(title_text="周期", row=1, col=2)

    return fig


def create_hybrid_mechanism_diagram() -> go.Figure:
    """
    创建混合机制流程图

    展示 VoltageTimesNet 如何融合 FFT 周期和预设周期
    """
    fig = go.Figure()

    # 流程图坐标
    # 输入 -> FFT分支 -> 融合 -> 输出
    # 输入 -> 预设分支 -> 融合 -> 输出

    # 节点位置
    nodes = {
        "input": (0.1, 0.5),
        "fft": (0.35, 0.75),
        "preset": (0.35, 0.25),
        "weight": (0.55, 0.5),
        "fusion": (0.75, 0.5),
        "output": (0.95, 0.5),
    }

    # 绘制节点
    node_configs = [
        ("input", "输入序列\nx(t)", THESIS_COLORS["neutral"]),
        ("fft", "FFT 周期发现\n(数据驱动)", THESIS_COLORS["fft"]),
        ("preset", "预设周期\n(领域知识)", THESIS_COLORS["preset"]),
        ("weight", "权重计算\nalpha=0.7", THESIS_COLORS["warning"]),
        ("fusion", "加权融合\nSoftmax", THESIS_COLORS["primary"]),
        ("output", "输出周期\n[T1,T2,...Tk]", THESIS_COLORS["accent"]),
    ]

    for name, label, color in node_configs:
        x, y = nodes[name]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=50, color=color, symbol="square", opacity=0.8),
                text=[label],
                textposition="middle center",
                textfont=dict(size=10, color="white" if name not in ["input", "output"] else "black"),
                hoverinfo="text",
                hovertext=label.replace("\n", " "),
                showlegend=False
            )
        )

    # 绘制箭头连接
    arrows = [
        ("input", "fft", ""),
        ("input", "preset", ""),
        ("fft", "weight", "FFT 权重"),
        ("preset", "weight", "预设权重"),
        ("weight", "fusion", ""),
        ("fusion", "output", ""),
    ]

    for start, end, label in arrows:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]

        # 使用 shape 绘制箭头线条（兼容新版 Plotly）
        fig.add_shape(
            type="line",
            x0=x0 + 0.05 * (1 if x1 > x0 else -1),
            y0=y0 + 0.05 * (1 if y1 > y0 else -1) if y1 != y0 else y0,
            x1=x1 - 0.05 * (1 if x1 > x0 else -1),
            y1=y1 - 0.05 * (1 if y1 > y0 else -1) if y1 != y0 else y1,
            xref="paper",
            yref="paper",
            line=dict(color=THESIS_COLORS["neutral"], width=2),
        )

        # 在终点添加箭头标记
        fig.add_annotation(
            x=x1 - 0.05 * (1 if x1 > x0 else -1),
            y=y1 - 0.05 * (1 if y1 > y0 else -1) if y1 != y0 else y1,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="▶" if x1 > x0 else "◀",
            font=dict(size=10, color=THESIS_COLORS["neutral"]),
        )

        if label:
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y + 0.05,
                text=label,
                showarrow=False,
                font=_get_font(9),
                bgcolor="white",
                borderpad=2
            )

    # 添加公式说明
    formula_text = (
        "<b>混合机制公式:</b><br><br>"
        "periods = alpha * FFT_periods + (1-alpha) * preset_periods<br><br>"
        "其中 alpha = 0.7 (可配置)<br>"
        "FFT_periods: 数据驱动发现的周期<br>"
        "preset_periods: 基于电力系统先验的周期"
    )

    fig.add_annotation(
        text=formula_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=_get_font(11),
        align="left",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=THESIS_COLORS["primary"],
        borderwidth=2,
        borderpad=10
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="VoltageTimesNet 混合周期机制流程图",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=500,
        template="plotly_white",
        margin=dict(l=20, r=20, t=80, b=120)
    )

    return fig


def create_performance_comparison() -> go.Figure:
    """
    创建性能对比图

    展示 TimesNet 和 VoltageTimesNet 系列模型的性能差异
    """
    models = ["TimesNet", "VoltageTimesNet", "VoltageTimesNet_v2"]
    metrics = ["precision", "recall", "f1", "accuracy", "auc"]
    metric_names_cn = ["精确率", "召回率", "F1分数", "准确率", "AUC"]

    # 创建子图: 雷达图 + 柱状图
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        subplot_titles=("多维性能雷达图", "F1 分数对比"),
        horizontal_spacing=0.15
    )

    # 颜色映射
    colors = {
        "TimesNet": THESIS_COLORS["secondary"],
        "VoltageTimesNet": THESIS_COLORS["accent"],
        "VoltageTimesNet_v2": THESIS_COLORS["primary"],
    }

    # 雷达图
    for model in models:
        values = [PERFORMANCE_DATA[model][m] for m in metrics]
        values_closed = values + [values[0]]  # 闭合多边形
        categories_closed = metric_names_cn + [metric_names_cn[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor=colors[model],
                opacity=0.25,
                line=dict(color=colors[model], width=2.5),
                name=model,
                hovertemplate=f"<b>{model}</b><br>%{{theta}}: %{{r:.4f}}<extra></extra>"
            ),
            row=1, col=1
        )

    # 柱状图: F1 分数对比
    f1_values = [PERFORMANCE_DATA[m]["f1"] for m in models]
    bar_colors = [colors[m] for m in models]

    fig.add_trace(
        go.Bar(
            x=models,
            y=f1_values,
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in f1_values],
            textposition="outside",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>F1: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )

    # 添加性能提升标注
    improvement = (PERFORMANCE_DATA["VoltageTimesNet_v2"]["f1"] - PERFORMANCE_DATA["TimesNet"]["f1"]) / PERFORMANCE_DATA["TimesNet"]["f1"] * 100

    fig.add_annotation(
        x="VoltageTimesNet_v2",
        y=PERFORMANCE_DATA["VoltageTimesNet_v2"]["f1"] + 0.02,
        text=f"+{improvement:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=THESIS_COLORS["accent"],
        font=dict(size=12, color=THESIS_COLORS["accent"], family=FONT_FAMILY),
        row=1, col=2
    )

    # 更新极坐标配置
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            tickfont={"size": 9},
            gridcolor="#e0e0e0",
        ),
        angularaxis=dict(
            tickfont={"size": 10, "family": FONT_FAMILY},
            gridcolor="#e0e0e0",
        ),
        bgcolor="white",
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="模型性能对比: TimesNet vs VoltageTimesNet",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.25
        ),
        height=500,
        template="plotly_white"
    )

    # 更新柱状图坐标轴
    fig.update_yaxes(title_text="F1 分数", range=[0.6, 0.7], row=1, col=2)
    fig.update_xaxes(title_text="模型", tickangle=15, row=1, col=2)

    return fig


def create_detailed_metrics_table() -> str:
    """
    创建详细指标表格 (Markdown 格式)
    """
    header = "| 模型 | 精确率 | 召回率 | F1 分数 | 准确率 | AUC |"
    separator = "|:-----|:------:|:------:|:-------:|:------:|:---:|"

    rows = []
    for model in ["TimesNet", "VoltageTimesNet", "VoltageTimesNet_v2"]:
        data = PERFORMANCE_DATA[model]
        row = f"| {model} | {data['precision']:.4f} | {data['recall']:.4f} | {data['f1']:.4f} | {data['accuracy']:.4f} | {data['auc']:.4f} |"
        rows.append(row)

    return "\n".join([header, separator] + rows)


# ============================================================================
# 说明文本内容
# ============================================================================

INNOVATION_DESCRIPTIONS = {
    "preset_period": """
## 创新点 1: 预设周期机制

### 问题背景
原始 TimesNet 使用纯 FFT 方法发现时间序列中的周期，这种数据驱动的方法虽然通用，但在特定领域可能错过重要的先验知识。

### VoltageTimesNet 的改进
我们引入**电力系统领域知识**作为预设周期:

| 预设周期 | 采样点数 | 物理含义 |
|:---------|:--------:|:---------|
| 日周期 | 288 | 日负荷变化模式 (5分钟采样) |
| 1分钟 | 60 | 短期电压波动 |
| 5分钟 | 300 | 暂态事件响应 |
| 15分钟 | 900 | 负荷变化周期 |

### 核心代码

```python
# 电力系统预设周期 (基于采样率)
VOLTAGE_PRESET_PERIODS = {
    "1min": 60,   # 1分钟短期波动
    "5min": 300,  # 5分钟暂态事件
    "15min": 900, # 15分钟负荷变化
    "1h": 3600,   # 1小时负荷模式
}
```

### 优势
1. **领域适配**: 自动捕获电力系统特有的周期性模式
2. **稳定性**: 即使数据噪声大，预设周期仍能保持检测能力
3. **可解释性**: 检测结果与电力工程知识直接关联
""",

    "hybrid_mechanism": """
## 创新点 2: 预设周期与 FFT 的混合融合机制

### 设计思想
结合**数据驱动** (FFT) 和**知识驱动** (预设周期) 两种范式的优势。

### 融合公式

```
final_periods = alpha * FFT_periods + (1-alpha) * preset_periods
```

其中:
- `alpha = 0.7` (默认值，可配置)
- `FFT_periods`: 通过 FFT 频谱分析发现的数据驱动周期
- `preset_periods`: 基于电力系统先验知识的预设周期

### 实现细节

```python
def FFT_for_Period_Voltage(x, k=2, preset_periods=None, preset_weight=0.3):
    '''
    混合周期发现函数

    Args:
        x: 输入张量 [B, T, C]
        k: top-k 周期数
        preset_periods: 预设周期列表
        preset_weight: 预设周期权重 (alpha = 1 - preset_weight)
    '''
    # 1. FFT 发现数据驱动周期
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)

    # 2. 按权重分配周期名额
    n_preset = max(1, int(k * preset_weight))
    n_fft = k - n_preset

    # 3. 融合两类周期
    final_periods = fft_periods[:n_fft] + valid_presets[:n_preset]

    return final_periods, period_weights
```

### 自适应权重
周期权重通过 Softmax 计算，基于 FFT 振幅:

```python
period_weight = F.softmax(period_weight, dim=1)
res = torch.sum(res * period_weight, -1)
```
""",

    "performance": """
## 创新点 3: 性能提升分析

### 实验结果汇总

基于 RuralVoltage 农村电压数据集的对比实验:

| 模型 | F1 分数 | 召回率 | 精确率 | AUC |
|:-----|:-------:|:------:|:------:|:---:|
| TimesNet (基线) | 0.6520 | 0.5705 | 0.7606 | 0.8389 |
| VoltageTimesNet | 0.6509 | 0.5726 | 0.7541 | 0.8412 |
| **VoltageTimesNet_v2** | **0.6622** | **0.5858** | 0.7614 | **0.8523** |

### 关键改进

1. **F1 分数提升**: 从 0.6520 提升到 0.6622 (+1.6%)
2. **召回率提升**: 从 0.5705 提升到 0.5858 (+2.7%)
3. **AUC 提升**: 从 0.8389 提升到 0.8523 (+1.6%)

### 为什么召回率更重要?

在电力系统异常检测场景中:
- **漏检 (假阴性)** 可能导致设备损坏、电网故障
- **误报 (假阳性)** 只是增加人工复核工作量

因此，**召回率的提升比精确率更有价值**。

### 消融实验

| alpha 值 | F1 分数 | 说明 |
|:--------:|:-------:|:-----|
| 1.0 | 0.6520 | 纯 FFT (TimesNet) |
| 0.7 | **0.6622** | 最优配置 |
| 0.5 | 0.6580 | 平衡配置 |
| 0.3 | 0.6510 | 偏重预设周期 |

结论: `alpha = 0.7` (70% FFT + 30% 预设) 是最优配置。
"""
}


# ============================================================================
# 主标签页函数
# ============================================================================

def create_innovation_tab():
    """
    创建创新对比标签页

    包含:
    1. Dropdown 选择创新点
    2. 双列布局: 左侧 TimesNet，右侧 VoltageTimesNet
    3. Plotly 对比可视化
    4. Markdown 技术说明
    """
    with gr.Tab("创新对比"):
        gr.Markdown("""
        # VoltageTimesNet 创新点解析

        本页面展示 VoltageTimesNet 相对于原始 TimesNet 的三大创新点:
        1. **预设周期机制**: 引入电力系统领域先验知识
        2. **混合融合机制**: 数据驱动 + 知识驱动的加权融合
        3. **性能提升**: F1、召回率等关键指标的改进
        """)

        # Gradio 6.x: choices 必须使用纯字符串，不支持 tuple 格式
        # 创建显示名称到内部值的映射
        innovation_choices = ["预设周期机制", "混合融合机制", "性能对比分析"]
        innovation_mapping = {
            "预设周期机制": "preset_period",
            "混合融合机制": "hybrid_mechanism",
            "性能对比分析": "performance",
        }

        with gr.Row():
            innovation_selector = gr.Dropdown(
                choices=innovation_choices,
                value="预设周期机制",
                label="选择创新点",
                info="选择要查看的创新点详情"
            )

        # 双列布局
        with gr.Row():
            # 左列: 可视化图表
            with gr.Column(scale=1):
                gr.Markdown("### 对比可视化")
                # Gradio 6.x: 使用 value 参数设置初始图表
                comparison_plot = gr.Plot(
                    value=create_preset_period_comparison(),
                    label="对比图表",
                    show_label=False
                )

            # 右列: 技术说明
            with gr.Column(scale=1):
                gr.Markdown("### 技术详解")
                description_md = gr.Markdown(
                    value=INNOVATION_DESCRIPTIONS["preset_period"],
                    label="说明"
                )

        # 指标表格区域 (仅在性能对比时显示)
        with gr.Row():
            metrics_table = gr.Markdown(
                value="",
                label="详细指标",
                visible=True
            )

        # 交互逻辑
        def update_innovation_view(innovation_label: str):
            """
            根据选择的创新点更新可视化和说明

            Args:
                innovation_label: 创新点显示名称 (Gradio 6.x 传递的是显示文本)

            Returns:
                plot: 更新后的图表
                description: 更新后的说明文本
                metrics: 更新后的指标表格
            """
            # Gradio 6.x: 将显示名称映射为内部值
            innovation_type = innovation_mapping.get(innovation_label, "preset_period")

            # 获取说明文本
            description = INNOVATION_DESCRIPTIONS.get(innovation_type, "")

            # 根据类型生成图表
            if innovation_type == "preset_period":
                fig = create_preset_period_comparison()
                metrics = ""
            elif innovation_type == "hybrid_mechanism":
                fig = create_hybrid_mechanism_diagram()
                metrics = ""
            elif innovation_type == "performance":
                fig = create_performance_comparison()
                metrics = "\n### 详细性能指标\n\n" + create_detailed_metrics_table()
            else:
                fig = create_preset_period_comparison()
                metrics = ""

            return fig, description, metrics

        # 绑定事件
        innovation_selector.change(
            fn=update_innovation_view,
            inputs=[innovation_selector],
            outputs=[comparison_plot, description_md, metrics_table]
        )

        # Gradio 6.x: 初始化已通过组件的 value 参数完成，无需额外设置

    return {
        "innovation_selector": innovation_selector,
        "comparison_plot": comparison_plot,
        "description_md": description_md,
        "metrics_table": metrics_table,
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试可视化函数
    print("测试 create_preset_period_comparison...")
    fig1 = create_preset_period_comparison()
    fig1.write_html("/tmp/test_preset_period.html")
    print("  -> 保存到 /tmp/test_preset_period.html")

    print("测试 create_hybrid_mechanism_diagram...")
    fig2 = create_hybrid_mechanism_diagram()
    fig2.write_html("/tmp/test_hybrid_mechanism.html")
    print("  -> 保存到 /tmp/test_hybrid_mechanism.html")

    print("测试 create_performance_comparison...")
    fig3 = create_performance_comparison()
    fig3.write_html("/tmp/test_performance.html")
    print("  -> 保存到 /tmp/test_performance.html")

    print("\n所有测试完成!")
