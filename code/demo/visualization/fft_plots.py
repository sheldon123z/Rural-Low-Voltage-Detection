"""
FFT 可视化模块
用于展示 TimesNet 中的 FFT 周期发现过程

支持功能:
1. FFT 频谱可视化
2. 1D → 2D 重塑过程可视化
3. 综合周期分析图
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 配色方案 (柔和科研风格)
THESIS_COLORS = {
    "primary": "#4878A8",      # 柔和蓝
    "secondary": "#72A86D",    # 柔和绿
    "accent": "#C4785C",       # 柔和橙
    "warning": "#D4A84C",      # 柔和黄
    "neutral": "#808080",      # 中性灰
    "light_gray": "#B0B0B0",   # 浅灰
}

# Plotly 中文字体配置
FONT_FAMILY = "Noto Serif CJK JP, SimSun, Microsoft YaHei, serif"
FONT_SIZE = 12


def _get_font(size: int = None) -> dict:
    """获取字体配置字典"""
    return {"family": FONT_FAMILY, "size": size or FONT_SIZE}


def fft_for_period(signal: np.ndarray, k: int = 5):
    """
    使用 FFT 发现时间序列中的主要周期

    Args:
        signal: 1D 时间序列信号 [T]
        k: 返回的 top-k 周期数

    Returns:
        periods: 周期列表
        amplitudes: 对应的振幅
        frequency_spectrum: 完整频谱
        frequencies: 频率列表
    """
    n = len(signal)
    # FFT 变换
    fft_result = np.fft.rfft(signal)
    amplitudes_full = np.abs(fft_result)
    frequencies = np.fft.rfftfreq(n)

    # 排除直流分量 (频率=0)
    amplitudes_for_sort = amplitudes_full.copy()
    amplitudes_for_sort[0] = 0

    # 找到 top-k 频率
    top_indices = np.argsort(amplitudes_for_sort)[-k:][::-1]

    # 计算周期 (避免除零)
    periods = []
    top_amplitudes = []
    for idx in top_indices:
        if frequencies[idx] > 0:
            period = int(round(1.0 / frequencies[idx]))
            periods.append(period)
            top_amplitudes.append(amplitudes_full[idx])
        else:
            periods.append(n)  # 整个序列长度
            top_amplitudes.append(amplitudes_full[idx])

    return periods, top_amplitudes, amplitudes_full, frequencies


def create_fft_visualization(signal: np.ndarray, top_k: int = 5) -> go.Figure:
    """
    创建 FFT 可视化交互式图表

    Args:
        signal: 1D 时间序列信号
        top_k: 标记的主要周期数量

    Returns:
        Plotly Figure 对象
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # 执行 FFT 分析
    periods, amplitudes, spectrum, frequencies = fft_for_period(signal, top_k)

    # 创建子图: 上方原始信号, 下方频谱
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("原始时间序列信号", "FFT 频谱分析"),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )

    # 上图: 原始信号
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=signal,
            mode="lines",
            name="原始信号",
            line=dict(color=THESIS_COLORS["primary"], width=1.5),
            hovertemplate="时间步: %{x}<br>数值: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )

    # 下图: 频谱
    fig.add_trace(
        go.Bar(
            x=frequencies[1:],  # 排除直流分量
            y=spectrum[1:],
            name="频谱振幅",
            marker_color=THESIS_COLORS["light_gray"],
            hovertemplate="频率: %{x:.4f}<br>振幅: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    # 标记 top-k 周期对应的频率
    top_indices = np.argsort(spectrum.copy())[-top_k-1:][::-1]
    top_indices = [idx for idx in top_indices if idx > 0][:top_k]

    colors_cycle = [
        THESIS_COLORS["accent"],
        THESIS_COLORS["secondary"],
        THESIS_COLORS["warning"],
        THESIS_COLORS["primary"],
        "#9B59B6",  # 紫色
    ]

    for i, idx in enumerate(top_indices):
        freq = frequencies[idx]
        amp = spectrum[idx]
        period = int(round(1.0 / freq)) if freq > 0 else n
        color = colors_cycle[i % len(colors_cycle)]

        # 标记频谱峰值
        fig.add_trace(
            go.Scatter(
                x=[freq],
                y=[amp],
                mode="markers+text",
                name=f"周期 {period}",
                marker=dict(size=12, color=color, symbol="star"),
                text=[f"T={period}"],
                textposition="top center",
                textfont=dict(size=10, color=color),
                hovertemplate=f"周期: {period}<br>频率: {freq:.4f}<br>振幅: {amp:.2f}<extra></extra>"
            ),
            row=2, col=1
        )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="TimesNet FFT 周期发现可视化",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        template="plotly_white",
        hovermode="x unified"
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="时间步", row=1, col=1)
    fig.update_yaxes(title_text="数值", row=1, col=1)
    fig.update_xaxes(title_text="频率", row=2, col=1)
    fig.update_yaxes(title_text="振幅", row=2, col=1)

    # 添加周期说明注释
    period_text = "发现的主要周期: " + ", ".join([f"T={p}" for p in periods[:top_k]])
    fig.add_annotation(
        text=period_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=_get_font(11),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=THESIS_COLORS["neutral"],
        borderwidth=1,
        borderpad=4
    )

    return fig


def create_2d_reshape_visualization(signal: np.ndarray, period: int) -> go.Figure:
    """
    展示 1D → 2D 重塑过程

    Args:
        signal: 1D 时间序列信号
        period: 重塑使用的周期

    Returns:
        Plotly Figure 对象
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # 计算需要的填充
    if n % period != 0:
        padded_length = ((n // period) + 1) * period
        signal_padded = np.pad(signal, (0, padded_length - n), mode='constant', constant_values=0)
    else:
        padded_length = n
        signal_padded = signal

    # 重塑为 2D
    num_periods = padded_length // period
    signal_2d = signal_padded.reshape(num_periods, period)

    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "原始 1D 信号",
            "2D 重塑矩阵 (热力图)",
            "按周期叠加显示",
            "重塑过程示意"
        ),
        specs=[
            [{"colspan": 2}, None],
            [{}, {}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        row_heights=[0.35, 0.65]
    )

    # 1. 原始 1D 信号
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=signal,
            mode="lines",
            name="原始信号",
            line=dict(color=THESIS_COLORS["primary"], width=1.5),
            hovertemplate="时间步: %{x}<br>数值: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )

    # 标记周期分割线
    for i in range(1, num_periods + 1):
        pos = i * period
        if pos <= n:
            fig.add_vline(
                x=pos, line_dash="dash",
                line_color=THESIS_COLORS["accent"],
                opacity=0.5,
                row=1, col=1
            )

    # 2. 2D 热力图
    fig.add_trace(
        go.Heatmap(
            z=signal_2d,
            x=[f"{i}" for i in range(period)],
            y=[f"周期{i+1}" for i in range(num_periods)],
            colorscale=[
                [0, "#FFFFFF"],
                [0.5, THESIS_COLORS["primary"]],
                [1, "#1a3a5c"]
            ],
            colorbar=dict(
                title=dict(text="数值", font=_get_font()),
                tickfont=_get_font(),
                len=0.4,
                y=0.25
            ),
            hovertemplate="周期内位置: %{x}<br>周期: %{y}<br>数值: %{z:.4f}<extra></extra>"
        ),
        row=2, col=1
    )

    # 3. 按周期叠加显示
    colors = [
        THESIS_COLORS["primary"],
        THESIS_COLORS["secondary"],
        THESIS_COLORS["accent"],
        THESIS_COLORS["warning"],
        "#9B59B6",
        "#1ABC9C",
        "#E74C3C",
    ]

    for i in range(min(num_periods, 7)):  # 最多显示7个周期
        fig.add_trace(
            go.Scatter(
                x=np.arange(period),
                y=signal_2d[i, :],
                mode="lines",
                name=f"周期 {i+1}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.8,
                hovertemplate=f"周期 {i+1}<br>位置: %{{x}}<br>数值: %{{y:.4f}}<extra></extra>"
            ),
            row=2, col=2
        )

    # 4. 重塑过程示意 (用文字和箭头说明)
    fig.add_annotation(
        text=f"<b>1D → 2D 重塑过程</b><br><br>"
             f"原始长度: {n}<br>"
             f"周期 T: {period}<br>"
             f"填充后长度: {padded_length}<br>"
             f"2D 矩阵形状: {num_periods} × {period}<br><br>"
             f"<i>reshape(T, N/T)</i>",
        xref="paper", yref="paper",
        x=0.95, y=0.1,
        showarrow=False,
        font=_get_font(11),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=THESIS_COLORS["primary"],
        borderwidth=2,
        borderpad=8
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=f"TimesNet 1D → 2D 重塑可视化 (周期 T={period})",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700,
        template="plotly_white"
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="时间步", row=1, col=1)
    fig.update_yaxes(title_text="数值", row=1, col=1)
    fig.update_xaxes(title_text="周期内位置", row=2, col=1)
    fig.update_yaxes(title_text="周期编号", row=2, col=1)
    fig.update_xaxes(title_text="周期内位置", row=2, col=2)
    fig.update_yaxes(title_text="数值", row=2, col=2)

    return fig


def create_period_analysis_plot(signal: np.ndarray, top_k: int = 5) -> go.Figure:
    """
    创建综合周期分析图

    包含:
    - 原始时序图
    - FFT 频谱图
    - 周期标注
    - 各周期重塑后的对比

    Args:
        signal: 1D 时间序列信号
        top_k: 分析的主要周期数量

    Returns:
        Plotly Figure 对象
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # FFT 分析
    periods, amplitudes, spectrum, frequencies = fft_for_period(signal, top_k)

    # 创建子图
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "原始时间序列",
            "FFT 频谱分析",
            "周期强度对比",
            "主要周期重塑热力图",
            "多周期叠加视图",
            "周期特征统计"
        ),
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        row_heights=[0.3, 0.35, 0.35]
    )

    # 1. 原始时间序列
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=signal,
            mode="lines",
            name="原始信号",
            line=dict(color=THESIS_COLORS["primary"], width=1.5),
            hovertemplate="时间步: %{x}<br>数值: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )

    # 2. FFT 频谱
    fig.add_trace(
        go.Bar(
            x=frequencies[1:50] if len(frequencies) > 50 else frequencies[1:],
            y=spectrum[1:50] if len(spectrum) > 50 else spectrum[1:],
            name="频谱",
            marker_color=THESIS_COLORS["light_gray"],
            showlegend=False,
            hovertemplate="频率: %{x:.4f}<br>振幅: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    # 标记主要周期
    colors_cycle = [
        THESIS_COLORS["accent"],
        THESIS_COLORS["secondary"],
        THESIS_COLORS["warning"],
        THESIS_COLORS["primary"],
        "#9B59B6",
    ]

    for i, (period, amp) in enumerate(zip(periods[:top_k], amplitudes[:top_k])):
        if period > 0 and period < n:
            freq = 1.0 / period
            color = colors_cycle[i % len(colors_cycle)]
            fig.add_trace(
                go.Scatter(
                    x=[freq],
                    y=[amp],
                    mode="markers",
                    name=f"T={period}",
                    marker=dict(size=10, color=color, symbol="star"),
                    hovertemplate=f"周期: {period}<br>振幅: {amp:.2f}<extra></extra>"
                ),
                row=2, col=1
            )

    # 3. 周期强度对比 (柱状图)
    valid_periods = [p for p in periods[:top_k] if p > 0]
    valid_amplitudes = amplitudes[:len(valid_periods)]

    fig.add_trace(
        go.Bar(
            x=[f"T={p}" for p in valid_periods],
            y=valid_amplitudes,
            name="周期强度",
            marker_color=[colors_cycle[i % len(colors_cycle)] for i in range(len(valid_periods))],
            showlegend=False,
            hovertemplate="周期: %{x}<br>强度: %{y:.2f}<extra></extra>"
        ),
        row=2, col=2
    )

    # 4. 主要周期重塑热力图 (使用最强周期)
    main_period = valid_periods[0] if valid_periods else 10
    if n % main_period != 0:
        padded_length = ((n // main_period) + 1) * main_period
        signal_padded = np.pad(signal, (0, padded_length - n), mode='constant')
    else:
        padded_length = n
        signal_padded = signal

    num_periods = padded_length // main_period
    signal_2d = signal_padded.reshape(num_periods, main_period)

    fig.add_trace(
        go.Heatmap(
            z=signal_2d[:min(20, num_periods), :],  # 最多显示20行
            colorscale=[
                [0, "#FFFFFF"],
                [0.5, THESIS_COLORS["primary"]],
                [1, "#1a3a5c"]
            ],
            showscale=False,
            hovertemplate="行: %{y}<br>列: %{x}<br>值: %{z:.4f}<extra></extra>"
        ),
        row=3, col=1
    )

    # 5. 多周期叠加视图
    for i, period in enumerate(valid_periods[:3]):  # 显示前3个周期
        if n % period != 0:
            pad_len = ((n // period) + 1) * period
            sig_pad = np.pad(signal, (0, pad_len - n), mode='constant')
        else:
            sig_pad = signal

        num_p = len(sig_pad) // period
        sig_2d = sig_pad.reshape(num_p, period)
        mean_pattern = sig_2d.mean(axis=0)

        color = colors_cycle[i % len(colors_cycle)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(period),
                y=mean_pattern,
                mode="lines",
                name=f"平均模式 T={period}",
                line=dict(color=color, width=2),
                hovertemplate=f"周期 T={period}<br>位置: %{{x}}<br>均值: %{{y:.4f}}<extra></extra>"
            ),
            row=3, col=2
        )

    # 6. 周期特征统计 (添加注释)
    stats_text = (
        f"<b>周期分析统计</b><br><br>"
        f"信号长度: {n}<br>"
        f"发现周期数: {len(valid_periods)}<br><br>"
        f"<b>Top-{min(top_k, len(valid_periods))} 周期:</b><br>"
    )
    for i, (p, a) in enumerate(zip(valid_periods[:top_k], valid_amplitudes[:top_k])):
        stats_text += f"  T={p}: 强度={a:.2f}<br>"

    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=_get_font(10),
        align="left",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=THESIS_COLORS["primary"],
        borderwidth=1,
        borderpad=6
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="TimesNet 综合周期分析",
            font=_get_font(16),
            x=0.5
        ),
        font=_get_font(),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=900,
        template="plotly_white"
    )

    # 更新坐标轴标签
    fig.update_xaxes(title_text="时间步", row=1, col=1)
    fig.update_yaxes(title_text="数值", row=1, col=1)
    fig.update_xaxes(title_text="频率", row=2, col=1)
    fig.update_yaxes(title_text="振幅", row=2, col=1)
    fig.update_xaxes(title_text="周期", row=2, col=2)
    fig.update_yaxes(title_text="强度", row=2, col=2)
    fig.update_xaxes(title_text=f"周期内位置 (T={main_period})", row=3, col=1)
    fig.update_yaxes(title_text="周期编号", row=3, col=1)
    fig.update_xaxes(title_text="周期内位置", row=3, col=2)
    fig.update_yaxes(title_text="平均数值", row=3, col=2)

    return fig


# 便捷测试函数
def _test_visualizations():
    """测试可视化函数"""
    # 生成测试信号: 多周期叠加 + 噪声
    t = np.arange(200)
    signal = (
        np.sin(2 * np.pi * t / 20) * 2 +      # 周期 20
        np.sin(2 * np.pi * t / 50) * 1.5 +    # 周期 50
        np.sin(2 * np.pi * t / 10) * 0.8 +    # 周期 10
        np.random.randn(200) * 0.3             # 噪声
    )

    print("测试 create_fft_visualization...")
    fig1 = create_fft_visualization(signal, top_k=5)
    fig1.write_html("/tmp/test_fft.html")
    print("  -> 保存到 /tmp/test_fft.html")

    print("测试 create_2d_reshape_visualization...")
    fig2 = create_2d_reshape_visualization(signal, period=20)
    fig2.write_html("/tmp/test_2d_reshape.html")
    print("  -> 保存到 /tmp/test_2d_reshape.html")

    print("测试 create_period_analysis_plot...")
    fig3 = create_period_analysis_plot(signal, top_k=5)
    fig3.write_html("/tmp/test_period_analysis.html")
    print("  -> 保存到 /tmp/test_period_analysis.html")

    print("所有测试完成!")


if __name__ == "__main__":
    _test_visualizations()
