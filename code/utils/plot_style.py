#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科研绘图样式配置模块

提供 MATLAB 风格的科研绘图配置，适用于学术论文发表。

特点:
1. 白色背景 + 黑色坐标轴 (经典科研风格)
2. 较粗的线条和标记 (清晰可见)
3. 专业的网格线样式 (灰色虚线)
4. 高对比度配色方案 (适合打印)
5. 中文字体支持 (文泉驿微米黑)
6. 高分辨率输出 (300 DPI)

使用方法:
    from utils.plot_style import apply_matlab_style, COLORS, get_chinese_font
    apply_matlab_style()  # 在绑图前调用
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import warnings

# ============================================
# 中文字体配置
# ============================================

# 系统可用的中文字体列表（按优先级排序）
CHINESE_FONTS = [
    "WenQuanYi Micro Hei",  # 文泉驿微米黑
    "WenQuanYi Zen Hei",  # 文泉驿正黑
    "Noto Sans CJK SC",  # Google Noto 简体中文
    "Noto Sans CJK JP",  # Google Noto 日文
    "Droid Sans Fallback",  # Android 回退字体
    "AR PL UKai CN",  # 文鼎楷体
    "AR PL UMing CN",  # 文鼎明体
    "SimHei",  # Windows 黑体
    "Microsoft YaHei",  # 微软雅黑
]


def get_available_chinese_font():
    """获取系统中可用的中文字体"""
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])

    for font in CHINESE_FONTS:
        if font in available_fonts:
            return font

    # 如果没有找到，尝试模糊匹配
    for font in available_fonts:
        if "WenQuanYi" in font or "Noto" in font or "CJK" in font:
            return font

    warnings.warn("未找到中文字体，可能无法正确显示中文")
    return "DejaVu Sans"


def get_chinese_font():
    """获取中文字体名称（供外部调用）"""
    return get_available_chinese_font()


# ============================================
# MATLAB 风格配色方案
# ============================================

# 经典 MATLAB 配色 (R2014b 之前)
MATLAB_COLORS_CLASSIC = [
    "#0000FF",  # 蓝色
    "#008000",  # 绿色
    "#FF0000",  # 红色
    "#00BFBF",  # 青色
    "#BF00BF",  # 品红
    "#BFBF00",  # 黄色
    "#404040",  # 深灰
]

# 现代 MATLAB 配色 (R2014b+)
MATLAB_COLORS_MODERN = [
    "#0072BD",  # 蓝色
    "#D95319",  # 橙色
    "#EDB120",  # 黄色
    "#7E2F8E",  # 紫色
    "#77AC30",  # 绿色
    "#4DBEEE",  # 浅蓝
    "#A2142F",  # 深红
]

# 高对比度配色（适合论文打印）
HIGH_CONTRAST_COLORS = [
    "#1f77b4",  # 蓝色
    "#ff7f0e",  # 橙色
    "#2ca02c",  # 绿色
    "#d62728",  # 红色
    "#9467bd",  # 紫色
    "#8c564b",  # 棕色
    "#e377c2",  # 粉色
    "#7f7f7f",  # 灰色
]

# 默认使用现代 MATLAB 配色
COLORS = MATLAB_COLORS_MODERN


# ============================================
# 线型和标记配置
# ============================================

LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-."]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]


# ============================================
# MATLAB 风格配置
# ============================================

def get_matlab_style_params():
    """获取 MATLAB 风格的参数字典"""
    chinese_font = get_available_chinese_font()

    return {
        # 字体配置
        "font.family": "sans-serif",
        "font.sans-serif": [chinese_font, "DejaVu Sans", "Arial"],
        "font.size": 12,
        "axes.unicode_minus": False,  # 解决负号显示问题

        # 坐标轴配置 (MATLAB 风格: 粗黑边框)
        "axes.linewidth": 1.5,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.axisbelow": True,

        # 网格配置 (MATLAB 风格: 浅灰色虚线)
        "grid.color": "#CCCCCC",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,

        # 刻度配置
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        # 图例配置
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.shadow": False,

        # 线条配置
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "lines.markeredgewidth": 1.5,

        # 图形配置
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.figsize": (8, 6),
        "figure.dpi": 100,
        "figure.autolayout": True,

        # 保存配置
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # 颜色循环 (使用 MATLAB 现代配色)
        "axes.prop_cycle": plt.cycler(color=MATLAB_COLORS_MODERN),
    }


def apply_matlab_style():
    """应用 MATLAB 风格的绑图配置"""
    # 重置为默认样式
    plt.style.use("default")

    # 应用自定义参数
    params = get_matlab_style_params()
    mpl.rcParams.update(params)

    print(f"已应用 MATLAB 科研绘图风格 (字体: {get_available_chinese_font()})")


def apply_ieee_style():
    """应用 IEEE 论文风格（单栏 3.5in, 双栏 7in）"""
    apply_matlab_style()

    # IEEE 特定配置
    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (3.5, 2.625),  # 单栏宽度
    })

    print("已应用 IEEE 论文绘图风格")


def apply_thesis_style():
    """应用学位论文风格（A4 纸张优化）"""
    apply_matlab_style()

    # 论文特定配置
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.figsize": (8, 6),
    })

    print("已应用学位论文绘图风格")


# ============================================
# 绑图辅助函数
# ============================================

def set_axis_style(ax, xlabel=None, ylabel=None, title=None, legend=True):
    """设置坐标轴样式"""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()

    # MATLAB 风格: 所有边框可见
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)


def add_grid(ax, which="major", axis="both"):
    """添加网格线"""
    ax.grid(True, which=which, axis=axis, linestyle="--", alpha=0.7)


def save_figure(fig, filepath, formats=("png", "pdf")):
    """保存图形为多种格式"""
    import os

    base_path = os.path.splitext(filepath)[0]

    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        print(f"已保存: {save_path}")


def create_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """创建图形和坐标轴"""
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = (8, 6)
        else:
            figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def plot_with_markers(ax, x, y, label=None, color=None, marker_interval=10, **kwargs):
    """绑制带稀疏标记的曲线（避免标记过密）"""
    # 绑制主线
    line, = ax.plot(x, y, label=label, color=color, **kwargs)

    # 添加稀疏标记
    if "marker" in kwargs:
        marker = kwargs.pop("marker", "o")
        ax.plot(x[::marker_interval], y[::marker_interval],
               linestyle="none", marker=marker, color=line.get_color(),
               markersize=kwargs.get("markersize", 8))

    return line


def create_colorbar(fig, mappable, ax, label=None):
    """创建颜色条"""
    cbar = fig.colorbar(mappable, ax=ax)
    if label:
        cbar.set_label(label)
    cbar.outline.set_linewidth(1.5)
    return cbar


# ============================================
# 颜色工具函数
# ============================================

def get_color(index):
    """获取指定索引的颜色"""
    return COLORS[index % len(COLORS)]


def get_colormap(n_colors, cmap_name="viridis"):
    """获取指定数量的颜色列表"""
    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(i / (n_colors - 1)) for i in range(n_colors)]


def lighten_color(color, amount=0.3):
    """使颜色变浅"""
    import colorsys

    try:
        c = mpl.colors.cnames[color]
    except KeyError:
        c = color

    c = mpl.colors.to_rgb(c)
    c = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(c[0], min(1, c[1] + amount), c[2])


# ============================================
# 初始化
# ============================================

if __name__ == "__main__":
    # 测试绘图风格
    apply_matlab_style()

    # 创建测试图
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(0, 2 * np.pi, 100)

    for i, label in enumerate(["正弦波", "余弦波", "正切波"]):
        if i == 0:
            y = np.sin(x)
        elif i == 1:
            y = np.cos(x)
        else:
            y = np.sin(x) * np.cos(x)

        ax.plot(x, y, label=label, marker=MARKERS[i], markevery=10)

    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅值 (V)")
    ax.set_title("MATLAB 风格科研绘图示例")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2 * np.pi)

    plt.tight_layout()
    plt.savefig("test_matlab_style.png", dpi=300)
    print("测试图已保存为 test_matlab_style.png")
