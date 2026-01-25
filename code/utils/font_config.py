"""
中文字体配置模块

本模块提供统一的中文字体检测和配置功能，支持 matplotlib 和 plotly。

系统可用字体（按优先级）:
1. WenQuanYi Micro Hei (文泉驿微米黑) - 主推
2. Noto Sans CJK SC (Google Noto 简体中文)
3. AR PL UKai CN (文鼎楷体)

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
import warnings
from typing import Optional, List, Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 抑制字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ============================================================================
# 字体配置常量
# ============================================================================

FONT_CONFIG = {
    # 中文字体优先级列表（按优先级排序）
    "chinese_fonts": [
        "WenQuanYi Micro Hei",      # 文泉驿微米黑 - 最推荐
        "WenQuanYi Zen Hei",        # 文泉驿正黑
        "Noto Sans CJK SC",         # Google Noto 简体中文
        "Noto Serif CJK SC",        # Google Noto 宋体
        "AR PL UKai CN",            # 文鼎楷体
        "AR PL UMing CN",           # 文鼎明体
        "SimHei",                   # 黑体（Windows）
        "Microsoft YaHei",          # 微软雅黑（Windows）
        "SimSun",                   # 宋体（Windows）
    ],

    # 英文字体优先级列表
    "english_fonts": [
        "Times New Roman",
        "DejaVu Serif",
        "Liberation Serif",
        "FreeSerif",
    ],

    # 数学字体集
    "math_fontset": "stix",

    # 论文规范字号（单位：pt）
    "thesis_sizes": {
        "title": 12,        # 图标题：约五号
        "label": 10.5,      # 坐标轴标签：五号 (10.5pt)
        "tick": 9,          # 刻度标签：小五号 (9pt)
        "legend": 9,        # 图例：小五号 (9pt)
        "annotation": 9,    # 注释文字：小五号
    },

    # 图形尺寸（单位：英寸）
    "figure_sizes": {
        "single_column": (3.5, 2.8),    # 单栏图
        "double_column": (7.0, 4.0),    # 双栏图
        "full_width": (6.0, 4.0),       # 全宽图
        "square": (4.0, 4.0),           # 正方形图
        "wide": (8.0, 4.0),             # 宽图
    },

    # 分辨率
    "dpi": {
        "display": 100,     # 显示质量
        "save": 300,        # 保存质量
    },
}

# Plotly 中文字体配置
PLOTLY_FONT_CONFIG = {
    "family": "WenQuanYi Micro Hei, Noto Sans CJK SC, sans-serif",
    "size": 12,
    "color": "black",
}

# ============================================================================
# 字体检测函数
# ============================================================================

def get_system_fonts() -> List[str]:
    """
    获取系统所有可用字体名称列表

    Returns:
        List[str]: 字体名称列表
    """
    fonts = set()
    for font in font_manager.fontManager.ttflist:
        fonts.add(font.name)
    return sorted(list(fonts))


def find_chinese_font() -> Optional[str]:
    """
    查找系统中可用的中文字体

    按优先级顺序查找，返回第一个可用的中文字体名称。

    Returns:
        Optional[str]: 找到的字体名称，未找到返回 None
    """
    system_fonts = get_system_fonts()
    system_fonts_lower = [f.lower() for f in system_fonts]

    for font in FONT_CONFIG["chinese_fonts"]:
        # 精确匹配
        if font in system_fonts:
            return font

        # 模糊匹配（忽略大小写）
        font_lower = font.lower()
        for i, sf in enumerate(system_fonts_lower):
            if font_lower in sf or sf in font_lower:
                return system_fonts[i]

    # 尝试通配符匹配
    keywords = ["wenquanyi", "noto", "cjk", "chinese", "simsun", "simhei"]
    for sf_lower, sf in zip(system_fonts_lower, system_fonts):
        for kw in keywords:
            if kw in sf_lower:
                return sf

    return None


def find_english_font() -> Optional[str]:
    """
    查找系统中可用的英文字体

    Returns:
        Optional[str]: 找到的字体名称，未找到返回 None
    """
    system_fonts = get_system_fonts()

    for font in FONT_CONFIG["english_fonts"]:
        if font in system_fonts:
            return font

    return "DejaVu Sans"  # 默认后备字体


# ============================================================================
# Matplotlib 配置函数
# ============================================================================

def setup_matplotlib_chinese(verbose: bool = False) -> str:
    """
    配置 matplotlib 使用中文字体

    Args:
        verbose: 是否输出详细信息

    Returns:
        str: 实际使用的中文字体名称
    """
    # 查找中文字体
    cn_font = find_chinese_font()
    en_font = find_english_font()

    if cn_font is None:
        warnings.warn("未找到中文字体，图表中文可能无法正常显示")
        cn_font = "DejaVu Sans"

    if verbose:
        print(f"使用中文字体: {cn_font}")
        print(f"使用英文字体: {en_font}")

    # 配置 matplotlib
    plt.rcParams["font.sans-serif"] = [cn_font, en_font, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = FONT_CONFIG["math_fontset"]

    return cn_font


def setup_matplotlib_thesis_style(chapter: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """
    配置 matplotlib 为论文样式

    按照北京林业大学硕士学位论文格式要求配置所有绑定参数。

    Args:
        chapter: 章节号，用于图表编号
        verbose: 是否输出详细信息

    Returns:
        Dict: 配置信息字典
    """
    # 设置中文字体
    cn_font = setup_matplotlib_chinese(verbose=verbose)

    sizes = FONT_CONFIG["thesis_sizes"]
    dpi = FONT_CONFIG["dpi"]

    # 字体大小
    plt.rcParams["axes.titlesize"] = sizes["title"]
    plt.rcParams["axes.labelsize"] = sizes["label"]
    plt.rcParams["xtick.labelsize"] = sizes["tick"]
    plt.rcParams["ytick.labelsize"] = sizes["tick"]
    plt.rcParams["legend.fontsize"] = sizes["legend"]

    # 图形设置
    plt.rcParams["figure.dpi"] = dpi["display"]
    plt.rcParams["savefig.dpi"] = dpi["save"]
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1

    # 坐标轴设置
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True

    # 刻度设置
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False

    # 图例设置
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.borderpad"] = 0.4

    # 线条设置
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 6

    # 网格设置
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5

    return {
        "chinese_font": cn_font,
        "chapter": chapter,
        "sizes": sizes,
        "dpi": dpi,
    }


# ============================================================================
# Plotly 配置函数
# ============================================================================

def get_plotly_layout_template() -> Dict[str, Any]:
    """
    获取 Plotly 中文布局模板

    Returns:
        Dict: Plotly 布局配置字典
    """
    cn_font = find_chinese_font() or "sans-serif"
    font_family = f"{cn_font}, Noto Sans CJK SC, sans-serif"

    sizes = FONT_CONFIG["thesis_sizes"]

    return {
        "font": {
            "family": font_family,
            "size": sizes["label"],
            "color": "black",
        },
        "title": {
            "font": {
                "family": font_family,
                "size": sizes["title"],
            },
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "title": {"font": {"size": sizes["label"]}},
            "tickfont": {"size": sizes["tick"]},
            "showgrid": True,
            "gridwidth": 0.5,
            "gridcolor": "#e0e0e0",
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "ticks": "inside",
        },
        "yaxis": {
            "title": {"font": {"size": sizes["label"]}},
            "tickfont": {"size": sizes["tick"]},
            "showgrid": True,
            "gridwidth": 0.5,
            "gridcolor": "#e0e0e0",
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "ticks": "inside",
        },
        "legend": {
            "font": {"size": sizes["legend"]},
            "bordercolor": "black",
            "borderwidth": 1,
        },
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }


def setup_plotly_chinese() -> Dict[str, Any]:
    """
    配置 Plotly 使用中文字体

    Returns:
        Dict: 配置信息
    """
    try:
        import plotly.io as pio

        template = get_plotly_layout_template()

        # 创建自定义模板
        pio.templates["chinese_thesis"] = {
            "layout": template
        }

        # 设置为默认模板
        pio.templates.default = "plotly_white+chinese_thesis"

        return {"success": True, "template": "chinese_thesis"}
    except ImportError:
        warnings.warn("Plotly 未安装，跳过 Plotly 配置")
        return {"success": False, "error": "Plotly not installed"}


# ============================================================================
# 学术配色方案
# ============================================================================

# 主配色方案（色盲友好）
ACADEMIC_COLORS = [
    "#1f77b4",  # 蓝色 - Primary
    "#ff7f0e",  # 橙色
    "#2ca02c",  # 绿色
    "#d62728",  # 红色
    "#9467bd",  # 紫色
    "#8c564b",  # 棕色
    "#e377c2",  # 粉色
    "#7f7f7f",  # 灰色
    "#bcbd22",  # 黄绿色
    "#17becf",  # 青色
]

# 高对比度配色（适合打印）
HIGH_CONTRAST_COLORS = [
    "#000000",  # 黑色
    "#E69F00",  # 橙色
    "#56B4E9",  # 天蓝色
    "#009E73",  # 青绿色
    "#F0E442",  # 黄色
    "#0072B2",  # 深蓝色
    "#D55E00",  # 深橙色
    "#CC79A7",  # 粉紫色
]

# 异常类型配色
ANOMALY_COLORS = {
    0: "#2ca02c",  # 正常 - 绿色
    1: "#d62728",  # 欠压 - 红色
    2: "#ff7f0e",  # 过压 - 橙色
    3: "#9467bd",  # 骤降 - 紫色
    4: "#1f77b4",  # 谐波 - 蓝色
    5: "#8c564b",  # 不平衡 - 棕色
}

# 异常类型名称（中英文）
ANOMALY_NAMES = {
    0: "正常 (Normal)",
    1: "欠压 (Undervoltage)",
    2: "过压 (Overvoltage)",
    3: "骤降 (Voltage Sag)",
    4: "谐波 (Harmonic)",
    5: "不平衡 (Unbalance)",
}

# 标记符号
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

# 线型
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


# ============================================================================
# 便捷函数
# ============================================================================

def get_color(index: int, palette: str = "academic") -> str:
    """
    获取指定索引的颜色

    Args:
        index: 颜色索引
        palette: 配色方案 ("academic", "high_contrast", "anomaly")

    Returns:
        str: 颜色十六进制值
    """
    if palette == "high_contrast":
        colors = HIGH_CONTRAST_COLORS
    elif palette == "anomaly":
        return ANOMALY_COLORS.get(index, ACADEMIC_COLORS[index % len(ACADEMIC_COLORS)])
    else:
        colors = ACADEMIC_COLORS

    return colors[index % len(colors)]


def get_marker(index: int) -> str:
    """
    获取指定索引的标记符号

    Args:
        index: 标记索引

    Returns:
        str: 标记符号
    """
    return MARKERS[index % len(MARKERS)]


def get_linestyle(index: int) -> str:
    """
    获取指定索引的线型

    Args:
        index: 线型索引

    Returns:
        str: 线型
    """
    return LINE_STYLES[index % len(LINE_STYLES)]


# ============================================================================
# 初始化
# ============================================================================

def init_all(verbose: bool = False) -> Dict[str, Any]:
    """
    初始化所有字体配置

    Args:
        verbose: 是否输出详细信息

    Returns:
        Dict: 配置信息
    """
    result = {}

    # 配置 matplotlib
    result["matplotlib"] = setup_matplotlib_thesis_style(verbose=verbose)

    # 配置 plotly
    result["plotly"] = setup_plotly_chinese()

    if verbose:
        print("\n字体配置完成:")
        print(f"  - Matplotlib 中文字体: {result['matplotlib']['chinese_font']}")
        print(f"  - Plotly 配置: {'成功' if result['plotly']['success'] else '失败'}")

    return result


# 模块导入时自动初始化 matplotlib
_init_result = setup_matplotlib_thesis_style(verbose=False)


# ============================================================================
# 测试函数
# ============================================================================

def test_chinese_font():
    """测试中文字体显示"""
    import numpy as np

    # 初始化
    setup_matplotlib_thesis_style(verbose=True)

    # 创建测试图
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    ax.plot(x, y1, label="正弦函数 sin(x)", color=get_color(0), marker=get_marker(0), markevery=10)
    ax.plot(x, y2, label="余弦函数 cos(x)", color=get_color(1), marker=get_marker(1), markevery=10)

    ax.set_xlabel("时间 (Time) / s")
    ax.set_ylabel("幅值 (Amplitude) / V")
    ax.set_title("图4.1 中文字体测试图\nFigure 4.1 Chinese Font Test")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig("/tmp/font_test.png", dpi=300)
    print("\n测试图已保存至: /tmp/font_test.png")
    plt.close()


if __name__ == "__main__":
    test_chinese_font()
