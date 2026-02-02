#!/usr/bin/env python3
"""
论文图表统一样式配置

符合毕业论文格式要求：
- 中文五号宋体 (10.5pt)
- 英文五号 Times New Roman
- 单位使用 "/" 分隔（如 "电压/V"）
- 朴素科研风格，柔和配色
- 只输出 PNG，300 DPI
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# 字体配置
# ============================================================
def setup_thesis_style():
    """设置论文格式的 matplotlib 样式"""
    plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman']
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 100

# ============================================================
# 柔和科研配色方案（替代鲜艳色）
# ============================================================
THESIS_COLORS = {
    # 主色调（柔和蓝绿系）
    'primary': '#4878A8',      # 柔和蓝
    'secondary': '#72A86D',    # 柔和绿
    'accent': '#C4785C',       # 柔和橙
    'warning': '#D4A84C',      # 柔和黄
    'purple': '#9B7BB8',       # 柔和紫

    # 中性色
    'neutral': '#808080',      # 中性灰
    'light_gray': '#B0B0B0',   # 浅灰
    'dark_gray': '#505050',    # 深灰

    # 特殊用途
    'positive': '#6BA86B',     # 正常/正向（柔和绿）
    'negative': '#C46B5C',     # 异常/负向（柔和红）
    'threshold': '#505050',    # 阈值线（深灰）
}

# TimesNet 系列模型配色
TIMESNET_COLORS = [
    '#4878A8',   # TimesNet - 柔和蓝
    '#5B9BD5',   # VoltageTimesNet - 浅蓝
    '#72A86D',   # VoltageTimesNet_v2 - 柔和绿
    '#9B7BB8',   # TPATimesNet - 柔和紫
]

# 其他模型配色
OTHER_COLORS = [
    '#808080',   # DLinear - 灰色
    '#606060',   # PatchTST - 深灰
]

# 三相电压配色
PHASE_COLORS = {
    'Va': '#4878A8',   # A相 - 柔和蓝
    'Vb': '#72A86D',   # B相 - 柔和绿
    'Vc': '#C4785C',   # C相 - 柔和橙
}

# 性能指标配色
METRIC_COLORS = {
    'precision': '#4878A8',    # 精确率 - 柔和蓝
    'recall': '#72A86D',       # 召回率 - 柔和绿
    'f1': '#C4785C',           # F1分数 - 柔和橙
    'accuracy': '#9B7BB8',     # 准确率 - 柔和紫
}

# 分类结果配色
CLASSIFICATION_COLORS = {
    'normal': '#72A86D',       # 正常 - 柔和绿
    'anomaly': '#C4785C',      # 异常 - 柔和橙/红
    'tp': '#6BA86B',           # TP - 绿
    'fp': '#D4A84C',           # FP - 黄
    'fn': '#C46B5C',           # FN - 红
    'tn': '#72A86D',           # TN - 绿
}

# ============================================================
# 工具函数
# ============================================================
def get_model_colors(models):
    """根据模型列表返回对应颜色"""
    colors = []
    for model in models:
        if 'VoltageTimesNet_v2' in model or 'V-TimesNet_v2' in model:
            colors.append(TIMESNET_COLORS[2])
        elif 'VoltageTimesNet' in model or 'V-TimesNet' in model:
            colors.append(TIMESNET_COLORS[1])
        elif 'TPATimesNet' in model or 'TPA-TimesNet' in model:
            colors.append(TIMESNET_COLORS[3])
        elif 'TimesNet' in model:
            colors.append(TIMESNET_COLORS[0])
        elif 'DLinear' in model:
            colors.append(OTHER_COLORS[0])
        elif 'PatchTST' in model:
            colors.append(OTHER_COLORS[1])
        else:
            colors.append(THESIS_COLORS['neutral'])
    return colors


def save_thesis_figure(fig, output_path, tight=True):
    """保存论文格式的图表（仅 PNG）"""
    if tight:
        fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    print(f"已生成: {output_path}")
    plt.close(fig)


def remove_spines(ax, keep_bottom=True, keep_left=True):
    """移除多余的边框（三线表风格）"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not keep_bottom:
        ax.spines['bottom'].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)
