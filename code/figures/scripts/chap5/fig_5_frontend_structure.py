"""
fig_5_frontend_structure.py
前端页面导航结构图 — 展示 SPA 5页面的功能模块与数据依赖
输出: Thesis/figures/chap5/fig_5_frontend_structure.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from thesis_style import setup_thesis_style, save_thesis_figure

setup_thesis_style()

fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# ── 配色 ─────────────────────────────────────────────────────────
C_SIDEBAR  = '#4878A8'
C_PAGE     = '#FFFFFF'
C_BORDER   = '#4878A8'
C_API      = '#D4A84C'
C_ARROW    = '#888888'
C_TEXT_W   = '#FFFFFF'
C_TEXT_D   = '#333333'
C_COMP     = '#EAF2FB'
C_COMP_BD  = '#7AADD4'

def draw_box(ax, x, y, w, h, label, sublabel='', fc=C_PAGE, ec=C_BORDER, lw=1.2,
             text_color=C_TEXT_D, fontsize=10.0, r=0.15):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    ty = y + h / 2 + (0.1 if sublabel else 0)
    ax.text(x + w / 2, ty, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.18, sublabel, ha='center', va='center',
                fontsize=8.5, color=text_color if text_color == C_TEXT_W else '#666666',
                zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, bidirectional=False):
    style = '<->' if bidirectional else '->'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=C_ARROW,
                                lw=1.2, connectionstyle='arc3,rad=0'))

def draw_api_label(ax, x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=8.0,
            color='#555555', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc='#FFF8E7', ec=C_API, lw=0.8, alpha=0.9))

# ── 侧边栏 ───────────────────────────────────────────────────────
draw_box(ax, 0.3, 0.5, 1.5, 5.0, '', fc=C_SIDEBAR, ec=C_SIDEBAR, lw=0)
ax.text(1.05, 5.2, '侧边栏导航', ha='center', va='center',
        fontsize=9.5, color=C_SIDEBAR, fontweight='bold')

nav_items = ['总览仪表板', '异常检测', '检测历史', '模型对比', '系统原理']
nav_cols = ['#72A86D', '#4878A8', '#7B68C8', '#C4785C', '#5BAAAA']
for i, (item, col) in enumerate(zip(nav_items, nav_cols)):
    yy = 4.3 - i * 0.85
    draw_box(ax, 0.38, yy, 1.34, 0.62, item,
             fc=col, ec=col, lw=0, text_color=C_TEXT_W, fontsize=9.0, r=0.1)

# ── 5个页面 ──────────────────────────────────────────────────────
pages = [
    {
        'title': '仪表板',
        'en': 'Dashboard',
        'x': 2.25, 'y': 4.1,
        'comps': ['指标卡片×4', '雷达图', '近期检测列表'],
        'api': '/metrics\n/detect/history',
        'col': '#EBF5FB',
    },
    {
        'title': '异常检测',
        'en': 'Detect',
        'x': 4.3, 'y': 4.1,
        'comps': ['CSV上传', '参数滑动条', '电压时序图', '分数分布图'],
        'api': '/detect/upload\n/detect/sample',
        'col': '#EAF5EA',
    },
    {
        'title': '检测历史',
        'en': 'History',
        'x': 6.35, 'y': 4.1,
        'comps': ['任务列表', '关键词筛选', '结果详情'],
        'api': '/detect/history\n/detect/{id}',
        'col': '#F3EDFB',
    },
    {
        'title': '模型对比',
        'en': 'Models',
        'x': 4.3, 'y': 1.3,
        'comps': ['模型卡片×5', 'F1对比图', '雷达图'],
        'api': '/models\n/models/current',
        'col': '#FDF3E3',
    },
    {
        'title': '系统原理',
        'en': 'About',
        'x': 6.35, 'y': 1.3,
        'comps': ['算法描述', '技术亮点', 'FFT示意图'],
        'api': '（静态内容）',
        'col': '#F5F5F5',
    },
]

pw, ph = 1.8, 2.5
for pg in pages:
    x, y = pg['x'], pg['y']
    # 页面主框
    draw_box(ax, x, y, pw, ph, '', fc=pg['col'], ec=C_BORDER, lw=1.3, r=0.12)
    # 标题
    ax.text(x + pw / 2, y + ph - 0.25, pg['title'],
            ha='center', va='center', fontsize=10.5,
            color=C_TEXT_D, fontweight='bold', zorder=5)
    ax.text(x + pw / 2, y + ph - 0.50, f"({pg['en']})",
            ha='center', va='center', fontsize=8.5, color='#666666', zorder=5)
    # 组件列表
    for j, comp in enumerate(pg['comps']):
        cy = y + ph - 0.85 - j * 0.45
        if cy > y + 0.55:
            comp_box = FancyBboxPatch((x + 0.12, cy - 0.16), pw - 0.24, 0.34,
                                     boxstyle='round,pad=0.05',
                                     facecolor=C_COMP, edgecolor=C_COMP_BD, lw=0.7, zorder=4)
            ax.add_patch(comp_box)
            ax.text(x + pw / 2, cy + 0.01, comp, ha='center', va='center',
                    fontsize=8.0, color='#333333', zorder=5)
    # API 标签（底部）
    ax.text(x + pw / 2, y + 0.28, pg['api'], ha='center', va='center',
            fontsize=7.5, color='#8B6914', style='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.12', fc='#FFF8E7', ec=C_API, lw=0.7, alpha=0.85))

# ── 侧边栏到页面的导航箭头 ──────────────────────────────────────
arrow_map = [
    (1.05, 4.61, 2.25, 5.1 + 0.5),   # 仪表板
    (1.05, 3.76, 2.25 + pw + 2.05, 5.1 + 0.5),   # 检测
    (1.05, 2.91, 2.25 + pw + 4.1, 5.1 + 0.5),   # 历史
    (1.05, 2.06, 2.25 + pw + 2.05, 1.3 + 2.5),   # 模型
    (1.05, 1.21, 2.25 + pw + 4.1, 1.3 + 2.5),   # 原理
]

for x1, y1, x2, y2 in arrow_map:
    ax.annotate('', xy=(x2, y2), xytext=(x1 + 0.3, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.0, connectionstyle='arc3,rad=0.05'))

# ── 页面间导航标注 ───────────────────────────────────────────────
# 检测→历史
ax.annotate('', xy=(6.35, 5.1 + 0.5), xytext=(2.25 + pw + 0.05, 5.1 + 0.5),
            arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.9, linestyle='dashed'))
ax.text(5.3, 5.75, '跳转详情', ha='center', fontsize=8.0, color='#AAAAAA')

# 仪表板→检测 (点击recent item)
ax.annotate('', xy=(4.3, 5.1 + 0.9), xytext=(2.25 + pw, 5.1 + 0.9),
            arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.9, linestyle='dashed'))
ax.text(3.9, 5.85, '点击记录', ha='center', fontsize=8.0, color='#AAAAAA')

# ── 标题 ─────────────────────────────────────────────────────────
ax.text(5.6, 5.78, '前端单页应用（SPA）页面结构与数据依赖',
        ha='center', va='center', fontsize=11.0, color='#333333', fontweight='bold')

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../Thesis/figures/chap5/fig_5_frontend_structure.png'
)
save_thesis_figure(fig, out_path)
print(f'Saved: {os.path.abspath(out_path)}')
