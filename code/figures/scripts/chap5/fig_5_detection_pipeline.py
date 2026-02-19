"""
fig_5_detection_pipeline.py
完整数据处理流水线图 — 从 CSV 文件到异常标签的全流程
输出: Thesis/figures/chap5/fig_5_detection_pipeline.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from thesis_style import setup_thesis_style, save_thesis_figure

setup_thesis_style()

fig, ax = plt.subplots(figsize=(9.0, 4.2))
ax.set_xlim(0, 12)
ax.set_ylim(-0.2, 4.8)
ax.axis('off')

# ── 配色方案 ─────────────────────────────────────────────────────
COLORS = {
    'input':    ('#D6EAF8', '#5DADE2'),   # 浅蓝
    'preproc':  ('#D5F5E3', '#58D68D'),   # 浅绿
    'model':    ('#FDEBD0', '#F0A500'),   # 浅橙
    'output':   ('#FDEDEC', '#C85250'),   # 浅红
    'arrow':    '#7F8C8D',
    'param':    '#7F5214',
    'param_bg': '#FEF9E7',
    'param_ec': '#D4A84C',
}

# ── 步骤定义 ─────────────────────────────────────────────────────
steps = [
    {
        'label': 'CSV\n数据输入',
        'sublabel': 'N × 16维',
        'params': ['· 16 个电气特征列', '· 最少 50 行', '· 格式/完整性校验'],
        'x': 0.2, 'color': 'input',
    },
    {
        'label': 'Z-score\n标准化',
        'sublabel': '逐列归一化',
        'params': ['μ̂ = mean(X)', 'σ̂ = std(X) + ε', 'X̂ = (X - μ̂) / σ̂'],
        'x': 2.2, 'color': 'preproc',
    },
    {
        'label': '滑动窗口\n切分',
        'sublabel': '重叠分段',
        'params': ['seq_len = 50', 'stride = 25', '共 2(N/50)-1 个窗口'],
        'x': 4.2, 'color': 'preproc',
    },
    {
        'label': 'VoltageTimesNet\n推理',
        'sublabel': '逐窗口重构',
        'params': ['d_model=128, e_layers=3', 'top_k=2, num_kernels=8', '输出: 重构误差序列'],
        'x': 6.2, 'color': 'model',
    },
    {
        'label': '分数\n聚合',
        'sublabel': '重叠均值',
        'params': ['s[t] = Σ(w_i·e_i[t])/Σw_i', '平滑重叠区跳变', '生成全序列分数向量'],
        'x': 8.2, 'color': 'model',
    },
    {
        'label': '百分位\n阈值',
        'sublabel': '二值判定',
        'params': ['τ = percentile(s, 97.9)', 'anomaly_ratio=2.085%', '超阈值 → 标记异常'],
        'x': 10.2, 'color': 'output',
    },
]

bw, bh_title, bh_body = 1.75, 1.05, 1.6
box_y = 2.5   # 主框起始 y

# ── 降级路径（统计方法）──────────────────────────────────────────
# 在第4步VoltageTimesNet下方添加降级备注
fallback_y = 0.25

# 降级路径框
fb_box = FancyBboxPatch((5.8, fallback_y), 4.4, 0.78,
                         boxstyle='round,pad=0.12',
                         facecolor='#F0F0F0', edgecolor='#AAAAAA', lw=1.0, zorder=2,
                         linestyle='dashed')
ax.add_patch(fb_box)
ax.text(8.0, fallback_y + 0.39, '降级方案：加权 Z-score 统计检测（模型不可用时）',
        ha='center', va='center', fontsize=9.0, color='#666666', style='italic')
# 降级条件标注
ax.annotate('', xy=(7.1, fallback_y + 0.78), xytext=(7.1, box_y - 0.1),
            arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.0, linestyle='dashed'))
ax.text(7.55, box_y - 0.35, '模型\n不可用', ha='center', va='center',
        fontsize=8.0, color='#999999', style='italic')

# ── 绘制每个步骤 ─────────────────────────────────────────────────
for i, step in enumerate(steps):
    x = step['x']
    fc, ec = COLORS[step['color']]

    # 标题框
    title_box = FancyBboxPatch((x, box_y + bh_body), bw, bh_title,
                               boxstyle='round,pad=0.1',
                               facecolor=ec, edgecolor=ec, lw=0, zorder=3)
    ax.add_patch(title_box)
    ax.text(x + bw / 2, box_y + bh_body + bh_title / 2 + 0.06,
            step['label'], ha='center', va='center',
            fontsize=9.5, color='white', fontweight='bold', zorder=4,
            linespacing=1.3)
    ax.text(x + bw / 2, box_y + bh_body + 0.16,
            step['sublabel'], ha='center', va='center',
            fontsize=8.5, color='white', alpha=0.9, zorder=4)

    # 参数框（下半部分）
    param_box = FancyBboxPatch((x, box_y), bw, bh_body,
                               boxstyle='round,pad=0.1',
                               facecolor=fc, edgecolor=ec, lw=1.2, zorder=3)
    ax.add_patch(param_box)
    for j, param in enumerate(step['params']):
        py = box_y + bh_body - 0.30 - j * 0.45
        ax.text(x + 0.12, py, param, ha='left', va='center',
                fontsize=8.2, color='#444444', zorder=4)

    # 步骤序号
    circle = plt.Circle((x + 0.18, box_y + bh_body + bh_title - 0.18),
                         0.16, color='white', zorder=5)
    ax.add_patch(circle)
    ax.text(x + 0.18, box_y + bh_body + bh_title - 0.18,
            str(i + 1), ha='center', va='center',
            fontsize=9.0, color=ec, fontweight='bold', zorder=6)

    # 箭头（非最后一步）
    if i < len(steps) - 1:
        arrow_x = x + bw + 0.02
        ax.annotate('', xy=(x + bw + 0.42, box_y + bh_body + bh_title / 2),
                    xytext=(arrow_x, box_y + bh_body + bh_title / 2),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                    lw=1.5, mutation_scale=14))

# ── 输出标签 ─────────────────────────────────────────────────────
out_x = steps[-1]['x'] + bw + 0.52
ax.text(out_x - 0.02, box_y + bh_body + bh_title / 2,
        '异常标签\n(0/1序列)\n+ 可视化',
        ha='center', va='center', fontsize=9.5, color='#C85250', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec='#C85250', lw=1.2))

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../Thesis/figures/chap5/fig_5_detection_pipeline.png'
)
save_thesis_figure(fig, out_path)
print(f'Saved: {os.path.abspath(out_path)}')
