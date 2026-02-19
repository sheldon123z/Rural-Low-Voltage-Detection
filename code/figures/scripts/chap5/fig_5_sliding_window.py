"""
fig_5_sliding_window.py
滑动窗口推理示意图 — 展示序列切分、重叠推理和分数聚合过程
输出: Thesis/figures/chap5/fig_5_sliding_window.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines

setup_thesis_style()

np.random.seed(42)

# ── 生成示例电压时序数据 ──────────────────────────────────────────
N = 120
t = np.arange(N)

# 正常段：正弦波 + 低噪声
base = 220 + 10 * np.sin(2 * np.pi * t / 20)
noise = np.random.randn(N) * 2
signal = base + noise

# 注入异常段 [55, 75]
anomaly_start, anomaly_end = 55, 75
signal[anomaly_start:anomaly_end] += np.random.randn(anomaly_end - anomaly_start) * 18 - 15

# ── 滑动窗口参数 ──────────────────────────────────────────────────
seq_len = 30
stride  = 15
windows = [(s, s + seq_len) for s in range(0, N - seq_len + 1, stride)]
n_windows = len(windows)

# ── 模拟异常分数（重叠平均）────────────────────────────────────────
raw_scores  = np.zeros(N)
raw_counts  = np.zeros(N)
anomaly_ratio = 0.15   # 约15%

for (ws, we) in windows:
    seg = signal[ws:we]
    mean_val, std_val = seg.mean(), seg.std() + 1e-6
    # 简单重构误差：Z-score 绝对偏差均值（模拟模型输出）
    err = np.abs((seg - mean_val) / std_val)
    # 异常段的误差人为放大
    for i in range(len(err)):
        if ws + i >= anomaly_start and ws + i < anomaly_end:
            err[i] = err[i] * 3.5 + 0.8
    raw_scores[ws:we] += err
    raw_counts[ws:we] += 1

mask = raw_counts > 0
scores = np.zeros(N)
scores[mask] = raw_scores[mask] / raw_counts[mask]

threshold = np.percentile(scores, (1 - anomaly_ratio) * 100)
labels = (scores > threshold).astype(int)

# ── 配色 ─────────────────────────────────────────────────────────
C_SIGNAL   = '#4878A8'   # 柔和蓝
C_ANOMALY  = '#C85250'   # 红
C_WIN      = ['#72A86D', '#D4A84C', '#7B68C8', '#C4785C', '#5BAAAA']
C_SCORE    = '#808080'
C_THRESH   = '#D4A84C'

# ── 画布 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 6),
                         gridspec_kw={'height_ratios': [2.5, 1.8, 1.5]})

# ──────────────────────────────────────────────────────────────────
# 子图1: 原始时序 + 异常区间标注
# ──────────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(t, signal, color=C_SIGNAL, linewidth=1.0, zorder=3, label='电压 Va')
ax1.axvspan(anomaly_start, anomaly_end, alpha=0.18, color=C_ANOMALY, zorder=1)
ax1.set_ylabel('电压/V', fontsize=10.5)
ax1.set_xlim(0, N - 1)
ax1.set_xticks([])
ax1.legend(fontsize=9, frameon=True, edgecolor='#CCCCCC', fancybox=False, loc='upper right')
ax1.text((anomaly_start + anomaly_end) / 2, ax1.get_ylim()[1] * 0.98,
         '异常区间', ha='center', va='top', fontsize=9, color=C_ANOMALY)
remove_spines(ax1)

# ──────────────────────────────────────────────────────────────────
# 子图2: 滑动窗口切分示意
# ──────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_xlim(0, N - 1)
ax2.set_ylim(-0.5, min(n_windows, 5) + 0.5)

for i, (ws, we) in enumerate(windows[:5]):   # 只展示前5个窗口
    color = C_WIN[i % len(C_WIN)]
    height = 0.6
    y_pos  = n_windows - 1 - i if n_windows <= 5 else 4 - i
    rect = mpatches.FancyBboxPatch(
        (ws, y_pos - height / 2), we - ws, height,
        boxstyle="round,pad=0.5",
        facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.2)
    ax2.add_patch(rect)
    ax2.text((ws + we) / 2, y_pos, f'W{i+1}', ha='center', va='center',
             fontsize=8.5, color=color, fontweight='bold')
    # 左侧标注范围
    ax2.annotate(f'[{ws},{we})', xy=(ws, y_pos - height / 2 - 0.08),
                 fontsize=7.5, color='#606060', ha='left')

ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_ylabel('窗口/Window', fontsize=10.5)
# 标注步幅
ax2.annotate('', xy=(windows[1][0], -0.25), xytext=(windows[0][0], -0.25),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
ax2.text((windows[0][0] + windows[1][0]) / 2, -0.42,
         f'步幅={stride}', ha='center', fontsize=8.5, color='#555555')
ax2.annotate('', xy=(windows[0][1], -0.25), xytext=(windows[0][0], -0.25),
             arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.0))
ax2.text((windows[0][0] + windows[0][1]) / 2, -0.42,
         f'序列长={seq_len}', ha='center', fontsize=8.5, color='#333333')
remove_spines(ax2)

# ──────────────────────────────────────────────────────────────────
# 子图3: 聚合后异常分数
# ──────────────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.fill_between(t, scores, alpha=0.55, color=C_SCORE, step='mid')
ax3.plot(t, scores, color=C_SCORE, linewidth=0.8)
ax3.axhline(threshold, color=C_THRESH, linewidth=1.4, linestyle='--',
            label=f'阈值 τ={threshold:.2f}')
ax3.fill_between(t, scores, threshold, where=(labels == 1),
                 alpha=0.55, color=C_ANOMALY, label='判定异常')
ax3.set_ylabel('异常分数', fontsize=10.5)
ax3.set_xlabel('时间步/Step', fontsize=10.5)
ax3.set_xlim(0, N - 1)
ax3.legend(fontsize=9, frameon=True, edgecolor='#CCCCCC', fancybox=False,
           loc='upper right', ncol=2)
remove_spines(ax3)

plt.tight_layout(h_pad=0.5)

# ── 保存 ─────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../Thesis/figures/chap5/fig_5_sliding_window.png'
)
save_thesis_figure(fig, out_path)
print(f'Saved: {os.path.abspath(out_path)}')
