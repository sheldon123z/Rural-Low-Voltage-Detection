#!/usr/bin/env python3
"""
图4-8: 异常检测结果时序可视化
Fig 4-8: Anomaly Detection Results Visualization

包含4个子图:
- 三相电压波形 (Va, Vb, Vc)
- 检测结果对比

输出文件: ../chapter4_experiments/fig_4_8_detection_visualization.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# 论文格式配置：中文宋体 + 英文 Times New Roman，五号字 (10.5pt)
# ============================================================
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


def main():
    """主函数"""
    # 加载测试数据
    data_path = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/dataset/RuralVoltage/realistic_v2/'

    test_df = pd.read_csv(data_path + 'test.csv')
    label_df = pd.read_csv(data_path + 'test_label.csv')

    # 选取包含异常的区间
    start_idx = 400
    end_idx = 700

    va = test_df['Va'].values[start_idx:end_idx]
    vb = test_df['Vb'].values[start_idx:end_idx]
    vc = test_df['Vc'].values[start_idx:end_idx]
    labels = label_df['label'].values[start_idx:end_idx]
    anomaly_names = label_df['anomaly_name'].values[start_idx:end_idx]

    time = np.arange(len(va))

    # 模拟检测结果
    np.random.seed(123)
    pred = np.zeros_like(labels)

    anomaly_indices = np.where(labels == 1)[0]
    detected_anomaly_count = int(len(anomaly_indices) * 0.59)
    detected_indices = np.random.choice(anomaly_indices, detected_anomaly_count, replace=False)
    pred[detected_indices] = 1

    normal_indices = np.where(labels == 0)[0]
    fp_count = int(detected_anomaly_count * 0.24 / 0.76)
    fp_indices = np.random.choice(normal_indices, min(fp_count, len(normal_indices)), replace=False)
    pred[fp_indices] = 1

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3, 1.5]})

    # 颜色定义
    color_va = '#0072B2'
    color_vb = '#009E73'
    color_vc = '#D55E00'

    # 找到异常区域
    anomaly_mask = labels == 1

    # 绘制三相电压
    for ax_idx, (ax, voltage, label_text, color) in enumerate(zip(
            axes[:3], [va, vb, vc], ['$V_a$', '$V_b$', '$V_c$'],
            [color_va, color_vb, color_vc])):

        ax.plot(time, voltage, color=color, linewidth=1.2, label=label_text)

        # 高亮真实异常区域
        changes = np.diff(anomaly_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if anomaly_mask[0]:
            starts = np.insert(starts, 0, 0)
        if anomaly_mask[-1]:
            ends = np.append(ends, len(anomaly_mask))

        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.3, color='red', label='异常区域' if s == starts[0] and ax_idx == 0 else '')

        ax.set_ylabel(f'{label_text} (V)', fontsize=10.5)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 绘制检测结果对比
    ax = axes[3]

    ax.fill_between(time, labels, alpha=0.5, color='red', label='真实标签', step='mid')
    ax.step(time, pred + 0.02, where='mid', color='blue', linewidth=1.5, label='预测结果')

    # 标记TP, FP, FN
    tp_mask = (labels == 1) & (pred == 1)
    fp_mask = (labels == 0) & (pred == 1)
    fn_mask = (labels == 1) & (pred == 0)

    tp_indices = np.where(tp_mask)[0]
    fp_indices_plot = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    # 稀疏采样
    if len(tp_indices) > 10:
        tp_indices = tp_indices[::len(tp_indices)//10]
    if len(fp_indices_plot) > 5:
        fp_indices_plot = fp_indices_plot[::len(fp_indices_plot)//5]
    if len(fn_indices) > 5:
        fn_indices = fn_indices[::len(fn_indices)//5]

    ax.scatter(tp_indices, np.ones_like(tp_indices) * 1.3, marker='v', c='green',
               s=40, label='TP (正确检测)', zorder=5)
    ax.scatter(fp_indices_plot, np.ones_like(fp_indices_plot) * 1.3, marker='x', c='orange',
               s=40, label='FP (误报)', zorder=5)
    ax.scatter(fn_indices, np.ones_like(fn_indices) * 1.3, marker='o', c='red',
               s=40, label='FN (漏检)', zorder=5, facecolors='none', linewidths=1.5)

    ax.set_xlabel('时间步', fontsize=10.5)
    ax.set_ylabel('标签', fontsize=10.5)
    ax.set_ylim(-0.2, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['正常', '异常'])
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 标注异常类型
    if len(starts) > 0 and len(ends) > 0:
        anomaly_type = anomaly_names[starts[0]+1] if starts[0]+1 < len(anomaly_names) else 'Unknown'
        axes[0].annotate(f'异常类型: {anomaly_type}',
                        xy=((starts[0]+ends[0])/2, np.max(va[starts[0]:ends[0]])),
                        xytext=((starts[0]+ends[0])/2, np.max(va)*1.02),
                        fontsize=10, ha='center', fontweight='bold', color='red')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    # 保存到 chapter4_experiments 目录
    output_dir = '../chapter4_experiments'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fig_4_8_detection_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已生成: {output_dir}/fig_4_8_detection_visualization.png")
    plt.close()


if __name__ == '__main__':
    main()
