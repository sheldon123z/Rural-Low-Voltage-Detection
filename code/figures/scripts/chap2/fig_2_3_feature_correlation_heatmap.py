#!/usr/bin/env python3
"""
图 2-3: RuralVoltage 16维特征相关性热力图

展示 16 维电气特征之间的 Pearson 相关系数，
揭示三相电压/电流间的高度相关性以及功率因数等特征的独立性。
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from thesis_style import setup_thesis_style, save_thesis_figure, remove_spines, THESIS_COLORS

setup_thesis_style()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    # 读取训练数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'dataset',
                             'RuralVoltage', 'realistic_v2', 'train.csv')
    df = pd.read_csv(data_path)

    # 去除时间戳列
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    # 中文特征名映射
    feature_names_zh = {
        'Va': 'A相电压',
        'Vb': 'B相电压',
        'Vc': 'C相电压',
        'Ia': 'A相电流',
        'Ib': 'B相电流',
        'Ic': 'C相电流',
        'P': '有功功率',
        'Q': '无功功率',
        'S': '视在功率',
        'PF': '功率因数',
        'THD_Va': 'A相谐波',
        'THD_Vb': 'B相谐波',
        'THD_Vc': 'C相谐波',
        'Freq': '频率',
        'V_unbalance': '电压不平衡',
        'I_unbalance': '电流不平衡',
    }

    # 重命名列
    rename_map = {}
    for col in df.columns:
        if col in feature_names_zh:
            rename_map[col] = feature_names_zh[col]
    df = df.rename(columns=rename_map)

    # 去除常数列和全NaN列（nunique=0 或 1，corr() 对这类列返回 NaN）
    df = df.loc[:, df.nunique() > 1]

    # 计算相关系数矩阵
    corr = df.corr()

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 7))

    # 自定义发散色图（蓝-白-红，柔和版）
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'thesis_diverging',
        ['#4878A8', '#FFFFFF', '#C4785C'],
        N=256
    )

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # 绘制热力图
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # 上三角遮罩
    for i in range(len(corr)):
        for j in range(len(corr)):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color='white', linewidth=0))

    # 添加数值标注（仅下三角 + 对角线）
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                val = corr.values[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                fontsize = 6.5 if len(corr) > 12 else 7.5
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=fontsize, color=color)

    # 设置刻度
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, fontsize=8.5, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns, fontsize=8.5)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=30, pad=0.02)
    cbar.set_label('Pearson相关系数', fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 保存
    output_dir = os.path.join(script_dir, '..', '..', 'output', 'chap2')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_2_3_feature_correlation_heatmap.png')
    save_thesis_figure(fig, output_path)


if __name__ == '__main__':
    main()
