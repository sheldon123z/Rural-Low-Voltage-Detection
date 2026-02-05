#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电压异常类型示意图生成脚本
用于论文第二章：展示5种电压异常的波形特征

作者: 郑晓东
日期: 2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import os

from thesis_style import get_output_dir

# 设置中文字体（与 thesis_style.py 一致）
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

# 不再需要 FontProperties，直接用 rcParams 全局设置
font_zh = None
font_zh_small = None
font_zh_title = None


def generate_normal_voltage(t, base_voltage=220, frequency=50, noise_level=2):
    """生成正常电压波形 - 稳定的220V正弦波"""
    voltage = base_voltage * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    voltage += np.random.normal(0, noise_level, len(t))
    return voltage


def generate_undervoltage(t, base_voltage=220, frequency=50, undervoltage_ratio=0.85):
    """生成欠压波形 - 电压低于198V (约90%额定值)"""
    reduced_voltage = base_voltage * undervoltage_ratio
    voltage = reduced_voltage * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    voltage += np.random.normal(0, 2, len(t))
    return voltage


def generate_overvoltage(t, base_voltage=220, frequency=50, overvoltage_ratio=1.12):
    """生成过压波形 - 电压高于235V (约107%额定值)"""
    elevated_voltage = base_voltage * overvoltage_ratio
    voltage = elevated_voltage * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    voltage += np.random.normal(0, 2, len(t))
    return voltage


def generate_voltage_sag(t, base_voltage=220, frequency=50, sag_start=0.04, sag_duration=0.02, sag_depth=0.7):
    """生成电压骤降波形 - 短时间内电压突然下降30%然后恢复"""
    voltage = base_voltage * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    envelope = np.ones_like(t)
    sag_end = sag_start + sag_duration
    sag_mask = (t >= sag_start) & (t <= sag_end)
    envelope[sag_mask] = sag_depth

    transition_time = 0.002
    fall_mask = (t >= sag_start - transition_time) & (t < sag_start)
    if np.any(fall_mask):
        envelope[fall_mask] = 1 - (1 - sag_depth) * (t[fall_mask] - (sag_start - transition_time)) / transition_time
    rise_mask = (t > sag_end) & (t <= sag_end + transition_time)
    if np.any(rise_mask):
        envelope[rise_mask] = sag_depth + (1 - sag_depth) * (t[rise_mask] - sag_end) / transition_time

    voltage = voltage * envelope
    voltage += np.random.normal(0, 2, len(t))
    return voltage


def generate_harmonic_distortion(t, base_voltage=220, frequency=50):
    """生成谐波畸变波形 - THD约15%"""
    voltage = base_voltage * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    voltage += base_voltage * np.sqrt(2) * 0.08 * np.sin(2 * np.pi * 3 * frequency * t + np.pi/6)
    voltage += base_voltage * np.sqrt(2) * 0.06 * np.sin(2 * np.pi * 5 * frequency * t + np.pi/4)
    voltage += base_voltage * np.sqrt(2) * 0.04 * np.sin(2 * np.pi * 7 * frequency * t + np.pi/3)
    voltage += base_voltage * np.sqrt(2) * 0.02 * np.sin(2 * np.pi * 9 * frequency * t + np.pi/5)
    voltage += np.random.normal(0, 2, len(t))
    return voltage


def generate_three_phase_unbalance(t, base_voltage=220, frequency=50):
    """生成三相不平衡波形 - 三相电压幅值不对称"""
    va = base_voltage * 1.05 * np.sqrt(2) * np.sin(2 * np.pi * frequency * t)
    vb = base_voltage * 0.92 * np.sqrt(2) * np.sin(2 * np.pi * frequency * t - 2*np.pi/3)
    vc = base_voltage * 1.0 * np.sqrt(2) * np.sin(2 * np.pi * frequency * t + 2*np.pi/3)
    va += np.random.normal(0, 2, len(t))
    vb += np.random.normal(0, 2, len(t))
    vc += np.random.normal(0, 2, len(t))
    return va, vb, vc


def main():
    """主函数：生成电压异常类型示意图"""
    np.random.seed(42)

    duration = 0.08
    fs = 10000
    t = np.linspace(0, duration, int(fs * duration))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08, left=0.06, right=0.98)

    # 颜色方案
    normal_color = '#2E7D32'
    anomaly_color = '#C62828'
    phase_colors = ['#1976D2', '#F57C00', '#388E3C']
    reference_color = '#757575'

    # (a) 正常电压
    ax = axes[0, 0]
    voltage_normal = generate_normal_voltage(t)
    ax.plot(t * 1000, voltage_normal, color=normal_color, linewidth=1.2)
    ax.axhline(y=220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=-220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(a) 正常电压', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.text(70, 340, 'Vp=311V\n(220V)', fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # (b) 欠压
    ax = axes[0, 1]
    voltage_under = generate_undervoltage(t)
    ax.plot(t * 1000, voltage_under, color=anomaly_color, linewidth=1.2)
    ax.axhline(y=220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=187*np.sqrt(2), color=anomaly_color, linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-187*np.sqrt(2), color=anomaly_color, linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(b) 欠压', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.annotate('', xy=(75, 187*np.sqrt(2)), xytext=(75, 220*np.sqrt(2)),
                arrowprops=dict(arrowstyle='<->', color=anomaly_color, lw=1.5))
    ax.text(77, (187+220)*np.sqrt(2)/2, '欠压\n<198V', fontsize=8, ha='left', va='center',
            color=anomaly_color)

    # (c) 过压
    ax = axes[0, 2]
    voltage_over = generate_overvoltage(t)
    ax.plot(t * 1000, voltage_over, color=anomaly_color, linewidth=1.2)
    ax.axhline(y=220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=246*np.sqrt(2), color=anomaly_color, linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-246*np.sqrt(2), color=anomaly_color, linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(c) 过压', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.annotate('', xy=(75, 246*np.sqrt(2)), xytext=(75, 220*np.sqrt(2)),
                arrowprops=dict(arrowstyle='<->', color=anomaly_color, lw=1.5))
    ax.text(77, (246+220)*np.sqrt(2)/2, '过压\n>235V', fontsize=8, ha='left', va='center',
            color=anomaly_color)

    # (d) 电压骤降
    ax = axes[1, 0]
    voltage_sag = generate_voltage_sag(t)
    ax.plot(t * 1000, voltage_sag, color=anomaly_color, linewidth=1.2)
    ax.axhline(y=220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-220*np.sqrt(2), color=reference_color, linestyle='--', linewidth=0.8, alpha=0.5)
    sag_region = Rectangle((40, -350), 20, 700, linewidth=0, facecolor='yellow', alpha=0.2)
    ax.add_patch(sag_region)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(d) 电压骤降', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.annotate('骤降30%', xy=(50, 154*np.sqrt(2)), xytext=(55, 280),
                fontsize=8, color=anomaly_color, 
                arrowprops=dict(arrowstyle='->', color=anomaly_color, lw=1))

    # (e) 谐波畸变
    ax = axes[1, 1]
    voltage_harmonic = generate_harmonic_distortion(t)
    voltage_ref = 220 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
    ax.plot(t * 1000, voltage_ref, color=reference_color, linewidth=0.8, linestyle='--', alpha=0.5, label='理想波形')
    ax.plot(t * 1000, voltage_harmonic, color=anomaly_color, linewidth=1.2, label='畸变波形')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(e) 谐波畸变', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(2, -350, 'THD>5%', fontsize=9, color=anomaly_color, fontweight='bold')

    # (f) 三相不平衡
    ax = axes[1, 2]
    va, vb, vc = generate_three_phase_unbalance(t)
    ax.plot(t * 1000, va, color=phase_colors[0], linewidth=1.2, label='A相 (+5%)')
    ax.plot(t * 1000, vb, color=phase_colors[1], linewidth=1.2, label='B相 (-8%)')
    ax.plot(t * 1000, vc, color=phase_colors[2], linewidth=1.2, label='C相')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('(f) 三相不平衡', fontweight='bold')
    ax.set_xlim([0, 80])
    ax.set_ylim([-400, 400])
    ax.legend(loc='upper right', framealpha=0.9, ncol=1)
    ax.text(2, -350, '不平衡度>2%', fontsize=9, color=anomaly_color, fontweight='bold',
            fontproperties=font_zh_small)

    # 保存图片
    output_dir = get_output_dir(2)
    output_path = os.path.join(output_dir, 'fig_2_2_voltage_anomaly_types.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图片已保存: {output_path}")


    plt.close()


if __name__ == '__main__':
    main()
