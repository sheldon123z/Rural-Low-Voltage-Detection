#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对性农村低电压数据集生成器

根据 TimesNet 变体模型的核心能力，设计针对性的测试数据集：

1. periodic_load   - 周期性负荷数据集 (VoltageTimesNet)
   - 明显的日负荷周期、15分钟结算周期
   - 测试预设周期融合机制

2. three_phase     - 三相不平衡数据集 (TPATimesNet)
   - 单相过载、相间不平衡、缺相故障
   - 测试三相注意力机制

3. multi_scale     - 多尺度复合异常数据集 (MTSTimesNet)
   - 短期骤降 + 中期波动 + 长期趋势同时存在
   - 测试多尺度时序建模能力

4. hybrid_period   - 混合周期数据集 (HybridTimesNet)
   - 已知周期 + 未知周期 + 高噪声
   - 测试置信度融合机制

5. comprehensive   - 综合评估数据集
   - 融合所有场景，用于公平对比

作者: 农村低电压异常检测研究
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse


class RuralVoltageDatasetGenerator:
    """农村低电压数据集生成器基类"""

    # 标准电压参数 (220V 单相系统)
    NOMINAL_VOLTAGE = 220.0
    VOLTAGE_STD = 5.0  # 正常波动标准差

    # 国标阈值 (GB/T 12325-2008)
    UNDERVOLTAGE_THRESHOLD = 198.0  # -10%
    OVERVOLTAGE_THRESHOLD = 235.4  # +7%

    # 三相系统参数
    PHASE_SHIFT = 120.0  # 三相相位差 (度)
    UNBALANCE_THRESHOLD = 0.02  # 2% 不平衡度阈值

    def __init__(self, seed: int = 42):
        """初始化生成器"""
        np.random.seed(seed)
        self.seed = seed

    def generate_base_voltage(
        self, n_samples: int, sampling_rate: float = 1.0
    ) -> np.ndarray:
        """
        生成基础三相电压信号

        Args:
            n_samples: 样本数量
            sampling_rate: 采样率 (Hz)

        Returns:
            shape (n_samples, 3) 的三相电压数组 [Va, Vb, Vc]
        """
        t = np.arange(n_samples) / sampling_rate

        # 基础正弦波 (50Hz 工频，但这里用于表示幅值变化趋势)
        # 实际数据是 RMS 值，不是瞬时值
        Va = self.NOMINAL_VOLTAGE + np.random.normal(0, self.VOLTAGE_STD, n_samples)
        Vb = self.NOMINAL_VOLTAGE + np.random.normal(0, self.VOLTAGE_STD, n_samples)
        Vc = self.NOMINAL_VOLTAGE + np.random.normal(0, self.VOLTAGE_STD, n_samples)

        return np.column_stack([Va, Vb, Vc])

    def add_daily_pattern(
        self, voltage: np.ndarray, sampling_rate: float = 1.0
    ) -> np.ndarray:
        """添加日负荷周期模式"""
        n_samples = len(voltage)
        t = np.arange(n_samples) / sampling_rate

        # 日周期 (86400秒)
        daily_pattern = 5.0 * np.sin(2 * np.pi * t / 86400 - np.pi / 2)

        # 早晚高峰叠加
        morning_peak = 3.0 * np.exp(-((t % 86400 - 7 * 3600) ** 2) / (2 * 3600**2))
        evening_peak = 4.0 * np.exp(-((t % 86400 - 19 * 3600) ** 2) / (2 * 3600**2))

        pattern = daily_pattern + morning_peak + evening_peak

        for i in range(3):
            voltage[:, i] += pattern

        return voltage

    def calculate_features(self, voltage: np.ndarray) -> np.ndarray:
        """
        计算完整的 17 维特征

        特征列表:
        0-2: Va, Vb, Vc (三相电压)
        3-5: Ia, Ib, Ic (三相电流，模拟)
        6-9: P, Q, S, PF (功率指标)
        10-12: THD_Va, THD_Vb, THD_Vc (谐波失真率)
        13-14: V_unbalance, I_unbalance (不平衡因子)
        15: Freq (频率)
        16: timestamp (时间特征)
        """
        n_samples = len(voltage)
        features = np.zeros((n_samples, 17))

        # 三相电压
        features[:, 0:3] = voltage

        # 模拟三相电流 (基于电压和随机负载)
        base_current = 15.0
        for i in range(3):
            load_factor = 0.8 + 0.4 * np.random.random(n_samples)
            features[:, 3 + i] = base_current * load_factor * (voltage[:, i] / 220.0)

        # 功率计算
        for i in range(n_samples):
            Va, Vb, Vc = features[i, 0:3]
            Ia, Ib, Ic = features[i, 3:6]

            # 视在功率
            S = (Va * Ia + Vb * Ib + Vc * Ic) / 1000  # kVA
            # 功率因数 (随机模拟)
            PF = 0.85 + 0.1 * np.random.random()
            # 有功功率
            P = S * PF
            # 无功功率
            Q = S * np.sqrt(1 - PF**2)

            features[i, 6] = P
            features[i, 7] = Q
            features[i, 8] = S
            features[i, 9] = PF

        # 谐波失真率 (正常 2-4%)
        features[:, 10:13] = 0.02 + 0.02 * np.random.random((n_samples, 3))

        # 电压不平衡因子
        V_avg = np.mean(voltage, axis=1)
        V_max_dev = np.max(np.abs(voltage - V_avg[:, np.newaxis]), axis=1)
        features[:, 13] = V_max_dev / (V_avg + 1e-6)

        # 电流不平衡因子
        I = features[:, 3:6]
        I_avg = np.mean(I, axis=1)
        I_max_dev = np.max(np.abs(I - I_avg[:, np.newaxis]), axis=1)
        features[:, 14] = I_max_dev / (I_avg + 1e-6)

        # 频率 (50Hz 附近波动)
        features[:, 15] = 50.0 + 0.05 * np.random.randn(n_samples)

        # 时间特征 (归一化到 0-1)
        features[:, 16] = np.linspace(0, 1, n_samples)

        return features

    def save_dataset(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        output_dir: str,
        dataset_name: str,
    ):
        """保存数据集"""
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 列名
        columns = [
            "Va",
            "Vb",
            "Vc",
            "Ia",
            "Ib",
            "Ic",
            "P",
            "Q",
            "S",
            "PF",
            "THD_Va",
            "THD_Vb",
            "THD_Vc",
            "V_unbalance",
            "I_unbalance",
            "Freq",
            "timestamp",
        ]

        # 保存训练集
        train_df = pd.DataFrame(train_data, columns=columns)
        train_df.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)

        # 保存测试集
        test_df = pd.DataFrame(test_data, columns=columns)
        test_df.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)

        # 保存标签 (转换为二分类: 0=正常, 1=异常)
        binary_labels = (test_labels > 0).astype(int)
        label_df = pd.DataFrame(binary_labels, columns=["label"])
        label_df.to_csv(os.path.join(dataset_dir, "test_label.csv"), index=False)

        print(f"数据集 {dataset_name} 已保存到 {dataset_dir}")
        print(f"  训练集: {len(train_data)} 样本")
        print(f"  测试集: {len(test_data)} 样本")
        print(f"  异常比例: {np.mean(test_labels > 0) * 100:.2f}%")


class PeriodicLoadDataset(RuralVoltageDatasetGenerator):
    """
    周期性负荷数据集 - 针对 VoltageTimesNet

    特点:
    - 明显的 15分钟/1小时/日 周期性负荷波动
    - 异常: 周期性欠压 (每天固定时段)
    - 测试预设周期融合机制
    """

    def generate(
        self, train_samples: int = 10000, test_samples: int = 3000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成周期性负荷数据集"""
        print("\n" + "=" * 50)
        print("生成周期性负荷数据集 (periodic_load)")
        print("目标模型: VoltageTimesNet")
        print("=" * 50)

        # === 训练集 (正常数据) ===
        train_voltage = self.generate_base_voltage(train_samples)
        train_voltage = self.add_daily_pattern(train_voltage)

        # 添加 15 分钟周期波动 (900秒)
        t = np.arange(train_samples)
        pattern_15min = 3.0 * np.sin(2 * np.pi * t / 900)
        for i in range(3):
            train_voltage[:, i] += pattern_15min

        # 添加 1 小时周期波动 (3600秒)
        pattern_1h = 2.0 * np.sin(2 * np.pi * t / 3600)
        for i in range(3):
            train_voltage[:, i] += pattern_1h

        train_features = self.calculate_features(train_voltage)

        # === 测试集 (含异常) ===
        test_voltage = self.generate_base_voltage(test_samples)
        test_voltage = self.add_daily_pattern(test_voltage)

        t = np.arange(test_samples)
        pattern_15min = 3.0 * np.sin(2 * np.pi * t / 900)
        pattern_1h = 2.0 * np.sin(2 * np.pi * t / 3600)
        for i in range(3):
            test_voltage[:, i] += pattern_15min + pattern_1h

        test_labels = np.zeros(test_samples, dtype=int)

        # 异常1: 周期性欠压 (每隔一定间隔出现)
        # 模拟每天 18:00-21:00 用电高峰期欠压
        anomaly_period = 3600  # 每小时检查一次
        anomaly_duration = 300  # 异常持续 5 分钟

        for start in range(0, test_samples, anomaly_period):
            if np.random.random() < 0.3:  # 30% 概率发生
                end = min(start + anomaly_duration, test_samples)
                # 欠压 (降低 15-25%)
                drop_factor = 0.75 + 0.1 * np.random.random()
                test_voltage[start:end, :] *= drop_factor
                test_labels[start:end] = 1  # 欠压异常

        # 异常2: 过压 (较少发生)
        for start in range(500, test_samples, anomaly_period * 2):
            if np.random.random() < 0.15:
                end = min(start + anomaly_duration // 2, test_samples)
                rise_factor = 1.08 + 0.04 * np.random.random()
                test_voltage[start:end, :] *= rise_factor
                test_labels[start:end] = 2  # 过压异常

        test_features = self.calculate_features(test_voltage)

        return train_features, test_features, test_labels


class ThreePhaseDataset(RuralVoltageDatasetGenerator):
    """
    三相不平衡数据集 - 针对 TPATimesNet

    特点:
    - 单相负荷不均衡导致的三相不平衡
    - 异常: 单相过载、相间不平衡、缺相故障
    - 测试三相注意力机制
    """

    def generate(
        self, train_samples: int = 10000, test_samples: int = 3000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成三相不平衡数据集"""
        print("\n" + "=" * 50)
        print("生成三相不平衡数据集 (three_phase)")
        print("目标模型: TPATimesNet")
        print("=" * 50)

        # === 训练集 (基本平衡) ===
        train_voltage = self.generate_base_voltage(train_samples)
        # 添加轻微的自然不平衡 (< 2%)
        train_voltage[:, 0] *= 1.0 + 0.01 * np.random.randn(train_samples)
        train_voltage[:, 1] *= 1.0 + 0.01 * np.random.randn(train_samples)
        train_voltage[:, 2] *= 1.0 + 0.01 * np.random.randn(train_samples)

        train_features = self.calculate_features(train_voltage)

        # === 测试集 (含三相不平衡异常) ===
        test_voltage = self.generate_base_voltage(test_samples)
        test_labels = np.zeros(test_samples, dtype=int)

        # 异常类型定义
        anomaly_types = [
            ("A相低压", 0, 0.82, 1.0, 1.0),  # Va 降低
            ("B相低压", 1, 1.0, 0.82, 1.0),  # Vb 降低
            ("C相低压", 2, 1.0, 1.0, 0.82),  # Vc 降低
            ("A相高压", 3, 1.12, 1.0, 1.0),  # Va 升高
            ("AB相不平衡", 4, 0.88, 1.08, 1.0),  # Va低 Vb高
            ("BC相不平衡", 5, 1.0, 0.88, 1.08),  # Vb低 Vc高
            ("严重不平衡", 6, 0.75, 1.15, 0.95),  # 三相都偏离
        ]

        anomaly_interval = 400
        anomaly_duration = 150

        idx = 0
        for start in range(0, test_samples - anomaly_duration, anomaly_interval):
            if np.random.random() < 0.6:  # 60% 概率发生异常
                anomaly = anomaly_types[idx % len(anomaly_types)]
                name, label, fa, fb, fc = anomaly

                end = start + anomaly_duration

                # 添加随机变化
                noise = 0.03 * np.random.randn()
                test_voltage[start:end, 0] *= fa + noise
                test_voltage[start:end, 1] *= fb + noise
                test_voltage[start:end, 2] *= fc + noise

                test_labels[start:end] = label + 1

                idx += 1

        # 添加缺相故障 (严重异常，少量)
        for _ in range(3):
            start = np.random.randint(0, test_samples - 100)
            end = start + 50
            phase = np.random.randint(0, 3)
            test_voltage[start:end, phase] *= 0.1  # 几乎为零
            test_labels[start:end] = 8  # 缺相异常

        test_features = self.calculate_features(test_voltage)

        return train_features, test_features, test_labels


class MultiScaleDataset(RuralVoltageDatasetGenerator):
    """
    多尺度复合异常数据集 - 针对 MTSTimesNet

    特点:
    - 短期 (2-20样本): 电压骤降、尖峰
    - 中期 (20-60样本): 负荷突变、阶跃
    - 长期 (60-200样本): 渐变趋势、漂移
    - 测试多尺度时序建模能力
    """

    def generate(
        self, train_samples: int = 15000, test_samples: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成多尺度复合异常数据集"""
        print("\n" + "=" * 50)
        print("生成多尺度复合异常数据集 (multi_scale)")
        print("目标模型: MTSTimesNet")
        print("=" * 50)

        # === 训练集 ===
        train_voltage = self.generate_base_voltage(train_samples)
        train_voltage = self.add_daily_pattern(train_voltage)
        train_features = self.calculate_features(train_voltage)

        # === 测试集 ===
        test_voltage = self.generate_base_voltage(test_samples)
        test_voltage = self.add_daily_pattern(test_voltage)
        test_labels = np.zeros(test_samples, dtype=int)

        # === 短期异常 (2-20 样本) ===
        print("  注入短期异常...")
        n_short = 30
        for _ in range(n_short):
            start = np.random.randint(0, test_samples - 20)
            duration = np.random.randint(2, 20)
            end = start + duration

            anomaly_type = np.random.choice(["spike", "sag", "impulse"])

            if anomaly_type == "spike":
                # 电压尖峰
                spike = 20 * np.random.randn(duration, 3)
                test_voltage[start:end, :] += spike
                test_labels[start:end] = 1
            elif anomaly_type == "sag":
                # 短暂骤降
                test_voltage[start:end, :] *= 0.7 + 0.1 * np.random.random()
                test_labels[start:end] = 2
            else:
                # 脉冲干扰
                impulse_point = start + duration // 2
                test_voltage[impulse_point, :] *= 1.5
                test_labels[impulse_point] = 3

        # === 中期异常 (20-60 样本) ===
        print("  注入中期异常...")
        n_medium = 20
        for _ in range(n_medium):
            start = np.random.randint(0, test_samples - 60)
            duration = np.random.randint(20, 60)
            end = start + duration

            anomaly_type = np.random.choice(["step", "ramp", "oscillation"])

            if anomaly_type == "step":
                # 阶跃变化
                step_value = -15 if np.random.random() < 0.6 else 10
                test_voltage[start:end, :] += step_value
                test_labels[start:end] = 4
            elif anomaly_type == "ramp":
                # 斜坡变化
                ramp = np.linspace(0, -20, duration)[:, np.newaxis]
                test_voltage[start:end, :] += ramp
                test_labels[start:end] = 5
            else:
                # 振荡
                t = np.arange(duration)
                osc = 8 * np.sin(2 * np.pi * t / 10)[:, np.newaxis]
                test_voltage[start:end, :] += osc
                test_labels[start:end] = 6

        # === 长期异常 (60-200 样本) ===
        print("  注入长期异常...")
        n_long = 10
        for _ in range(n_long):
            start = np.random.randint(0, test_samples - 200)
            duration = np.random.randint(60, 200)
            end = start + duration

            anomaly_type = np.random.choice(["drift", "trend", "seasonal_shift"])

            if anomaly_type == "drift":
                # 渐变漂移
                drift = np.linspace(0, -25, duration)[:, np.newaxis]
                test_voltage[start:end, :] += drift
                test_labels[start:end] = 7
            elif anomaly_type == "trend":
                # 持续上升/下降趋势
                trend_slope = -0.1 if np.random.random() < 0.7 else 0.08
                trend = trend_slope * np.arange(duration)[:, np.newaxis]
                test_voltage[start:end, :] += trend
                test_labels[start:end] = 8
            else:
                # 季节性偏移
                test_voltage[start:end, :] -= 12
                test_labels[start:end] = 9

        # === 复合异常 (多尺度叠加) ===
        print("  注入复合异常...")
        n_compound = 5
        for _ in range(n_compound):
            start = np.random.randint(0, test_samples - 150)

            # 长期趋势
            long_dur = 150
            drift = np.linspace(0, -15, long_dur)[:, np.newaxis]
            test_voltage[start : start + long_dur, :] += drift

            # 叠加中期阶跃
            mid_start = start + 50
            mid_dur = 40
            test_voltage[mid_start : mid_start + mid_dur, :] -= 8

            # 叠加短期尖峰
            short_start = start + 80
            test_voltage[short_start : short_start + 5, :] += 15 * np.random.randn(5, 3)

            test_labels[start : start + long_dur] = 10  # 复合异常标签

        test_features = self.calculate_features(test_voltage)

        return train_features, test_features, test_labels


class HybridPeriodDataset(RuralVoltageDatasetGenerator):
    """
    混合周期数据集 - 针对 HybridTimesNet

    特点:
    - 已知电网周期 (15min, 1h) + 未知周期 (光伏波动、谐波)
    - 高噪声环境
    - 测试置信度融合机制
    """

    def generate(
        self, train_samples: int = 12000, test_samples: int = 4000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成混合周期数据集"""
        print("\n" + "=" * 50)
        print("生成混合周期数据集 (hybrid_period)")
        print("目标模型: HybridTimesNet")
        print("=" * 50)

        # === 训练集 ===
        train_voltage = self.generate_base_voltage(train_samples)

        t = np.arange(train_samples)

        # 已知周期: 15分钟, 1小时
        pattern_15min = 3.0 * np.sin(2 * np.pi * t / 900)
        pattern_1h = 2.0 * np.sin(2 * np.pi * t / 3600)

        # 未知周期: 光伏波动 (云遮挡，约 5-10 分钟不规则周期)
        # 使用非整数周期模拟
        pattern_pv = 4.0 * np.sin(2 * np.pi * t / 420)  # ~7分钟周期
        pattern_pv += 2.0 * np.sin(2 * np.pi * t / 180)  # ~3分钟周期

        for i in range(3):
            train_voltage[:, i] += pattern_15min + pattern_1h + pattern_pv

        # 添加较高噪声 (模拟工业环境)
        noise = 3.0 * np.random.randn(train_samples, 3)
        train_voltage += noise

        train_features = self.calculate_features(train_voltage)

        # === 测试集 ===
        test_voltage = self.generate_base_voltage(test_samples)

        t = np.arange(test_samples)
        pattern_15min = 3.0 * np.sin(2 * np.pi * t / 900)
        pattern_1h = 2.0 * np.sin(2 * np.pi * t / 3600)
        pattern_pv = 4.0 * np.sin(2 * np.pi * t / 420)
        pattern_pv += 2.0 * np.sin(2 * np.pi * t / 180)

        for i in range(3):
            test_voltage[:, i] += pattern_15min + pattern_1h + pattern_pv

        # 高噪声
        noise = 4.0 * np.random.randn(test_samples, 3)
        test_voltage += noise

        test_labels = np.zeros(test_samples, dtype=int)

        # === 异常注入 ===

        # 1. 已知周期相关异常 (15分钟周期欠压)
        print("  注入已知周期异常...")
        for start in range(0, test_samples, 900):
            if np.random.random() < 0.25:
                end = min(start + 200, test_samples)
                test_voltage[start:end, :] *= 0.85
                test_labels[start:end] = 1

        # 2. 未知周期异常 (新的谐波注入)
        print("  注入未知周期异常...")
        # 引入训练集中没有的新周期
        new_period = 250  # 新的未知周期
        n_injections = 8
        for _ in range(n_injections):
            start = np.random.randint(0, test_samples - 500)
            duration = 400

            t_local = np.arange(duration)
            new_pattern = 10 * np.sin(2 * np.pi * t_local / new_period)
            test_voltage[start : start + duration, :] += new_pattern[:, np.newaxis]
            test_labels[start : start + duration] = 2

        # 3. 周期干涉异常 (多周期叠加导致的极端值)
        print("  注入周期干涉异常...")
        for _ in range(6):
            start = np.random.randint(0, test_samples - 100)
            duration = 80

            # 模拟多个周期同相位叠加
            t_local = np.arange(duration)
            interference = (
                5 * np.sin(2 * np.pi * t_local / 20)
                + 5 * np.sin(2 * np.pi * t_local / 25)
                + 5 * np.sin(2 * np.pi * t_local / 30)
            )
            test_voltage[start : start + duration, :] += interference[:, np.newaxis]
            test_labels[start : start + duration] = 3

        # 4. 高噪声区段 (测试鲁棒性)
        print("  注入高噪声区段...")
        for _ in range(5):
            start = np.random.randint(0, test_samples - 200)
            duration = 150

            extreme_noise = 8.0 * np.random.randn(duration, 3)
            test_voltage[start : start + duration, :] += extreme_noise

            # 同时叠加欠压
            test_voltage[start : start + duration, :] *= 0.9
            test_labels[start : start + duration] = 4

        test_features = self.calculate_features(test_voltage)

        return train_features, test_features, test_labels


class ComprehensiveDataset(RuralVoltageDatasetGenerator):
    """
    综合评估数据集 - 融合所有场景

    用于公平对比所有模型的综合能力
    """

    def generate(
        self, train_samples: int = 20000, test_samples: int = 6000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成综合评估数据集"""
        print("\n" + "=" * 50)
        print("生成综合评估数据集 (comprehensive)")
        print("目标: 公平对比所有模型")
        print("=" * 50)

        # === 训练集 ===
        train_voltage = self.generate_base_voltage(train_samples)
        train_voltage = self.add_daily_pattern(train_voltage)

        t = np.arange(train_samples)
        # 多种周期叠加
        pattern_15min = 2.0 * np.sin(2 * np.pi * t / 900)
        pattern_1h = 1.5 * np.sin(2 * np.pi * t / 3600)
        pattern_pv = 2.0 * np.sin(2 * np.pi * t / 420)

        for i in range(3):
            train_voltage[:, i] += pattern_15min + pattern_1h + pattern_pv

        # 轻微三相不平衡
        train_voltage[:, 0] *= 1.0 + 0.01 * np.random.randn(train_samples)
        train_voltage[:, 1] *= 1.0 + 0.01 * np.random.randn(train_samples)

        # 中等噪声
        noise = 2.0 * np.random.randn(train_samples, 3)
        train_voltage += noise

        train_features = self.calculate_features(train_voltage)

        # === 测试集 ===
        test_voltage = self.generate_base_voltage(test_samples)
        test_voltage = self.add_daily_pattern(test_voltage)

        t = np.arange(test_samples)
        pattern_15min = 2.0 * np.sin(2 * np.pi * t / 900)
        pattern_1h = 1.5 * np.sin(2 * np.pi * t / 3600)
        pattern_pv = 2.0 * np.sin(2 * np.pi * t / 420)

        for i in range(3):
            test_voltage[:, i] += pattern_15min + pattern_1h + pattern_pv

        noise = 2.0 * np.random.randn(test_samples, 3)
        test_voltage += noise

        test_labels = np.zeros(test_samples, dtype=int)

        # === 综合异常注入 ===

        # 1. 周期性欠压 (20%)
        print("  注入周期性欠压异常...")
        for start in range(0, test_samples, 600):
            if np.random.random() < 0.2:
                end = min(start + 150, test_samples)
                test_voltage[start:end, :] *= 0.85
                test_labels[start:end] = 1

        # 2. 三相不平衡 (25%)
        print("  注入三相不平衡异常...")
        anomaly_configs = [
            (0.85, 1.0, 1.0),
            (1.0, 0.85, 1.0),
            (1.0, 1.0, 0.85),
            (0.88, 1.08, 1.0),
            (1.0, 0.88, 1.08),
        ]
        for start in range(100, test_samples, 500):
            if np.random.random() < 0.25:
                end = min(start + 120, test_samples)
                config = anomaly_configs[np.random.randint(len(anomaly_configs))]
                test_voltage[start:end, 0] *= config[0]
                test_voltage[start:end, 1] *= config[1]
                test_voltage[start:end, 2] *= config[2]
                test_labels[start:end] = 2

        # 3. 短期骤降 (15%)
        print("  注入短期骤降异常...")
        for _ in range(25):
            start = np.random.randint(0, test_samples - 20)
            duration = np.random.randint(5, 20)
            test_voltage[start : start + duration, :] *= 0.7
            test_labels[start : start + duration] = 3

        # 4. 长期趋势 (15%)
        print("  注入长期趋势异常...")
        for _ in range(8):
            start = np.random.randint(0, test_samples - 200)
            duration = np.random.randint(100, 200)
            drift = np.linspace(0, -20, duration)[:, np.newaxis]
            test_voltage[start : start + duration, :] += drift
            test_labels[start : start + duration] = 4

        # 5. 谐波畸变 (10%)
        print("  注入谐波畸变异常...")
        for _ in range(10):
            start = np.random.randint(0, test_samples - 100)
            duration = 80
            t_local = np.arange(duration)
            # 高次谐波
            harmonic = 8 * np.sin(2 * np.pi * t_local * 5 / 50)  # 5次谐波
            harmonic += 4 * np.sin(2 * np.pi * t_local * 7 / 50)  # 7次谐波
            test_voltage[start : start + duration, :] += harmonic[:, np.newaxis]
            test_labels[start : start + duration] = 5

        # 6. 复合异常 (15%)
        print("  注入复合异常...")
        for _ in range(5):
            start = np.random.randint(0, test_samples - 180)

            # 长期趋势 + 三相不平衡 + 短期骤降
            drift = np.linspace(0, -10, 180)[:, np.newaxis]
            test_voltage[start : start + 180, :] += drift
            test_voltage[start : start + 180, 0] *= 0.95

            # 中间插入短期骤降
            test_voltage[start + 80 : start + 90, :] *= 0.75

            test_labels[start : start + 180] = 6

        test_features = self.calculate_features(test_voltage)

        return train_features, test_features, test_labels


def generate_all_datasets(output_dir: str, seed: int = 42):
    """生成所有针对性数据集"""
    print("\n" + "=" * 60)
    print("农村低电压针对性数据集生成器")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"随机种子: {seed}")
    print("=" * 60)

    datasets = {
        "periodic_load": (
            PeriodicLoadDataset,
            {"train_samples": 10000, "test_samples": 3000},
        ),
        "three_phase": (
            ThreePhaseDataset,
            {"train_samples": 10000, "test_samples": 3000},
        ),
        "multi_scale": (
            MultiScaleDataset,
            {"train_samples": 15000, "test_samples": 5000},
        ),
        "hybrid_period": (
            HybridPeriodDataset,
            {"train_samples": 12000, "test_samples": 4000},
        ),
        "comprehensive": (
            ComprehensiveDataset,
            {"train_samples": 20000, "test_samples": 6000},
        ),
    }

    results = {}

    for name, (GeneratorClass, params) in datasets.items():
        generator = GeneratorClass(seed=seed)
        train_data, test_data, test_labels = generator.generate(**params)

        generator.save_dataset(train_data, test_data, test_labels, output_dir, name)

        results[name] = {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "anomaly_ratio": float(np.mean(test_labels > 0)),
            "anomaly_types": int(np.max(test_labels)),
        }

    # 保存数据集信息
    info_path = os.path.join(output_dir, "datasets_info.json")
    import json

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "随机种子": seed,
                "数据集信息": results,
                "目标模型对应关系": {
                    "periodic_load": "VoltageTimesNet (预设周期融合)",
                    "three_phase": "TPATimesNet (三相注意力)",
                    "multi_scale": "MTSTimesNet (多尺度时序)",
                    "hybrid_period": "HybridTimesNet (混合周期发现)",
                    "comprehensive": "所有模型 (综合评估)",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("所有数据集生成完成!")
    print(f"数据集信息已保存到: {info_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成针对性农村低电压数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="输出目录 (默认: ./)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "periodic_load",
            "three_phase",
            "multi_scale",
            "hybrid_period",
            "comprehensive",
        ],
        help="生成指定数据集 (默认: all)",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        generate_all_datasets(args.output_dir, args.seed)
    else:
        # 生成单个数据集
        dataset_map = {
            "periodic_load": (
                PeriodicLoadDataset,
                {"train_samples": 10000, "test_samples": 3000},
            ),
            "three_phase": (
                ThreePhaseDataset,
                {"train_samples": 10000, "test_samples": 3000},
            ),
            "multi_scale": (
                MultiScaleDataset,
                {"train_samples": 15000, "test_samples": 5000},
            ),
            "hybrid_period": (
                HybridPeriodDataset,
                {"train_samples": 12000, "test_samples": 4000},
            ),
            "comprehensive": (
                ComprehensiveDataset,
                {"train_samples": 20000, "test_samples": 6000},
            ),
        }

        GeneratorClass, params = dataset_map[args.dataset]
        generator = GeneratorClass(seed=args.seed)
        train_data, test_data, test_labels = generator.generate(**params)
        generator.save_dataset(
            train_data, test_data, test_labels, args.output_dir, args.dataset
        )
