"""
Sample Data Generator for Rural Voltage Anomaly Detection

This script generates synthetic voltage data for testing the anomaly detection pipeline.
The generated data includes:
- Normal operation patterns with daily load variations
- Various anomaly types: undervoltage, overvoltage, voltage sags, harmonics, unbalance

Usage:
    python generate_sample_data.py --train_samples 10000 --test_samples 2000 --anomaly_ratio 0.1
"""

import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta


def generate_daily_load_pattern(n_samples, base_load=1.0):
    """Generate daily load pattern (24-hour cycle)."""
    t = np.linspace(0, 2 * np.pi * (n_samples / 86400), n_samples)

    # Morning peak (7-9 AM), evening peak (6-9 PM), night low
    pattern = base_load * (
        1.0 +
        0.2 * np.sin(t - np.pi/2) +  # Daily cycle
        0.1 * np.sin(2*t) +  # Half-day variation
        0.05 * np.sin(3*t)   # 8-hour variation
    )
    return pattern


def generate_normal_voltage(n_samples, nominal=220.0, noise_std=2.0):
    """Generate normal three-phase voltage with small random variations."""
    # Daily load pattern affects voltage
    load_pattern = generate_daily_load_pattern(n_samples)

    # Base voltages with 120-degree phase relationships (in magnitude space)
    Va = nominal * load_pattern + np.random.normal(0, noise_std, n_samples)
    Vb = nominal * load_pattern + np.random.normal(0, noise_std, n_samples)
    Vc = nominal * load_pattern + np.random.normal(0, noise_std, n_samples)

    # Small natural unbalance (< 2%)
    unbalance = np.random.uniform(-0.02, 0.02, 3)
    Va *= (1 + unbalance[0])
    Vb *= (1 + unbalance[1])
    Vc *= (1 + unbalance[2])

    return Va, Vb, Vc


def generate_current(Va, Vb, Vc, power_factor=0.9, load_factor=1.0):
    """Generate current based on voltage and power factor."""
    # Simplified current calculation
    base_current = 15.0 * load_factor  # Base current in A

    Ia = base_current * (Va / 220.0) + np.random.normal(0, 0.5, len(Va))
    Ib = base_current * (Vb / 220.0) + np.random.normal(0, 0.5, len(Vb))
    Ic = base_current * (Vc / 220.0) + np.random.normal(0, 0.5, len(Vc))

    return Ia, Ib, Ic


def generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic):
    """Generate power metrics from voltage and current."""
    # Active power (kW)
    P = (Va * Ia + Vb * Ib + Vc * Ic) / 1000 * 0.9  # Assuming 0.9 PF

    # Reactive power (kVar)
    Q = P * np.tan(np.arccos(0.9)) + np.random.normal(0, 0.1, len(P))

    # Apparent power (kVA)
    S = np.sqrt(P**2 + Q**2)

    # Power factor
    PF = P / (S + 1e-8)
    PF = np.clip(PF, -1, 1)

    return P, Q, S, PF


def generate_quality_metrics(Va, Vb, Vc, add_harmonics=False):
    """Generate power quality metrics."""
    n_samples = len(Va)

    # THD (normally < 3%)
    if add_harmonics:
        THD_Va = np.random.uniform(5, 10, n_samples)  # Anomalous THD
        THD_Vb = np.random.uniform(5, 10, n_samples)
        THD_Vc = np.random.uniform(5, 10, n_samples)
    else:
        THD_Va = np.random.uniform(1, 3, n_samples)
        THD_Vb = np.random.uniform(1, 3, n_samples)
        THD_Vc = np.random.uniform(1, 3, n_samples)

    # Frequency (nominally 50 Hz, small variations)
    Freq = 50.0 + np.random.normal(0, 0.05, n_samples)

    # Voltage unbalance factor
    V_mean = (Va + Vb + Vc) / 3
    V_max_dev = np.maximum(np.abs(Va - V_mean),
                          np.maximum(np.abs(Vb - V_mean), np.abs(Vc - V_mean)))
    V_unbalance = V_max_dev / (V_mean + 1e-8) * 100

    # Current unbalance
    I_unbalance = np.random.uniform(0, 3, n_samples)

    return THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance


def inject_undervoltage(Va, Vb, Vc, start, duration, severity=0.15):
    """Inject undervoltage anomaly."""
    end = min(start + duration, len(Va))
    factor = 1 - severity + np.random.uniform(-0.02, 0.02)
    Va[start:end] *= factor
    Vb[start:end] *= factor
    Vc[start:end] *= factor
    return Va, Vb, Vc


def inject_overvoltage(Va, Vb, Vc, start, duration, severity=0.12):
    """Inject overvoltage anomaly."""
    end = min(start + duration, len(Va))
    factor = 1 + severity + np.random.uniform(-0.02, 0.02)
    Va[start:end] *= factor
    Vb[start:end] *= factor
    Vc[start:end] *= factor
    return Va, Vb, Vc


def inject_voltage_sag(Va, Vb, Vc, start, duration, depth=0.3):
    """Inject voltage sag (temporary voltage drop)."""
    end = min(start + duration, len(Va))
    # Voltage sag with ramp up/down
    ramp_len = min(5, duration // 4)

    for i in range(start, end):
        if i < start + ramp_len:
            factor = 1 - depth * (i - start) / ramp_len
        elif i > end - ramp_len:
            factor = 1 - depth * (end - i) / ramp_len
        else:
            factor = 1 - depth

        Va[i] *= factor
        Vb[i] *= factor
        Vc[i] *= factor

    return Va, Vb, Vc


def inject_unbalance(Va, Vb, Vc, start, duration, unbalance_factor=0.08):
    """Inject three-phase unbalance."""
    end = min(start + duration, len(Va))
    # Reduce one phase significantly
    phase = np.random.randint(0, 3)
    if phase == 0:
        Va[start:end] *= (1 - unbalance_factor)
    elif phase == 1:
        Vb[start:end] *= (1 - unbalance_factor)
    else:
        Vc[start:end] *= (1 - unbalance_factor)
    return Va, Vb, Vc


def generate_dataset(n_samples, anomaly_ratio=0.0, seed=42):
    """Generate complete dataset with optional anomalies."""
    np.random.seed(seed)

    # Generate normal data
    Va, Vb, Vc = generate_normal_voltage(n_samples)
    Ia, Ib, Ic = generate_current(Va, Vb, Vc)
    P, Q, S, PF = generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic)
    THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance = generate_quality_metrics(Va, Vb, Vc)

    # Initialize labels (0 = normal)
    labels = np.zeros(n_samples, dtype=int)

    # Inject anomalies
    if anomaly_ratio > 0:
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_types = [1, 2, 3, 4, 5]  # undervoltage, overvoltage, sag, harmonic, unbalance

        # Distribute anomalies
        anomalies_per_type = n_anomalies // len(anomaly_types)
        remaining = n_anomalies - anomalies_per_type * len(anomaly_types)

        current_pos = int(n_samples * 0.1)  # Start after initial normal period

        for anomaly_type in anomaly_types:
            n_events = anomalies_per_type + (1 if remaining > 0 else 0)
            remaining -= 1

            for _ in range(max(1, n_events // 10)):  # Create several events
                if current_pos >= n_samples - 100:
                    break

                duration = np.random.randint(20, 100)  # 20-100 samples per event
                end = min(current_pos + duration, n_samples)

                if anomaly_type == 1:  # Undervoltage
                    Va, Vb, Vc = inject_undervoltage(Va, Vb, Vc, current_pos, duration)
                elif anomaly_type == 2:  # Overvoltage
                    Va, Vb, Vc = inject_overvoltage(Va, Vb, Vc, current_pos, duration)
                elif anomaly_type == 3:  # Voltage sag
                    Va, Vb, Vc = inject_voltage_sag(Va, Vb, Vc, current_pos, duration)
                elif anomaly_type == 4:  # Harmonic
                    THD_Va[current_pos:end] = np.random.uniform(6, 12, end - current_pos)
                    THD_Vb[current_pos:end] = np.random.uniform(6, 12, end - current_pos)
                    THD_Vc[current_pos:end] = np.random.uniform(6, 12, end - current_pos)
                elif anomaly_type == 5:  # Unbalance
                    Va, Vb, Vc = inject_unbalance(Va, Vb, Vc, current_pos, duration)

                labels[current_pos:end] = anomaly_type
                current_pos = end + np.random.randint(50, 200)  # Gap between anomalies

        # Recalculate dependent metrics after anomaly injection
        Ia, Ib, Ic = generate_current(Va, Vb, Vc)
        P, Q, S, PF = generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic)

        # Recalculate unbalance
        V_mean = (Va + Vb + Vc) / 3
        V_max_dev = np.maximum(np.abs(Va - V_mean),
                              np.maximum(np.abs(Vb - V_mean), np.abs(Vc - V_mean)))
        V_unbalance = V_max_dev / (V_mean + 1e-8) * 100

    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Va': Va,
        'Vb': Vb,
        'Vc': Vc,
        'Ia': Ia,
        'Ib': Ib,
        'Ic': Ic,
        'P': P,
        'Q': Q,
        'S': S,
        'PF': PF,
        'THD_Va': THD_Va,
        'THD_Vb': THD_Vb,
        'THD_Vc': THD_Vc,
        'Freq': Freq,
        'V_unbalance': V_unbalance,
        'I_unbalance': I_unbalance,
    })

    # Binary labels for anomaly detection (0: normal, 1: any anomaly)
    binary_labels = (labels > 0).astype(int)

    return df, labels, binary_labels


def main():
    parser = argparse.ArgumentParser(description='Generate sample voltage data')
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=2000,
                        help='Number of test samples')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1,
                        help='Ratio of anomalies in test data')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    print(f"Generating training data ({args.train_samples} samples, normal only)...")
    train_df, _, _ = generate_dataset(args.train_samples, anomaly_ratio=0.0, seed=42)
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    print(f"  Saved to train.csv")

    print(f"Generating test data ({args.test_samples} samples, {args.anomaly_ratio*100:.0f}% anomalies)...")
    test_df, labels, binary_labels = generate_dataset(
        args.test_samples, anomaly_ratio=args.anomaly_ratio, seed=123
    )
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    print(f"  Saved to test.csv")

    # Save labels
    label_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'label': binary_labels,
        'anomaly_type': labels
    })
    label_df.to_csv(os.path.join(args.output_dir, 'test_label.csv'), index=False)
    print(f"  Saved to test_label.csv")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Test anomaly ratio: {np.mean(binary_labels)*100:.2f}%")
    print(f"  Anomaly type distribution:")
    for i in range(6):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        names = ['Normal', 'Undervoltage', 'Overvoltage', 'Voltage_Sag', 'Harmonic', 'Unbalance']
        print(f"    {names[i]}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
