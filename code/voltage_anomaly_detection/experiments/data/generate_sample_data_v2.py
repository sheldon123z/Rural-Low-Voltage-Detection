"""
Enhanced Sample Data Generator for Rural Voltage Anomaly Detection V2

This module generates more realistic synthetic voltage data with:
1. Enhanced daily/weekly load patterns
2. More realistic anomaly injection
3. Feature engineering support
4. Configurable anomaly types and severity

Usage:
    python generate_sample_data_v2.py --train_samples 50000 --test_samples 10000 --anomaly_ratio 0.1
"""

import numpy as np
import pandas as pd
import argparse
import os
import json
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional


# Constants based on China GB/T 12325-2008
VOLTAGE_NOMINAL = 220.0
VOLTAGE_LOWER_LIMIT = 198.0  # -10%
VOLTAGE_UPPER_LIMIT = 242.0  # +10%
THD_LIMIT = 5.0  # %
UNBALANCE_LIMIT = 4.0  # %
FREQUENCY_NOMINAL = 50.0


class VoltagePatternGenerator:
    """Generate realistic voltage patterns with multiple time scales."""
    
    def __init__(self, sampling_rate: int = 1):
        """
        Args:
            sampling_rate: Samples per second (default 1Hz)
        """
        self.sampling_rate = sampling_rate
        
    def generate_daily_pattern(self, n_samples: int) -> np.ndarray:
        """Generate 24-hour load pattern."""
        t = np.arange(n_samples) / (self.sampling_rate * 3600)  # Convert to hours
        
        # Multi-component daily pattern
        # Morning peak (7-9 AM), noon dip, evening peak (6-9 PM), night low
        pattern = (
            1.0 +
            0.08 * np.sin(2 * np.pi * (t - 6) / 24) +  # Main daily cycle
            0.05 * np.sin(4 * np.pi * (t - 6) / 24) +  # Half-day variation
            0.03 * np.sin(6 * np.pi * t / 24) +        # 8-hour variation
            0.02 * np.sin(8 * np.pi * t / 24)          # 6-hour variation
        )
        return pattern
    
    def generate_weekly_pattern(self, n_samples: int) -> np.ndarray:
        """Generate weekly load pattern (weekend vs weekday)."""
        t = np.arange(n_samples) / (self.sampling_rate * 3600 * 24)  # Convert to days
        
        # Weekly pattern: lower load on weekends
        pattern = 1.0 - 0.05 * np.cos(2 * np.pi * t / 7)
        return pattern
    
    def generate_seasonal_pattern(self, n_samples: int, season: str = 'summer') -> np.ndarray:
        """Generate seasonal pattern (summer higher load due to AC)."""
        if season == 'summer':
            base = 1.05
        elif season == 'winter':
            base = 1.03
        else:
            base = 1.0
        return np.ones(n_samples) * base
    
    def generate_random_fluctuation(self, n_samples: int, std: float = 0.02) -> np.ndarray:
        """Generate random small fluctuations."""
        return np.random.normal(0, std, n_samples)


class ThreePhaseVoltageGenerator:
    """Generate three-phase voltage with realistic characteristics."""
    
    def __init__(self, nominal_voltage: float = VOLTAGE_NOMINAL):
        self.nominal = nominal_voltage
        self.pattern_gen = VoltagePatternGenerator()
        
    def generate_balanced_voltage(
        self, 
        n_samples: int,
        noise_std: float = 2.0,
        natural_unbalance: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate balanced three-phase voltage."""
        
        # Combine multiple patterns
        daily = self.pattern_gen.generate_daily_pattern(n_samples)
        weekly = self.pattern_gen.generate_weekly_pattern(n_samples)
        random = self.pattern_gen.generate_random_fluctuation(n_samples)
        
        combined_pattern = daily * weekly + random
        
        # Base voltages
        Va = self.nominal * combined_pattern + np.random.normal(0, noise_std, n_samples)
        Vb = self.nominal * combined_pattern + np.random.normal(0, noise_std, n_samples)
        Vc = self.nominal * combined_pattern + np.random.normal(0, noise_std, n_samples)
        
        # Add small natural unbalance (< 2%)
        unbalance = np.random.uniform(-natural_unbalance, natural_unbalance, 3)
        Va *= (1 + unbalance[0])
        Vb *= (1 + unbalance[1])
        Vc *= (1 + unbalance[2])
        
        return Va, Vb, Vc


class CurrentGenerator:
    """Generate current based on voltage and load characteristics."""
    
    def __init__(self, base_current: float = 15.0):
        self.base_current = base_current
        
    def generate_current(
        self, 
        Va: np.ndarray, 
        Vb: np.ndarray, 
        Vc: np.ndarray,
        power_factor: float = 0.9,
        load_variation: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate three-phase current."""
        n_samples = len(Va)
        
        # Load variation pattern
        load_factor = 1 + load_variation * np.sin(
            2 * np.pi * np.arange(n_samples) / (n_samples / 10)
        )
        
        Ia = self.base_current * load_factor * (Va / 220.0) + np.random.normal(0, 0.5, n_samples)
        Ib = self.base_current * load_factor * (Vb / 220.0) + np.random.normal(0, 0.5, n_samples)
        Ic = self.base_current * load_factor * (Vc / 220.0) + np.random.normal(0, 0.5, n_samples)
        
        return Ia, Ib, Ic


class PowerMetricsGenerator:
    """Generate power metrics from voltage and current."""
    
    @staticmethod
    def generate_power_metrics(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        Ia: np.ndarray, Ib: np.ndarray, Ic: np.ndarray,
        power_factor: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate P, Q, S, PF."""
        
        # Active power (kW)
        P = (Va * Ia + Vb * Ib + Vc * Ic) / 1000 * power_factor
        
        # Reactive power (kVar)
        Q = P * np.tan(np.arccos(power_factor)) + np.random.normal(0, 0.1, len(P))
        
        # Apparent power (kVA)
        S = np.sqrt(P**2 + Q**2)
        
        # Power factor with small variations
        PF = power_factor + np.random.normal(0, 0.02, len(P))
        PF = np.clip(PF, 0.7, 1.0)
        
        return P, Q, S, PF


class PowerQualityMetricsGenerator:
    """Generate power quality metrics."""
    
    @staticmethod
    def generate_quality_metrics(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        anomalous_thd: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate THD, Frequency, and Unbalance factors."""
        n_samples = len(Va)
        
        # THD (normally < 3%, anomalous > 5%)
        if anomalous_thd:
            THD_Va = np.random.uniform(5, 12, n_samples)
            THD_Vb = np.random.uniform(5, 12, n_samples)
            THD_Vc = np.random.uniform(5, 12, n_samples)
        else:
            THD_Va = np.random.uniform(1, 3, n_samples)
            THD_Vb = np.random.uniform(1, 3, n_samples)
            THD_Vc = np.random.uniform(1, 3, n_samples)
        
        # Frequency (50 Hz with small variations)
        Freq = FREQUENCY_NOMINAL + np.random.normal(0, 0.03, n_samples)
        
        # Voltage unbalance factor
        V_mean = (Va + Vb + Vc) / 3
        V_max_dev = np.maximum(
            np.abs(Va - V_mean),
            np.maximum(np.abs(Vb - V_mean), np.abs(Vc - V_mean))
        )
        V_unbalance = V_max_dev / (V_mean + 1e-8) * 100
        
        # Current unbalance (random for normal)
        I_unbalance = np.random.uniform(0, 3, n_samples)
        
        return THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance


class AnomalyInjector:
    """Inject various types of anomalies into voltage data."""
    
    ANOMALY_TYPES = {
        1: 'Undervoltage',
        2: 'Overvoltage',
        3: 'Voltage_Sag',
        4: 'Harmonic',
        5: 'Unbalance',
    }
    
    @staticmethod
    def inject_undervoltage(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        start: int, duration: int, severity: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject undervoltage: voltage drops below -10%."""
        end = min(start + duration, len(Va))
        
        # Gradual onset and recovery
        ramp_len = min(10, duration // 4)
        for i in range(start, end):
            if i < start + ramp_len:
                factor = 1 - severity * (i - start) / ramp_len
            elif i > end - ramp_len:
                factor = 1 - severity * (end - i) / ramp_len
            else:
                factor = 1 - severity + np.random.uniform(-0.02, 0.02)
            
            Va[i] *= factor
            Vb[i] *= factor
            Vc[i] *= factor
            
        return Va, Vb, Vc
    
    @staticmethod
    def inject_overvoltage(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        start: int, duration: int, severity: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject overvoltage: voltage rises above +10%."""
        end = min(start + duration, len(Va))
        
        ramp_len = min(10, duration // 4)
        for i in range(start, end):
            if i < start + ramp_len:
                factor = 1 + severity * (i - start) / ramp_len
            elif i > end - ramp_len:
                factor = 1 + severity * (end - i) / ramp_len
            else:
                factor = 1 + severity + np.random.uniform(-0.02, 0.02)
            
            Va[i] *= factor
            Vb[i] *= factor
            Vc[i] *= factor
            
        return Va, Vb, Vc
    
    @staticmethod
    def inject_voltage_sag(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        start: int, duration: int, depth: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject voltage sag: sudden temporary voltage drop."""
        end = min(start + duration, len(Va))
        
        # Rapid onset, gradual recovery (characteristic of sags)
        onset_len = min(3, duration // 10)
        recovery_len = min(15, duration // 2)
        
        for i in range(start, end):
            if i < start + onset_len:
                factor = 1 - depth * (i - start) / onset_len
            elif i > end - recovery_len:
                factor = 1 - depth * (end - i) / recovery_len
            else:
                factor = 1 - depth
            
            # May affect only one or two phases
            phase_affected = np.random.choice([0, 1, 2], size=np.random.randint(1, 4), replace=False)
            if 0 in phase_affected:
                Va[i] *= factor
            if 1 in phase_affected:
                Vb[i] *= factor
            if 2 in phase_affected:
                Vc[i] *= factor
                
        return Va, Vb, Vc
    
    @staticmethod
    def inject_unbalance(
        Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray,
        start: int, duration: int, unbalance_factor: float = 0.08
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject three-phase unbalance."""
        end = min(start + duration, len(Va))
        
        # Random phase selection and unbalance magnitude
        phase = np.random.randint(0, 3)
        sign = np.random.choice([-1, 1])
        
        for i in range(start, end):
            factor = 1 + sign * unbalance_factor * np.random.uniform(0.8, 1.2)
            if phase == 0:
                Va[i] *= factor
            elif phase == 1:
                Vb[i] *= factor
            else:
                Vc[i] *= factor
                
        return Va, Vb, Vc


class RuralVoltageDatasetGenerator:
    """Main class for generating the complete dataset."""
    
    def __init__(
        self,
        train_samples: int = 50000,
        test_samples: int = 10000,
        anomaly_ratio: float = 0.1,
        seed: int = 42
    ):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        
        np.random.seed(seed)
        
        self.voltage_gen = ThreePhaseVoltageGenerator()
        self.current_gen = CurrentGenerator()
        
    def generate_normal_data(self, n_samples: int) -> pd.DataFrame:
        """Generate normal operation data."""
        # Generate voltage
        Va, Vb, Vc = self.voltage_gen.generate_balanced_voltage(n_samples)
        
        # Generate current
        Ia, Ib, Ic = self.current_gen.generate_current(Va, Vb, Vc)
        
        # Generate power metrics
        P, Q, S, PF = PowerMetricsGenerator.generate_power_metrics(
            Va, Vb, Vc, Ia, Ib, Ic
        )
        
        # Generate quality metrics
        THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance = \
            PowerQualityMetricsGenerator.generate_quality_metrics(Va, Vb, Vc)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Va': Va, 'Vb': Vb, 'Vc': Vc,
            'Ia': Ia, 'Ib': Ib, 'Ic': Ic,
            'P': P, 'Q': Q, 'S': S, 'PF': PF,
            'THD_Va': THD_Va, 'THD_Vb': THD_Vb, 'THD_Vc': THD_Vc,
            'Freq': Freq, 'V_unbalance': V_unbalance, 'I_unbalance': I_unbalance
        })
        
        return data
    
    def generate_test_data_with_anomalies(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate test data with injected anomalies."""
        n_samples = self.test_samples
        
        # Start with normal data
        Va, Vb, Vc = self.voltage_gen.generate_balanced_voltage(n_samples)
        
        # Initialize labels
        labels = np.zeros(n_samples, dtype=int)
        
        # Calculate number of anomaly events
        n_anomaly_samples = int(n_samples * self.anomaly_ratio)
        anomaly_types = list(AnomalyInjector.ANOMALY_TYPES.keys())
        
        # Distribute anomalies
        current_pos = int(n_samples * 0.05)  # Start after initial normal period
        anomalies_per_type = n_anomaly_samples // len(anomaly_types)
        
        injector = AnomalyInjector()
        
        for anomaly_type in anomaly_types:
            # Create several events per type
            n_events = max(1, anomalies_per_type // np.random.randint(20, 50))
            
            for _ in range(n_events):
                if current_pos >= n_samples - 200:
                    break
                
                # Random duration (20-100 samples)
                duration = np.random.randint(20, 100)
                
                # Inject anomaly
                if anomaly_type == 1:  # Undervoltage
                    Va, Vb, Vc = injector.inject_undervoltage(
                        Va, Vb, Vc, current_pos, duration, 
                        severity=np.random.uniform(0.12, 0.20)
                    )
                elif anomaly_type == 2:  # Overvoltage
                    Va, Vb, Vc = injector.inject_overvoltage(
                        Va, Vb, Vc, current_pos, duration,
                        severity=np.random.uniform(0.10, 0.15)
                    )
                elif anomaly_type == 3:  # Voltage sag
                    Va, Vb, Vc = injector.inject_voltage_sag(
                        Va, Vb, Vc, current_pos, duration,
                        depth=np.random.uniform(0.2, 0.5)
                    )
                elif anomaly_type == 4:  # Harmonic (handled in THD later)
                    pass
                elif anomaly_type == 5:  # Unbalance
                    Va, Vb, Vc = injector.inject_unbalance(
                        Va, Vb, Vc, current_pos, duration,
                        unbalance_factor=np.random.uniform(0.06, 0.12)
                    )
                
                # Set labels
                end_pos = min(current_pos + duration, n_samples)
                labels[current_pos:end_pos] = anomaly_type
                
                # Move to next position with gap
                current_pos = end_pos + np.random.randint(100, 500)
        
        # Generate current based on (potentially anomalous) voltage
        Ia, Ib, Ic = self.current_gen.generate_current(Va, Vb, Vc)
        
        # Generate power metrics
        P, Q, S, PF = PowerMetricsGenerator.generate_power_metrics(
            Va, Vb, Vc, Ia, Ib, Ic
        )
        
        # Generate quality metrics (with anomalous THD for labeled samples)
        THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance = \
            PowerQualityMetricsGenerator.generate_quality_metrics(Va, Vb, Vc)
        
        # Inject harmonic anomalies into THD
        harmonic_mask = labels == 4
        if np.any(harmonic_mask):
            THD_Va[harmonic_mask] = np.random.uniform(6, 15, np.sum(harmonic_mask))
            THD_Vb[harmonic_mask] = np.random.uniform(6, 15, np.sum(harmonic_mask))
            THD_Vc[harmonic_mask] = np.random.uniform(6, 15, np.sum(harmonic_mask))
        
        # Create DataFrame
        data = pd.DataFrame({
            'Va': Va, 'Vb': Vb, 'Vc': Vc,
            'Ia': Ia, 'Ib': Ib, 'Ic': Ic,
            'P': P, 'Q': Q, 'S': S, 'PF': PF,
            'THD_Va': THD_Va, 'THD_Vb': THD_Vb, 'THD_Vc': THD_Vc,
            'Freq': Freq, 'V_unbalance': V_unbalance, 'I_unbalance': I_unbalance
        })
        
        return data, labels
    
    def generate_dataset(self, output_dir: str):
        """Generate and save the complete dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating RuralVoltage V2 Dataset...")
        print(f"  Train samples: {self.train_samples}")
        print(f"  Test samples: {self.test_samples}")
        print(f"  Anomaly ratio: {self.anomaly_ratio}")
        
        # Generate training data (mainly normal)
        print("Generating training data...")
        train_data = self.generate_normal_data(self.train_samples)
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        
        # Generate test data with anomalies
        print("Generating test data with anomalies...")
        test_data, test_labels = self.generate_test_data_with_anomalies()
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        # Save labels
        label_df = pd.DataFrame({'label': test_labels})
        label_df.to_csv(os.path.join(output_dir, 'test_label.csv'), index=False)
        
        # Save metadata
        metadata = {
            'version': '2.0',
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'anomaly_ratio': self.anomaly_ratio,
            'seed': self.seed,
            'features': list(train_data.columns),
            'n_features': len(train_data.columns),
            'anomaly_types': AnomalyInjector.ANOMALY_TYPES,
            'anomaly_distribution': {
                str(k): int(np.sum(test_labels == k)) 
                for k in AnomalyInjector.ANOMALY_TYPES.keys()
            },
            'normal_samples': int(np.sum(test_labels == 0)),
            'generated_at': datetime.now().isoformat(),
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        print("\nDataset generated successfully!")
        print(f"  Output directory: {output_dir}")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        print("\nAnomaly distribution in test set:")
        for k, v in AnomalyInjector.ANOMALY_TYPES.items():
            count = np.sum(test_labels == k)
            pct = count / len(test_labels) * 100
            print(f"  {v}: {count} samples ({pct:.2f}%)")
        print(f"  Normal: {np.sum(test_labels == 0)} samples ({np.sum(test_labels == 0) / len(test_labels) * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate Rural Voltage Dataset V2')
    parser.add_argument('--train_samples', type=int, default=50000,
                        help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=10000,
                        help='Number of test samples')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1,
                        help='Ratio of anomalous samples in test set')
    parser.add_argument('--output_dir', type=str, 
                        default='../../dataset/RuralVoltageV2',
                        help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generator = RuralVoltageDatasetGenerator(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed
    )
    
    generator.generate_dataset(args.output_dir)


if __name__ == '__main__':
    main()
