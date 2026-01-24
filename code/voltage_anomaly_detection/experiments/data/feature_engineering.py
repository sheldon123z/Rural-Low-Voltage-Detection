"""
Feature Engineering Module for Rural Voltage Anomaly Detection

This module provides feature engineering utilities:
1. Statistical features (rolling statistics)
2. Frequency domain features (FFT-based)
3. Inter-phase features (phase differences)
4. Symmetrical component features

Usage:
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(window_size=20)
    enhanced_data = engineer.transform(data)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import List, Optional, Tuple


class StatisticalFeatures:
    """Extract statistical features from time series."""
    
    def __init__(self, window_sizes: List[int] = [10, 20, 50]):
        self.window_sizes = window_sizes
    
    def extract(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract rolling statistical features.
        
        Args:
            data: Input DataFrame
            columns: Columns to process (default: all numeric)
            
        Returns:
            DataFrame with additional statistical features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        result = data.copy()
        
        for col in columns:
            for ws in self.window_sizes:
                # Rolling mean
                result[f'{col}_mean_{ws}'] = data[col].rolling(window=ws, min_periods=1).mean()
                
                # Rolling std
                result[f'{col}_std_{ws}'] = data[col].rolling(window=ws, min_periods=1).std().fillna(0)
                
                # Rolling max - min (range)
                result[f'{col}_range_{ws}'] = (
                    data[col].rolling(window=ws, min_periods=1).max() -
                    data[col].rolling(window=ws, min_periods=1).min()
                )
                
        return result
    
    def extract_advanced(self, data: pd.DataFrame, columns: List[str], window: int = 20) -> pd.DataFrame:
        """Extract advanced statistical features."""
        result = data.copy()
        
        for col in columns:
            series = data[col]
            
            # Skewness
            result[f'{col}_skew_{window}'] = series.rolling(window=window, min_periods=1).skew().fillna(0)
            
            # Kurtosis
            result[f'{col}_kurt_{window}'] = series.rolling(window=window, min_periods=1).kurt().fillna(0)
            
            # Percentiles
            result[f'{col}_q25_{window}'] = series.rolling(window=window, min_periods=1).quantile(0.25)
            result[f'{col}_q75_{window}'] = series.rolling(window=window, min_periods=1).quantile(0.75)
            
            # Rate of change
            result[f'{col}_roc'] = series.pct_change().fillna(0)
            
        return result


class FrequencyFeatures:
    """Extract frequency domain features using FFT."""
    
    def __init__(self, sampling_rate: float = 1.0, n_fft: int = 64):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
    
    def extract(self, data: pd.DataFrame, columns: List[str], window: int = 64) -> pd.DataFrame:
        """
        Extract FFT-based frequency features.
        
        Args:
            data: Input DataFrame
            columns: Columns to process
            window: Window size for STFT
            
        Returns:
            DataFrame with frequency features
        """
        result = data.copy()
        
        for col in columns:
            series = np.asarray(data[col].values, dtype=np.float64)
            n_samples = len(series)
            
            # Initialize feature arrays
            dominant_freq = np.zeros(n_samples)
            spectral_energy = np.zeros(n_samples)
            spectral_entropy = np.zeros(n_samples)
            
            for i in range(window, n_samples):
                segment = series[i-window:i]
                
                # Apply FFT
                fft_result = fft(segment)
                freqs = fftfreq(window, d=1.0/self.sampling_rate)
                
                # Get positive frequencies
                pos_mask = freqs > 0
                pos_freqs = freqs[pos_mask]
                pos_magnitudes = np.abs(fft_result[pos_mask])
                
                if len(pos_magnitudes) > 0:
                    # Dominant frequency
                    dominant_idx = np.argmax(pos_magnitudes)
                    dominant_freq[i] = pos_freqs[dominant_idx]
                    
                    # Spectral energy
                    spectral_energy[i] = np.sum(pos_magnitudes ** 2)
                    
                    # Spectral entropy
                    normalized_mag = pos_magnitudes / (np.sum(pos_magnitudes) + 1e-8)
                    spectral_entropy[i] = -np.sum(normalized_mag * np.log(normalized_mag + 1e-8))
            
            result[f'{col}_dominant_freq'] = dominant_freq
            result[f'{col}_spectral_energy'] = spectral_energy
            result[f'{col}_spectral_entropy'] = spectral_entropy
        
        return result
    
    def extract_harmonic_features(
        self, 
        data: pd.DataFrame, 
        voltage_cols: List[str] = ['Va', 'Vb', 'Vc'],
        fundamental_freq: float = 50.0
    ) -> pd.DataFrame:
        """Extract harmonic-related features for power quality analysis."""
        result = data.copy()
        
        # Calculate harmonic ratios (simplified simulation based on THD)
        for col in voltage_cols:
            if col in data.columns:
                thd_col = f'THD_{col}'
                if thd_col in data.columns:
                    # Estimate harmonic content from THD
                    result[f'{col}_3rd_harmonic_est'] = data[thd_col] * 0.6  # 3rd harmonic typically dominant
                    result[f'{col}_5th_harmonic_est'] = data[thd_col] * 0.3
                    result[f'{col}_7th_harmonic_est'] = data[thd_col] * 0.1
        
        return result


class InterPhaseFeatures:
    """Extract features capturing relationships between phases."""
    
    def __init__(self):
        pass
    
    def extract(
        self, 
        data: pd.DataFrame,
        voltage_cols: List[str] = ['Va', 'Vb', 'Vc'],
        current_cols: List[str] = ['Ia', 'Ib', 'Ic']
    ) -> pd.DataFrame:
        """
        Extract inter-phase relationship features.
        
        Args:
            data: Input DataFrame
            voltage_cols: Three-phase voltage column names
            current_cols: Three-phase current column names
            
        Returns:
            DataFrame with inter-phase features
        """
        result = data.copy()
        
        # Voltage phase differences
        if all(col in data.columns for col in voltage_cols):
            Va, Vb, Vc = data[voltage_cols[0]], data[voltage_cols[1]], data[voltage_cols[2]]
            
            # Absolute differences
            result['V_ab_diff'] = np.abs(Va - Vb)
            result['V_bc_diff'] = np.abs(Vb - Vc)
            result['V_ca_diff'] = np.abs(Vc - Va)
            
            # Relative differences
            V_mean = (Va + Vb + Vc) / 3
            result['V_ab_rel_diff'] = result['V_ab_diff'] / (V_mean + 1e-8)
            result['V_bc_rel_diff'] = result['V_bc_diff'] / (V_mean + 1e-8)
            result['V_ca_rel_diff'] = result['V_ca_diff'] / (V_mean + 1e-8)
            
            # Max deviation from mean
            result['V_max_deviation'] = np.maximum(
                np.abs(Va - V_mean),
                np.maximum(np.abs(Vb - V_mean), np.abs(Vc - V_mean))
            )
            
            # Coefficient of variation
            V_std = np.std(np.column_stack([Va, Vb, Vc]), axis=1)
            result['V_cv'] = V_std / (V_mean + 1e-8)
        
        # Current phase differences
        if all(col in data.columns for col in current_cols):
            Ia, Ib, Ic = data[current_cols[0]], data[current_cols[1]], data[current_cols[2]]
            
            result['I_ab_diff'] = np.abs(Ia - Ib)
            result['I_bc_diff'] = np.abs(Ib - Ic)
            result['I_ca_diff'] = np.abs(Ic - Ia)
            
            I_mean = (Ia + Ib + Ic) / 3
            result['I_max_deviation'] = np.maximum(
                np.abs(Ia - I_mean),
                np.maximum(np.abs(Ib - I_mean), np.abs(Ic - I_mean))
            )
        
        return result


class SymmetricalComponentFeatures:
    """
    Extract symmetrical component features (positive, negative, zero sequence).
    
    In balanced three-phase systems:
    - Positive sequence: Normal rotating field
    - Negative sequence: Indicates unbalance/faults
    - Zero sequence: Indicates ground faults
    """
    
    # Transformation matrix constant (a = e^(j*2π/3))
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a ** 2
    
    def extract(
        self, 
        data: pd.DataFrame,
        voltage_cols: List[str] = ['Va', 'Vb', 'Vc']
    ) -> pd.DataFrame:
        """
        Extract symmetrical component features.
        
        For three-phase voltage Va, Vb, Vc:
        V0 = (Va + Vb + Vc) / 3  (zero sequence)
        V1 = (Va + a*Vb + a²*Vc) / 3  (positive sequence)  
        V2 = (Va + a²*Vb + a*Vc) / 3  (negative sequence)
        
        Since we don't have phase angles in the data, we use magnitude approximation.
        """
        result = data.copy()
        
        if not all(col in data.columns for col in voltage_cols):
            return result
        
        Va = np.asarray(data[voltage_cols[0]].values, dtype=np.float64)
        Vb = np.asarray(data[voltage_cols[1]].values, dtype=np.float64)
        Vc = np.asarray(data[voltage_cols[2]].values, dtype=np.float64)
        
        # Simplified symmetrical components (magnitude-based approximation)
        # Zero sequence (average of three phases)
        V0 = (Va + Vb + Vc) / 3
        
        # For magnitude-based analysis without phase angles:
        # Positive sequence ≈ average (for balanced system)
        # Negative sequence ≈ proportional to unbalance
        
        # Approximate positive sequence as the average
        V1_approx = (Va + Vb + Vc) / 3
        
        # Approximate negative sequence using unbalance
        V_max = np.maximum(Va, np.maximum(Vb, Vc))
        V_min = np.minimum(Va, np.minimum(Vb, Vc))
        V2_approx = (V_max - V_min) / 3  # Proportional to unbalance
        
        # Negative sequence ratio (unbalance factor)
        neg_seq_ratio = V2_approx / (V1_approx + 1e-8)
        
        result['V_zero_seq'] = V0
        result['V_pos_seq_approx'] = V1_approx
        result['V_neg_seq_approx'] = V2_approx
        result['V_neg_seq_ratio'] = neg_seq_ratio
        
        # Zero sequence ratio (indicates ground fault tendency)
        result['V_zero_seq_ratio'] = np.abs(V0 - V1_approx) / (V1_approx + 1e-8)
        
        return result


class FeatureEngineer:
    """Main class combining all feature engineering techniques."""
    
    def __init__(
        self,
        enable_statistical: bool = True,
        enable_frequency: bool = True,
        enable_interphase: bool = True,
        enable_symmetrical: bool = True,
        stat_window_sizes: List[int] = [10, 20],
        freq_window: int = 32
    ):
        self.enable_statistical = enable_statistical
        self.enable_frequency = enable_frequency
        self.enable_interphase = enable_interphase
        self.enable_symmetrical = enable_symmetrical
        
        self.stat_extractor = StatisticalFeatures(window_sizes=stat_window_sizes)
        self.freq_extractor = FrequencyFeatures(n_fft=freq_window)
        self.interphase_extractor = InterPhaseFeatures()
        self.symm_extractor = SymmetricalComponentFeatures()
        
    def transform(
        self, 
        data: pd.DataFrame,
        voltage_cols: List[str] = ['Va', 'Vb', 'Vc'],
        current_cols: List[str] = ['Ia', 'Ib', 'Ic']
    ) -> pd.DataFrame:
        """
        Apply all enabled feature engineering transformations.
        
        Args:
            data: Input DataFrame with raw features
            voltage_cols: Names of voltage columns
            current_cols: Names of current columns
            
        Returns:
            DataFrame with additional engineered features
        """
        result = data.copy()
        
        if self.enable_statistical:
            print("Extracting statistical features...")
            # Apply to voltage columns
            result = self.stat_extractor.extract(result, columns=voltage_cols[:3])
        
        if self.enable_frequency:
            print("Extracting frequency features...")
            # Apply to voltage columns (most important for quality analysis)
            if all(col in data.columns for col in voltage_cols):
                result = self.freq_extractor.extract(result, columns=voltage_cols)
                result = self.freq_extractor.extract_harmonic_features(result, voltage_cols)
        
        if self.enable_interphase:
            print("Extracting inter-phase features...")
            result = self.interphase_extractor.extract(result, voltage_cols, current_cols)
        
        if self.enable_symmetrical:
            print("Extracting symmetrical component features...")
            result = self.symm_extractor.extract(result, voltage_cols)
        
        # Fill any NaN values
        result = result.bfill().ffill().fillna(0)
        
        return result
    
    def get_feature_names(self, original_columns: List[str]) -> List[str]:
        """Get list of all feature names after transformation."""
        # Create dummy data to get feature names
        dummy = pd.DataFrame(np.random.randn(100, len(original_columns)), 
                           columns=original_columns)
        transformed = self.transform(dummy)
        return transformed.columns.tolist()


def main():
    """Demo of feature engineering."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Va': 220 + np.random.normal(0, 2, n_samples),
        'Vb': 220 + np.random.normal(0, 2, n_samples),
        'Vc': 220 + np.random.normal(0, 2, n_samples),
        'Ia': 15 + np.random.normal(0, 0.5, n_samples),
        'Ib': 15 + np.random.normal(0, 0.5, n_samples),
        'Ic': 15 + np.random.normal(0, 0.5, n_samples),
        'THD_Va': np.random.uniform(1, 3, n_samples),
        'THD_Vb': np.random.uniform(1, 3, n_samples),
        'THD_Vc': np.random.uniform(1, 3, n_samples),
    })
    
    print(f"Original data shape: {data.shape}")
    print(f"Original columns: {data.columns.tolist()}")
    
    # Apply feature engineering
    engineer = FeatureEngineer(
        stat_window_sizes=[10, 20],
        freq_window=32
    )
    
    enhanced = engineer.transform(data)
    
    print(f"\nEnhanced data shape: {enhanced.shape}")
    print(f"New columns added: {len(enhanced.columns) - len(data.columns)}")
    print(f"\nAll columns:\n{enhanced.columns.tolist()}")


if __name__ == '__main__':
    main()
