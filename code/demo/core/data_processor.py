"""
Data Processor Module for Gradio Demo
农村低压配电网电压异常检测项目

Provides:
- DataProcessor: Class for data preprocessing, normalization, and windowing
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    Data preprocessing module for voltage anomaly detection.

    Supports:
    - CSV data loading
    - StandardScaler normalization
    - Sliding window segmentation
    - Feature extraction

    Example:
        >>> processor = DataProcessor(seq_len=100)
        >>> processor.fit(train_data)
        >>> windows = processor.transform(test_data)
    """

    # Default feature columns for RuralVoltage dataset
    DEFAULT_FEATURE_COLS = [
        "Va", "Vb", "Vc",           # Three-phase voltages
        "Ia", "Ib", "Ic",           # Three-phase currents
        "P", "Q", "S", "PF",        # Power metrics
        "THD_Va", "THD_Vb", "THD_Vc",  # Harmonic distortion
        "Freq",                     # Frequency
        "V_unbalance", "I_unbalance"  # Unbalance ratios
    ]

    def __init__(
        self,
        seq_len: int = 100,
        step: int = 1,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True
    ):
        """
        Initialize DataProcessor.

        Args:
            seq_len: Length of sliding window (default: 100)
            step: Step size for sliding window (default: 1)
            feature_cols: List of feature column names to use
            normalize: Whether to apply StandardScaler normalization
        """
        self.seq_len = seq_len
        self.step = step
        self.feature_cols = feature_cols
        self.normalize = normalize

        self.scaler = StandardScaler() if normalize else None
        self._is_fitted = False
        self._n_features = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "DataProcessor":
        """
        Fit the scaler on training data.

        Args:
            data: Training data as numpy array [N, C] or DataFrame

        Returns:
            self for chaining
        """
        data_array = self._to_numpy(data)

        if self.normalize:
            self.scaler.fit(data_array)

        self._n_features = data_array.shape[1]
        self._is_fitted = True

        return self

    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_windows: bool = True
    ) -> np.ndarray:
        """
        Transform data using fitted scaler and optionally create windows.

        Args:
            data: Data to transform [N, C] or DataFrame
            return_windows: If True, return sliding windows [num_windows, seq_len, C]
                          If False, return normalized data [N, C]

        Returns:
            Transformed data
        """
        data_array = self._to_numpy(data)

        # Normalize
        if self.normalize and self._is_fitted:
            data_array = self.scaler.transform(data_array)

        # Create windows
        if return_windows:
            return self.create_windows(data_array)

        return data_array

    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_windows: bool = True
    ) -> np.ndarray:
        """
        Fit scaler and transform data.

        Args:
            data: Training data
            return_windows: Whether to return sliding windows

        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data, return_windows=return_windows)

    def create_windows(
        self,
        data: np.ndarray,
        step: Optional[int] = None
    ) -> np.ndarray:
        """
        Create sliding windows from sequential data.

        Args:
            data: Input data [N, C]
            step: Optional override for step size

        Returns:
            Windows array [num_windows, seq_len, C]
        """
        if step is None:
            step = self.step

        n_samples, n_features = data.shape

        # Calculate number of windows
        n_windows = (n_samples - self.seq_len) // step + 1

        if n_windows <= 0:
            raise ValueError(
                f"Data length {n_samples} is too short for "
                f"seq_len={self.seq_len}, step={step}"
            )

        # Create windows using stride tricks for efficiency
        windows = np.zeros((n_windows, self.seq_len, n_features), dtype=np.float32)
        for i in range(n_windows):
            start_idx = i * step
            windows[i] = data[start_idx:start_idx + self.seq_len]

        return windows

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data [N, C] or [B, T, C]

        Returns:
            Data in original scale
        """
        if not self.normalize or self.scaler is None:
            return data

        original_shape = data.shape

        # Handle 3D input
        if len(original_shape) == 3:
            B, T, C = original_shape
            data = data.reshape(-1, C)
            data = self.scaler.inverse_transform(data)
            data = data.reshape(B, T, C)
        else:
            data = self.scaler.inverse_transform(data)

        return data

    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert input to numpy array, selecting feature columns if needed."""
        if isinstance(data, pd.DataFrame):
            # Select feature columns
            if self.feature_cols:
                available_cols = [c for c in self.feature_cols if c in data.columns]
                if available_cols:
                    data = data[available_cols]
                else:
                    # Exclude common non-feature columns
                    exclude_cols = ["timestamp", "date", "time", "label", "index"]
                    feature_cols = [c for c in data.columns if c not in exclude_cols]
                    data = data[feature_cols]
            else:
                # Use all numeric columns
                exclude_cols = ["timestamp", "date", "time", "label", "index"]
                feature_cols = [c for c in data.columns if c not in exclude_cols]
                data = data[feature_cols]

            data = data.values

        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)

        return data.astype(np.float32)

    @classmethod
    def load_csv(
        cls,
        file_path: Union[str, Path],
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            feature_cols: Optional list of feature columns to use

        Returns:
            Tuple of (data, labels, feature_names)
            - data: Feature values [N, C]
            - labels: Label values [N] if 'label' column exists, else None
            - feature_names: List of feature column names
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # Extract labels if present
        labels = None
        if "label" in df.columns:
            labels = df["label"].values

        # Determine feature columns
        exclude_cols = ["timestamp", "date", "time", "label", "index", "Unnamed: 0"]
        if feature_cols:
            available_cols = [c for c in feature_cols if c in df.columns]
            if available_cols:
                use_cols = available_cols
            else:
                use_cols = [c for c in df.columns if c not in exclude_cols]
        else:
            use_cols = [c for c in df.columns if c not in exclude_cols]

        data = df[use_cols].values
        data = np.nan_to_num(data, nan=0.0).astype(np.float32)

        return data, labels, use_cols

    @classmethod
    def load_dataset(
        cls,
        root_path: Union[str, Path],
        dataset_type: str = "RuralVoltage"
    ) -> Dict[str, Any]:
        """
        Load a complete dataset (train, test, labels).

        Args:
            root_path: Path to dataset directory
            dataset_type: Type of dataset ("RuralVoltage", "PSM", etc.)

        Returns:
            Dict with train_data, test_data, test_labels, feature_names
        """
        root_path = Path(root_path)

        if dataset_type == "RuralVoltage":
            train_path = root_path / "train.csv"
            test_path = root_path / "test.csv"
            label_path = root_path / "test_label.csv"

            train_data, _, feature_names = cls.load_csv(train_path)
            test_data, _, _ = cls.load_csv(test_path, feature_cols=feature_names)
            _, test_labels, _ = cls.load_csv(label_path)

        elif dataset_type == "PSM":
            train_path = root_path / "train.csv"
            test_path = root_path / "test.csv"
            label_path = root_path / "test_label.csv"

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            label_df = pd.read_csv(label_path)

            train_data = train_df.values[:, 1:]  # Skip first column
            test_data = test_df.values[:, 1:]
            test_labels = label_df.values[:, 1:]
            feature_names = list(train_df.columns[1:])

            train_data = np.nan_to_num(train_data, nan=0.0).astype(np.float32)
            test_data = np.nan_to_num(test_data, nan=0.0).astype(np.float32)

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return {
            "train_data": train_data,
            "test_data": test_data,
            "test_labels": test_labels,
            "feature_names": feature_names
        }

    def get_scaler_params(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get scaler parameters (mean and scale).

        Returns:
            Dict with 'mean' and 'scale' arrays, or None if not fitted
        """
        if not self._is_fitted or self.scaler is None:
            return None

        return {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_
        }

    def set_scaler_params(self, mean: np.ndarray, scale: np.ndarray) -> None:
        """
        Set scaler parameters directly (useful for loading saved parameters).

        Args:
            mean: Mean values for each feature
            scale: Scale (std) values for each feature
        """
        if self.scaler is None:
            self.scaler = StandardScaler()

        self.scaler.mean_ = mean
        self.scaler.scale_ = scale
        self.scaler.var_ = scale ** 2
        self.scaler.n_features_in_ = len(mean)
        self._n_features = len(mean)
        self._is_fitted = True

    @property
    def n_features(self) -> Optional[int]:
        """Number of features."""
        return self._n_features

    @property
    def is_fitted(self) -> bool:
        """Whether the processor has been fitted."""
        return self._is_fitted


def preprocess_for_inference(
    data: Union[np.ndarray, pd.DataFrame],
    scaler_mean: Optional[np.ndarray] = None,
    scaler_scale: Optional[np.ndarray] = None,
    seq_len: int = 100,
    step: int = 1
) -> np.ndarray:
    """
    Convenience function to preprocess data for model inference.

    Args:
        data: Raw input data
        scaler_mean: Optional pre-computed scaler mean
        scaler_scale: Optional pre-computed scaler scale
        seq_len: Window length
        step: Step size

    Returns:
        Preprocessed windows ready for model input
    """
    processor = DataProcessor(seq_len=seq_len, step=step)

    if scaler_mean is not None and scaler_scale is not None:
        processor.set_scaler_params(scaler_mean, scaler_scale)
        return processor.transform(data, return_windows=True)
    else:
        return processor.fit_transform(data, return_windows=True)


if __name__ == "__main__":
    # Test module
    print("Testing DataProcessor...")

    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(1000, 16).astype(np.float32)

    # Test processor
    processor = DataProcessor(seq_len=100, step=1)
    windows = processor.fit_transform(sample_data)

    print(f"Input shape: {sample_data.shape}")
    print(f"Windows shape: {windows.shape}")
    print(f"Expected windows: {(1000 - 100) // 1 + 1}")
    print(f"Is fitted: {processor.is_fitted}")
    print(f"N features: {processor.n_features}")

    # Test inverse transform
    original = processor.inverse_transform(windows[:5])
    print(f"Inverse transform shape: {original.shape}")
