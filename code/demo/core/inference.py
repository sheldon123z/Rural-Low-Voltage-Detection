"""
Voltage Anomaly Detection Inference Module

This module provides a high-level API for voltage anomaly detection inference.
Supports CPU inference with torch.no_grad() optimization.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Add code directory to path for model imports
CODE_DIR = Path(__file__).parent.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import model_dict


class VoltageAnomalyDetector:
    """
    High-level API for voltage anomaly detection inference.

    This class wraps the model loading, preprocessing, and inference logic
    for easy use in applications like Gradio demos.

    Example:
        >>> detector = VoltageAnomalyDetector("VoltageTimesNet_v2", checkpoint_path)
        >>> detector.load_model()
        >>> results = detector.predict(data, threshold=0.5)
        >>> print(results["labels"])  # Anomaly labels (0 or 1)
    """

    # Default model configurations
    DEFAULT_CONFIGS = {
        "VoltageTimesNet_v2": {
            "enc_in": 16,
            "c_out": 16,
            "seq_len": 100,
            "d_model": 64,
            "d_ff": 64,
            "e_layers": 2,
            "top_k": 5,
            "num_kernels": 6,
            "dropout": 0.1,
            "embed": "fixed",
            "freq": "h",
            "task_name": "anomaly_detection",
            "pred_len": 0,
            "label_len": 0,
        },
        "VoltageTimesNet": {
            "enc_in": 16,
            "c_out": 16,
            "seq_len": 100,
            "d_model": 64,
            "d_ff": 64,
            "e_layers": 2,
            "top_k": 5,
            "num_kernels": 6,
            "dropout": 0.1,
            "embed": "fixed",
            "freq": "h",
            "task_name": "anomaly_detection",
            "pred_len": 0,
            "label_len": 0,
        },
        "TimesNet": {
            "enc_in": 16,
            "c_out": 16,
            "seq_len": 100,
            "d_model": 64,
            "d_ff": 64,
            "e_layers": 2,
            "top_k": 5,
            "num_kernels": 6,
            "dropout": 0.1,
            "embed": "fixed",
            "freq": "h",
            "task_name": "anomaly_detection",
            "pred_len": 0,
            "label_len": 0,
        },
        "DLinear": {
            "enc_in": 16,
            "c_out": 16,
            "seq_len": 100,
            "pred_len": 100,
            "individual": False,
            "task_name": "anomaly_detection",
            "label_len": 0,
        },
    }

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the VoltageAnomalyDetector.

        Args:
            model_name: Name of the model (e.g., "VoltageTimesNet_v2", "TimesNet")
            checkpoint_path: Path to the model checkpoint file (.pth)
            device: Device to run inference on ("cpu" or "cuda")
            config_path: Path to model config JSON file (optional)
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.config_path = config_path

        self.model: Optional[nn.Module] = None
        self.config: Dict = {}
        self._is_loaded = False

        # MSE criterion for reconstruction error
        self.anomaly_criterion = nn.MSELoss(reduction='none')

    def _load_config(self) -> Dict:
        """
        Load model configuration from file or use defaults.

        Returns:
            Dictionary containing model configuration
        """
        config = {}

        # Try loading from config file first
        if self.config_path is not None:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)

        # Fall back to default config
        if not config and self.model_name in self.DEFAULT_CONFIGS:
            config = self.DEFAULT_CONFIGS[self.model_name].copy()

        # Ensure model name is set
        config["model"] = self.model_name

        return config

    def _config_to_args(self, config: Dict) -> Namespace:
        """
        Convert config dictionary to argparse Namespace.

        Args:
            config: Configuration dictionary

        Returns:
            Namespace object with configuration attributes
        """
        # Merge with defaults
        default_config = self.DEFAULT_CONFIGS.get(self.model_name, {})
        merged_config = {**default_config, **config}

        # Ensure required fields
        merged_config.setdefault("task_name", "anomaly_detection")
        merged_config.setdefault("embed", "fixed")
        merged_config.setdefault("freq", "h")
        merged_config.setdefault("dropout", 0.1)
        merged_config.setdefault("pred_len", 0)
        merged_config.setdefault("label_len", 0)

        return Namespace(**merged_config)

    def load_model(self, strict: bool = False) -> None:
        """
        Load the model and weights.

        This method initializes the model architecture and loads
        pretrained weights if a checkpoint path is provided.

        Args:
            strict: Whether to strictly enforce that the keys in state_dict
                    match the keys returned by the model's state_dict function.
                    If False, allows loading checkpoints with minor mismatches.
                    Default is False for better compatibility.

        Raises:
            ValueError: If model name is not found in model registry
            FileNotFoundError: If checkpoint file does not exist
        """
        # Load configuration
        self.config = self._load_config()
        args = self._config_to_args(self.config)

        # Check model exists
        if self.model_name not in model_dict:
            available = list(model_dict.keys())
            raise ValueError(
                f"Model '{self.model_name}' not found. "
                f"Available models: {available}"
            )

        # Build model
        Model = model_dict[self.model_name]
        self.model = Model(args)

        # Load checkpoint if provided
        if self.checkpoint_path is not None:
            checkpoint_file = Path(self.checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {self.checkpoint_path}"
                )

            # Load weights
            state_dict = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True
            )

            # Load state dict with optional strict mode
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=strict
            )

            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

            print(f"Loaded checkpoint from: {self.checkpoint_path}")

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

        print(f"Model '{self.model_name}' loaded successfully on {self.device}")

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data for model inference.

        Args:
            data: Input numpy array with shape:
                  - (seq_len, n_features) for single sample
                  - (batch_size, seq_len, n_features) for batch

        Returns:
            Preprocessed tensor with shape (batch_size, seq_len, n_features)
        """
        # Ensure numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Convert to float32
        data = data.astype(np.float32)

        # Add batch dimension if needed
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # (1, seq_len, n_features)

        # Validate shape
        if data.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D array, got shape {data.shape}"
            )

        # Convert to tensor
        tensor = torch.from_numpy(data).to(self.device)

        return tensor

    def get_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for input data.

        The reconstruction error is computed as the mean squared error
        between the input and the model's reconstruction.

        Args:
            data: Input numpy array with shape (batch_size, seq_len, n_features)
                  or (seq_len, n_features) for single sample

        Returns:
            Reconstruction error array with shape (n_samples,)
            where n_samples = batch_size * seq_len
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess data
        batch_x = self.preprocess(data)

        # Inference with no gradient computation
        with torch.no_grad():
            # Forward pass - reconstruct input
            outputs = self.model(batch_x, None, None, None)

            # Compute reconstruction error per sample
            # Shape: (batch, seq_len, features) -> (batch, seq_len)
            error = torch.mean(
                self.anomaly_criterion(batch_x, outputs),
                dim=-1
            )

            # Flatten to (n_samples,)
            error = error.reshape(-1)

            # Convert to numpy
            error_np = error.cpu().numpy()

        return error_np

    def predict(
        self,
        data: np.ndarray,
        threshold: float = 0.5,
        return_scores: bool = True,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform anomaly detection inference.

        Args:
            data: Input numpy array with shape (batch_size, seq_len, n_features)
                  or (seq_len, n_features) for single sample
            threshold: Anomaly score threshold for binary classification.
                       Samples with scores above threshold are labeled as anomalies.
            return_scores: Whether to return raw anomaly scores

        Returns:
            Dictionary containing:
                - "scores": np.ndarray of anomaly scores (if return_scores=True)
                - "labels": np.ndarray of binary labels (0=normal, 1=anomaly)
                - "threshold": float threshold used for classification
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get reconstruction errors as anomaly scores
        scores = self.get_reconstruction_error(data)

        # Apply threshold to get binary labels
        labels = (scores > threshold).astype(np.int32)

        # Build result dictionary
        result = {
            "labels": labels,
            "threshold": threshold,
        }

        if return_scores:
            result["scores"] = scores

        return result

    def predict_with_percentile_threshold(
        self,
        data: np.ndarray,
        anomaly_ratio: float = 1.0,
        return_scores: bool = True,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform anomaly detection using percentile-based threshold.

        This method computes the threshold based on the anomaly ratio,
        similar to the training evaluation approach.

        Args:
            data: Input numpy array
            anomaly_ratio: Expected percentage of anomalies (e.g., 1.0 means 1%)
            return_scores: Whether to return raw anomaly scores

        Returns:
            Dictionary containing scores, labels, and computed threshold
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get reconstruction errors
        scores = self.get_reconstruction_error(data)

        # Compute threshold using percentile
        threshold = np.percentile(scores, 100 - anomaly_ratio)

        # Apply threshold
        labels = (scores > threshold).astype(np.int32)

        result = {
            "labels": labels,
            "threshold": float(threshold),
        }

        if return_scores:
            result["scores"] = scores

        return result

    @property
    def seq_len(self) -> int:
        """Get the expected input sequence length."""
        return self.config.get("seq_len", 100)

    @property
    def n_features(self) -> int:
        """Get the expected number of input features."""
        return self.config.get("enc_in", 16)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"VoltageAnomalyDetector("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"status={status})"
        )
