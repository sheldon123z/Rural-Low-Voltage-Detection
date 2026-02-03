"""
Model Loader Module for Gradio Demo
农村低压配电网电压异常检测项目

Provides:
- load_model(): Load a model with optional checkpoint
- get_available_models(): List available models
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from argparse import Namespace

import torch

# Add code directory to path for importing models
CODE_DIR = Path(__file__).parent.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import model_dict


# Default model configurations for anomaly detection
DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "VoltageTimesNet_v2": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 0,
        "label_len": 0,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
    },
    "VoltageTimesNet": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 0,
        "label_len": 0,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
    },
    "TimesNet": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 0,
        "label_len": 0,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
    },
    "TPATimesNet": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 0,
        "label_len": 0,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
    },
    "MTSTimesNet": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 0,
        "label_len": 0,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
    },
    "DLinear": {
        "task_name": "anomaly_detection",
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "pred_len": 100,  # DLinear requires pred_len = seq_len for anomaly detection
        "individual": False,
        "moving_avg": 25,
    },
}

# Models suitable for voltage anomaly detection demo
DEMO_MODELS = [
    "VoltageTimesNet_v2",
    "VoltageTimesNet",
    "TimesNet",
    "TPATimesNet",
    "MTSTimesNet",
    "DLinear",
]


def get_available_models() -> List[str]:
    """
    Get list of available models for the demo.

    Returns:
        List of model names that can be loaded
    """
    return [m for m in DEMO_MODELS if m in model_dict]


def get_all_models() -> List[str]:
    """
    Get list of all registered models.

    Returns:
        List of all model names in model_dict
    """
    return list(model_dict.keys())


def create_model_config(
    model_name: str,
    config_override: Optional[Dict[str, Any]] = None
) -> Namespace:
    """
    Create model configuration namespace.

    Args:
        model_name: Name of the model
        config_override: Optional dict to override default config

    Returns:
        Namespace object with model configuration
    """
    # Get default config or create minimal config
    if model_name in DEFAULT_MODEL_CONFIGS:
        config = DEFAULT_MODEL_CONFIGS[model_name].copy()
    else:
        # Minimal config for unknown models
        config = {
            "task_name": "anomaly_detection",
            "enc_in": 16,
            "c_out": 16,
            "seq_len": 100,
            "pred_len": 0,
            "label_len": 0,
            "d_model": 64,
            "d_ff": 64,
            "e_layers": 2,
            "top_k": 5,
            "num_kernels": 6,
            "embed": "timeF",
            "freq": "h",
            "dropout": 0.1,
        }

    # Apply overrides
    if config_override:
        config.update(config_override)

    return Namespace(**config)


def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load a model with optional checkpoint.

    Args:
        model_name: Name of the model (e.g., "VoltageTimesNet_v2", "TimesNet")
        checkpoint_path: Optional path to model checkpoint (.pth file)
        config_override: Optional dict to override default model config
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Loaded model in eval mode

    Raises:
        ValueError: If model_name is not found
        FileNotFoundError: If checkpoint_path doesn't exist

    Example:
        >>> model = load_model("VoltageTimesNet_v2")
        >>> model = load_model("TimesNet", checkpoint_path="./best_model.pth")
        >>> model = load_model("TimesNet", config_override={"seq_len": 50})
    """
    # Validate model name
    if model_name not in model_dict:
        available = get_available_models()
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {available}"
        )

    # Create config
    config = create_model_config(model_name, config_override)

    # Build model
    Model = model_dict[model_name]
    model = Model(config)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Load state dict with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from: {checkpoint_path}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Dict with model information
    """
    if model_name not in model_dict:
        return {"error": f"Model '{model_name}' not found"}

    config = DEFAULT_MODEL_CONFIGS.get(model_name, {})

    info = {
        "name": model_name,
        "available": True,
        "config": config,
        "description": _get_model_description(model_name),
    }

    return info


def _get_model_description(model_name: str) -> str:
    """Get model description."""
    descriptions = {
        "VoltageTimesNet_v2": "Enhanced TimesNet with recall optimization for voltage anomaly detection",
        "VoltageTimesNet": "TimesNet variant with preset periods for voltage patterns",
        "TimesNet": "FFT-based period discovery with 2D convolution for temporal patterns",
        "TPATimesNet": "Three-Phase Attention TimesNet for multi-phase voltage analysis",
        "MTSTimesNet": "Multi-scale Temporal TimesNet for multi-resolution patterns",
        "DLinear": "Lightweight linear model with trend-seasonal decomposition",
    }
    return descriptions.get(model_name, "No description available")


if __name__ == "__main__":
    # Test module
    print("Available models:", get_available_models())

    for model_name in get_available_models():
        print(f"\nLoading {model_name}...")
        try:
            model = load_model(model_name)
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"  - Parameters: {params:,}")
            print(f"  - Config: {get_model_info(model_name)['config']}")
        except Exception as e:
            print(f"  - Error: {e}")
