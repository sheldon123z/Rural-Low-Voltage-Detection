# Models module for Voltage Anomaly Detection
from .Autoformer import Model as Autoformer
from .DLinear import Model as DLinear
from .FiLM import Model as FiLM
from .HybridTimesNet import Model as HybridTimesNet  # Hybrid Period Discovery TimesNet
from .Informer import Model as Informer
from .iTransformer import Model as iTransformer
from .KANAD import Model as KANAD
from .LightTS import Model as LightTS
from .MICN import Model as MICN
from .MTSTimesNet import Model as MTSTimesNet  # Multi-scale Temporal TimesNet
from .Nonstationary_Transformer import Model as Nonstationary_Transformer
from .PatchTST import Model as PatchTST
from .Reformer import Model as Reformer
from .SegRNN import Model as SegRNN
from .TimeMixer import Model as TimeMixer
from .TimesNet import Model as TimesNet

# Innovative TimesNet variants for Rural Voltage Anomaly Detection
from .TPATimesNet import Model as TPATimesNet  # Three-Phase Attention TimesNet
from .Transformer import Model as Transformer
from .VoltageTimesNet import Model as VoltageTimesNet

__all__ = [
    "TimesNet",
    "Transformer",
    "DLinear",
    "PatchTST",
    "iTransformer",
    "Autoformer",
    "Informer",
    "FiLM",
    "LightTS",
    "SegRNN",
    "KANAD",
    "Nonstationary_Transformer",
    "MICN",
    "TimeMixer",
    "Reformer",
    "VoltageTimesNet",
    # Innovative models
    "TPATimesNet",
    "MTSTimesNet",
    "HybridTimesNet",
    "model_dict",
    "get_model",
]

# Model registry for dynamic model selection
model_dict = {
    "TimesNet": TimesNet,
    "Transformer": Transformer,
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "Autoformer": Autoformer,
    "Informer": Informer,
    "FiLM": FiLM,
    "LightTS": LightTS,
    "SegRNN": SegRNN,
    "KANAD": KANAD,
    "Nonstationary_Transformer": Nonstationary_Transformer,
    "MICN": MICN,
    "TimeMixer": TimeMixer,
    "Reformer": Reformer,
    "VoltageTimesNet": VoltageTimesNet,
    # Innovative models for rural voltage anomaly detection
    "TPATimesNet": TPATimesNet,
    "MTSTimesNet": MTSTimesNet,
    "HybridTimesNet": HybridTimesNet,
}


def get_model(args):
    """
    Get model instance by args.

    Args:
        args: Namespace with model configuration, must have 'model' attribute

    Returns:
        model: Model instance
    """
    model_name = args.model
    if model_name not in model_dict:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(model_dict.keys())}"
        )
    Model = model_dict[model_name]
    return Model(args)
