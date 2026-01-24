# Models module for Voltage Anomaly Detection
from .TimesNet import Model as TimesNet
from .Transformer import Model as Transformer
from .DLinear import Model as DLinear
from .PatchTST import Model as PatchTST
from .iTransformer import Model as iTransformer
from .Autoformer import Model as Autoformer
from .Informer import Model as Informer
from .FiLM import Model as FiLM
from .LightTS import Model as LightTS
from .SegRNN import Model as SegRNN
from .KANAD import Model as KANAD
from .Nonstationary_Transformer import Model as Nonstationary_Transformer
from .MICN import Model as MICN
from .TimeMixer import Model as TimeMixer
from .Reformer import Model as Reformer
from .VoltageTimesNet import Model as VoltageTimesNet

# Innovative TimesNet variants for Rural Voltage Anomaly Detection
from .TPATimesNet import Model as TPATimesNet  # Three-Phase Attention TimesNet
from .MTSTimesNet import Model as MTSTimesNet  # Multi-scale Temporal TimesNet
from .HybridTimesNet import Model as HybridTimesNet  # Hybrid Period Discovery TimesNet

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
