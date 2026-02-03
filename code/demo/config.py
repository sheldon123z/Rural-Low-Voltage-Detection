"""
Gradio Demo 配置文件
农村低压配电网电压异常检测项目
"""

import os
from pathlib import Path

# 路径配置
DEMO_DIR = Path(__file__).parent
CODE_DIR = DEMO_DIR.parent
PROJECT_DIR = CODE_DIR.parent

# 模型路径
MODEL_DIR = CODE_DIR / "newest_models"
BEST_MODEL_PATH = MODEL_DIR / "best_voltagetimesnet_v2.pth"
MODEL_CONFIG_PATH = MODEL_DIR / "best_model_config.json"

# 数据路径
DATASET_DIR = CODE_DIR / "dataset"
RURAL_VOLTAGE_DIR = DATASET_DIR / "RuralVoltage" / "realistic_v2"
PSM_DIR = DATASET_DIR / "PSM"

# 预计算数据路径
PRECOMPUTED_DIR = DEMO_DIR / "precomputed"

# 模型配置
MODEL_CONFIGS = {
    "VoltageTimesNet_v2": {
        "enc_in": 16,
        "c_out": 16,
        "seq_len": 100,
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
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
    },
    "DLinear": {
        "enc_in": 16,
        "seq_len": 100,
        "pred_len": 100,
        "individual": False,
    },
}

# 可视化配色方案 (柔和科研风格)
THESIS_COLORS = {
    "primary": "#4878A8",      # 柔和蓝
    "secondary": "#72A86D",    # 柔和绿
    "accent": "#C4785C",       # 柔和橙
    "warning": "#D4A84C",      # 柔和黄
    "neutral": "#808080",      # 中性灰
    "light_gray": "#B0B0B0",   # 浅灰
    "anomaly": "#E74C3C",      # 异常红
    "normal": "#2ECC71",       # 正常绿
}

# 模型对比颜色
MODEL_COLORS = {
    "VoltageTimesNet_v2": "#4878A8",
    "VoltageTimesNet": "#72A86D",
    "TimesNet": "#C4785C",
    "TPATimesNet": "#D4A84C",
    "MTSTimesNet": "#9B59B6",
    "DLinear": "#808080",
}

# Gradio 主题配置
GRADIO_THEME = "soft"

# 中文字体配置
FONT_CONFIG = {
    "chinese": ["Noto Serif CJK JP", "SimSun", "Microsoft YaHei"],
    "english": ["Times New Roman", "serif"],
}

# 推理配置
INFERENCE_CONFIG = {
    "batch_size": 32,
    "device": "cpu",
    "default_threshold": 0.5,
}

# 演示数据配置
DEMO_DATA_CONFIG = {
    "sample_length": 1000,
    "window_size": 100,
    "step_size": 1,
}
