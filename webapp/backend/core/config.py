from pathlib import Path
import json

# 项目根路径
# Path(__file__) = webapp/backend/core/config.py
# .parent = webapp/backend/core/
# .parent.parent = webapp/backend/
# .parent.parent.parent = webapp/
# .parent.parent.parent.parent = Rural-Low-Voltage-Detection/
BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent  # Rural-Low-Voltage-Detection/
CODE_DIR = PROJECT_ROOT / "code"

# 模型相关路径
MODEL_DIR = CODE_DIR / "newest_models"
BEST_MODEL_PATH = MODEL_DIR / "best_voltagetimesnet_v2.pth"
BEST_MODEL_CONFIG_PATH = MODEL_DIR / "best_model_config.json"

# 数据集路径
DATASET_DIR = CODE_DIR / "dataset"
RURAL_VOLTAGE_DIR = DATASET_DIR / "RuralVoltage" / "realistic_v2"

# 上传和结果目录
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 数据库
DATABASE_URL = f"sqlite:///{BACKEND_DIR}/detection.db"

# RuralVoltage 特征列
FEATURE_COLUMNS = [
    "Va", "Vb", "Vc", "Ia", "Ib", "Ic",
    "P", "Q", "S", "PF",
    "THD_Va", "THD_Vb", "THD_Vc",
    "Freq", "V_unbalance", "I_unbalance"
]

# 模型性能指标（预计算，用于模型对比页面）
MODEL_METRICS = {
    "VoltageTimesNet": {
        "display_name": "VoltageTimesNet (本论文)",
        "accuracy": 0.9393, "precision": 0.7371,
        "recall": 0.9110, "f1": 0.8149,
        "description": "本论文提出的核心模型，融合预设电气周期与FFT自适应发现，专为农村三相电压设计。",
        "is_primary": True,
        "model_key": "VoltageTimesNet_v2",
        "checkpoint": str(BEST_MODEL_PATH),
    },
    "TimesNet": {
        "display_name": "TimesNet (基线)",
        "accuracy": 0.8584, "precision": 0.5143,
        "recall": 0.7115, "f1": 0.5970,
        "description": "原始 TimesNet，纯 FFT 周期发现，未针对电压数据优化。",
        "is_primary": False,
        "model_key": "TimesNet",
        "checkpoint": None,
    },
    "LSTMAutoEncoder": {
        "display_name": "LSTM AutoEncoder",
        "accuracy": 0.7905, "precision": 0.3654,
        "recall": 0.5712, "f1": 0.4457,
        "description": "基于LSTM的自编码器，传统深度学习基线方法。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
    "IsolationForest": {
        "display_name": "Isolation Forest",
        "accuracy": 0.3474, "precision": 0.3474,
        "recall": 1.0000, "f1": 0.5157,
        "description": "经典无监督异常检测，召回率高但精确率低。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
    "OneClassSVM": {
        "display_name": "One-Class SVM",
        "accuracy": 0.3474, "precision": 0.3474,
        "recall": 1.0000, "f1": 0.5157,
        "description": "支持向量机单类分类，适用于小样本场景。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
}

# API配置
API_V1_PREFIX = "/api/v1"
APP_TITLE = "农村低电压检测平台"
APP_VERSION = "1.0.0"
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
