import sys
import time
import importlib
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from core.config import FEATURE_COLUMNS, BEST_MODEL_PATH, BEST_MODEL_CONFIG_PATH

# 推理模块路径（延迟导入以避免 core 命名冲突）
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Rural-Low-Voltage-Detection/
_CODE_DIR = _PROJECT_ROOT / "code"
_DEMO_DIR = _CODE_DIR / "demo"

# HuggingFace 模型仓库信息
_HF_REPO_ID = "Sheldon123z/rural-voltage-detection-models"
# 按优先级尝试的文件名列表（从Optuna优化版到基础版）
_HF_FILENAMES = [
    "RuralVoltage/best_voltagetimesnet_v2.pth",
    "RuralVoltage/VoltageTimesNet_v2_sl50_dm128/checkpoint.pth",
    "RuralVoltage/VoltageTimesNet_v2_sl100_dm64/checkpoint.pth",
    "RuralVoltage/VoltageTimesNet_v2/checkpoint.pth",
]


def _try_download_model() -> Optional[Path]:
    """尝试从 HuggingFace 下载模型权重"""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        print(f"[DetectionService] 正在从 HuggingFace 下载模型: {_HF_REPO_ID}")

        # 先列出仓库文件，找到可用的权重文件
        try:
            repo_files = list(list_repo_files(_HF_REPO_ID))
            voltage_files = [f for f in repo_files if "VoltageTimesNet_v2" in f and f.endswith(".pth")]
            print(f"[DetectionService] 仓库中找到权重文件: {voltage_files}")
            download_candidates = voltage_files if voltage_files else _HF_FILENAMES
        except Exception:
            download_candidates = _HF_FILENAMES

        # 按优先级尝试下载
        for filename in download_candidates:
            try:
                print(f"[DetectionService] 尝试下载: {filename}")
                downloaded = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    filename=filename,
                    local_dir=str(BEST_MODEL_PATH.parent),
                    local_dir_use_symlinks=False,
                )
                # 复制到标准路径
                import shutil
                shutil.copy2(downloaded, str(BEST_MODEL_PATH))
                print(f"[DetectionService] 模型下载成功: {BEST_MODEL_PATH}")
                return BEST_MODEL_PATH
            except Exception as e:
                print(f"[DetectionService] 文件 {filename} 下载失败: {e}")
                continue

        print("[DetectionService] 所有候选文件下载失败")
        return None
    except ImportError:
        print("[DetectionService] huggingface_hub 未安装，跳过自动下载")
        return None
    except Exception as e:
        print(f"[DetectionService] 下载异常: {e}")
        return None


def _ensure_model_path() -> Optional[str]:
    """确保模型权重可用，返回实际路径或 None"""
    if BEST_MODEL_PATH.exists():
        return str(BEST_MODEL_PATH)
    print(f"[DetectionService] 本地模型文件不存在: {BEST_MODEL_PATH}")
    downloaded = _try_download_model()
    if downloaded and downloaded.exists():
        return str(downloaded)
    print("[DetectionService] 模型不可用，将使用统计方法进行检测（无深度学习模型）")
    return None


class DetectionService:
    """异常检测服务，封装推理模块"""

    def __init__(self):
        self._detector = None
        self._loaded_model = None
        self._model_available = None  # None=未知, True=可用, False=不可用

    def _get_detector(self):
        """懒加载检测器，自动处理模型下载"""
        if self._detector is None and self._model_available is not False:
            model_path = _ensure_model_path()
            if model_path is None:
                self._model_available = False
                return None

            # 动态导入 demo/core/inference.py 避免与 backend/core 冲突
            # 问题：inference.py 执行 "from models import model_dict"，
            # 但 sys.modules['models'] 已缓存了 backend/models/__init__.py（无 model_dict）。
            # 解决：先将 code/models 注册到 sys.modules['models']，再 exec_module。
            spec = importlib.util.spec_from_file_location(
                "demo_inference",
                str(_DEMO_DIR / "core" / "inference.py"),
            )
            inference_mod = importlib.util.module_from_spec(spec)

            # 将 code/ 添加到 sys.path 前面
            if str(_CODE_DIR) not in sys.path:
                sys.path.insert(0, str(_CODE_DIR))

            # 将 code/models 预先注册为 sys.modules['models']，覆盖 backend/models
            _code_models_spec = importlib.util.spec_from_file_location(
                "models",
                str(_CODE_DIR / "models" / "__init__.py"),
                submodule_search_locations=[str(_CODE_DIR / "models")],
            )
            _code_models_mod = importlib.util.module_from_spec(_code_models_spec)
            sys.modules["models"] = _code_models_mod
            _code_models_spec.loader.exec_module(_code_models_mod)

            spec.loader.exec_module(inference_mod)
            VoltageAnomalyDetector = inference_mod.VoltageAnomalyDetector
            try:
                self._detector = VoltageAnomalyDetector(
                    model_name="VoltageTimesNet_v2",
                    checkpoint_path=model_path,
                    device="cpu",
                    config_path=str(BEST_MODEL_CONFIG_PATH),
                )
                self._detector.load_model()
                self._loaded_model = "VoltageTimesNet"
                self._model_available = True
                print("[DetectionService] VoltageTimesNet 模型加载成功")
            except Exception as e:
                print(f"[DetectionService] 模型加载失败（权重与配置不匹配）: {e}")
                print("[DetectionService] 将使用统计方法进行检测")
                self._detector = None
                self._model_available = False
        return self._detector

    @property
    def model_available(self) -> bool:
        """检查深度学习模型是否可用"""
        if self._model_available is None:
            self._get_detector()
        return self._model_available is True

    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证CSV文件格式"""
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            return {"valid": False, "error": f"缺少特征列: {missing}"}
        if len(df) < 50:
            return {"valid": False, "error": f"数据行数不足（最少50行，当前{len(df)}行）"}
        return {"valid": True, "rows": len(df)}

    def _detect_with_statistics(
        self,
        df: pd.DataFrame,
        anomaly_ratio: float,
    ) -> Dict[str, Any]:
        """
        统计方法降级检测（模型不可用时使用）。
        使用 Z-score + 电压不平衡度 + 多特征综合评分。
        """
        start_time = time.time()
        data = df[FEATURE_COLUMNS].values.astype(np.float32)
        n_samples = len(data)

        # Z-score 异常分数
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        z_scores = np.abs((data - mean) / std)

        # 综合多特征：电压列权重更高
        voltage_cols = [FEATURE_COLUMNS.index(c) for c in ["Va", "Vb", "Vc"] if c in FEATURE_COLUMNS]
        imbalance_cols = [FEATURE_COLUMNS.index(c) for c in ["V_unbalance", "I_unbalance"] if c in FEATURE_COLUMNS]

        weights = np.ones(len(FEATURE_COLUMNS))
        for i in voltage_cols:
            weights[i] = 2.0
        for i in imbalance_cols:
            weights[i] = 3.0

        weighted_scores = (z_scores * weights).sum(axis=1) / weights.sum()

        # 平滑
        from scipy.ndimage import uniform_filter1d
        try:
            scores = uniform_filter1d(weighted_scores, size=5)
        except ImportError:
            scores = weighted_scores

        threshold = np.percentile(scores, 100 - anomaly_ratio)
        labels = (scores > threshold).astype(int)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "total_samples": n_samples,
            "anomaly_count": int(labels.sum()),
            "anomaly_rate": round(labels.sum() / n_samples * 100, 2),
            "threshold": float(threshold),
            "processing_time_ms": elapsed_ms,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "timestamps": list(range(n_samples)),
            "feature_data": {
                col: df[col].tolist() for col in ["Va", "Vb", "Vc", "Freq", "V_unbalance"]
                if col in df.columns
            },
            "method": "statistics",  # 标记使用了统计方法
        }

    def detect(
        self,
        df: pd.DataFrame,
        anomaly_ratio: float = 2.085,
    ) -> Dict[str, Any]:
        """执行异常检测（优先使用深度学习模型，降级使用统计方法）"""
        start_time = time.time()

        detector = self._get_detector()

        if detector is None:
            # 深度学习模型不可用，使用统计方法
            result = self._detect_with_statistics(df, anomaly_ratio)
            result["model_note"] = "使用统计方法（模型权重加载中或不可用）"
            return result

        seq_len = detector.seq_len  # 50

        # 提取特征列并转换为 numpy
        data = df[FEATURE_COLUMNS].values.astype(np.float32)

        # 标准化（z-score per column）
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_normalized = (data - mean) / std

        # 滑动窗口推理
        n_samples = len(data_normalized)
        scores = np.zeros(n_samples)
        counts = np.zeros(n_samples)

        for start in range(0, n_samples - seq_len + 1, seq_len // 2):
            end = start + seq_len
            window = data_normalized[start:end]
            result = detector.predict_with_percentile_threshold(
                window, anomaly_ratio=anomaly_ratio
            )
            window_scores = result["scores"]
            scores[start:end] += window_scores
            counts[start:end] += 1

        # 归一化重叠分数
        mask = counts > 0
        scores[mask] /= counts[mask]

        # 应用阈值
        threshold = np.percentile(scores, 100 - anomaly_ratio)
        labels = (scores > threshold).astype(int)

        elapsed_ms = int((time.time() - start_time) * 1000)
        anomaly_count = int(labels.sum())

        return {
            "total_samples": n_samples,
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / n_samples * 100, 2),
            "threshold": float(threshold),
            "processing_time_ms": elapsed_ms,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "timestamps": list(range(n_samples)),
            "feature_data": {
                col: df[col].tolist() for col in ["Va", "Vb", "Vc", "Freq", "V_unbalance"]
                if col in df.columns
            },
            "method": "deep_learning",
        }


# 全局单例
_detection_service = None

def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
