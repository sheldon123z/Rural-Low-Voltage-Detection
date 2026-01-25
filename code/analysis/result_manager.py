"""
实验结果管理器

本模块提供实验结果的加载、管理和时间戳分组功能。

目录结构：
results/{dataset}_{experiment}_{YYYYMMDD_HHMMSS}/
├── *.log                             # 训练日志
└── analysis_{YYYYMMDD_HHMMSS}/       # 分析结果
    ├── figures/                      # 静态图表
    ├── interactive/                  # 交互式图表
    ├── 实验结果.json
    └── 实验分析报告.md

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
import re
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np


class ResultManager:
    """
    实验结果管理器

    负责实验结果的目录管理、日志解析和数据加载。

    Attributes:
        base_dir: 结果根目录
    """

    def __init__(self, base_dir: str = "./results"):
        """
        初始化结果管理器

        Args:
            base_dir: 结果根目录路径
        """
        self.base_dir = Path(base_dir)
        self.test_results_dir = self.base_dir.parent / "test_results"

    # ========================================================================
    # 目录管理
    # ========================================================================

    def create_experiment_dir(
        self,
        dataset: str,
        experiment: str = "comparison",
        use_timestamp: bool = True
    ) -> Path:
        """
        创建实验目录

        Args:
            dataset: 数据集名称
            experiment: 实验类型
            use_timestamp: 是否使用时间戳

        Returns:
            Path: 创建的目录路径
        """
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{dataset}_{experiment}_{timestamp}"
        else:
            dir_name = f"{dataset}_{experiment}"

        exp_dir = self.base_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        return exp_dir

    def create_analysis_dir(self, experiment_dir: Path) -> Path:
        """
        在实验目录下创建分析子目录

        Args:
            experiment_dir: 实验目录路径

        Returns:
            Path: 分析目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path(experiment_dir) / f"analysis_{timestamp}"

        # 创建子目录
        (analysis_dir / "figures").mkdir(parents=True, exist_ok=True)
        (analysis_dir / "interactive").mkdir(parents=True, exist_ok=True)

        return analysis_dir

    def find_latest_experiment(self, pattern: str = "*") -> Optional[Path]:
        """
        查找最新的实验目录

        Args:
            pattern: 匹配模式（如 "PSM_*"）

        Returns:
            Optional[Path]: 最新实验目录，未找到返回 None
        """
        dirs = sorted(
            self.base_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if dirs:
            return dirs[0]
        return None

    def list_experiments(self, dataset: str = None) -> List[Dict[str, Any]]:
        """
        列出所有实验

        Args:
            dataset: 可选的数据集过滤器

        Returns:
            List[Dict]: 实验信息列表
        """
        experiments = []

        pattern = f"{dataset}_*" if dataset else "*"
        for exp_dir in sorted(self.base_dir.glob(pattern)):
            if not exp_dir.is_dir():
                continue

            info = self._parse_experiment_dir(exp_dir)
            if info:
                experiments.append(info)

        return experiments

    def _parse_experiment_dir(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """解析实验目录名称"""
        name = exp_dir.name

        # 匹配格式: {dataset}_{type}_{timestamp}
        match = re.match(r"(\w+)_(\w+)_(\d{8}_\d{6})", name)
        if not match:
            return None

        dataset, exp_type, timestamp = match.groups()

        # 查找日志文件
        log_files = list(exp_dir.glob("*.log"))
        models = [f.stem for f in log_files]

        # 查找分析目录
        analysis_dirs = sorted(exp_dir.glob("analysis_*"))

        return {
            "id": name,
            "path": str(exp_dir),
            "dataset": dataset,
            "type": exp_type,
            "timestamp": timestamp,
            "created_at": datetime.strptime(timestamp, "%Y%m%d_%H%M%S"),
            "models": models,
            "analysis_runs": [d.name for d in analysis_dirs],
        }

    # ========================================================================
    # 日志解析
    # ========================================================================

    def parse_training_log(self, log_path: str) -> Dict[str, Any]:
        """
        解析训练日志文件

        Args:
            log_path: 日志文件路径

        Returns:
            Dict: 解析后的日志数据
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {log_path}")

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 移除 ANSI 转义码
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        content = ansi_escape.sub("", content)

        result = {
            "model": log_path.stem,
            "path": str(log_path),
            "args": {},
            "training_history": [],
            "final_metrics": {},
            "success": True,
        }

        # 解析参数
        args_match = re.search(r"Args in experiment:(.*?)(?=Epoch:|$)", content, re.DOTALL)
        if args_match:
            args_text = args_match.group(1)
            for line in args_text.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    result["args"][key.strip()] = value.strip()

        # 解析训练历史
        epoch_pattern = r"Epoch:\s*(\d+).*?Train Loss:\s*([\d.]+).*?Vali Loss:\s*([\d.]+).*?Test Loss:\s*([\d.]+)"
        for match in re.finditer(epoch_pattern, content):
            epoch, train_loss, vali_loss, test_loss = match.groups()
            result["training_history"].append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "test_loss": float(test_loss),
            })

        # 解析最终指标
        metrics_pattern = r"Accuracy:\s*([\d.]+).*?Precision:\s*([\d.]+).*?Recall:\s*([\d.]+).*?F1-score:\s*([\d.]+)"
        metrics_match = re.search(metrics_pattern, content)
        if metrics_match:
            acc, prec, rec, f1 = metrics_match.groups()
            result["final_metrics"] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
            }
        else:
            result["success"] = False

        return result

    def parse_all_logs(self, experiment_dir: str) -> Dict[str, Dict]:
        """
        解析实验目录下的所有日志

        Args:
            experiment_dir: 实验目录路径

        Returns:
            Dict: {模型名: 日志数据}
        """
        exp_dir = Path(experiment_dir)
        results = {}

        for log_file in sorted(exp_dir.glob("*.log")):
            try:
                data = self.parse_training_log(str(log_file))
                if data["success"]:
                    results[data["model"]] = data
            except Exception as e:
                print(f"解析日志失败 {log_file.name}: {e}")

        return results

    # ========================================================================
    # NPY 数据加载
    # ========================================================================

    def find_test_results_dir(self, model_name: str, dataset: str = "PSM") -> Optional[Path]:
        """
        查找模型的测试结果目录

        Args:
            model_name: 模型名称
            dataset: 数据集名称

        Returns:
            Optional[Path]: 测试结果目录路径
        """
        pattern = f"{dataset}_{model_name}_*"
        dirs = list(self.test_results_dir.glob(pattern))

        if dirs:
            # 返回最新的目录
            return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return None

    def load_npy_results(self, test_results_path: str) -> Dict[str, np.ndarray]:
        """
        加载 NPY 预测结果

        Args:
            test_results_path: 测试结果目录路径

        Returns:
            Dict: {文件名: numpy数组}
        """
        results_dir = Path(test_results_path)
        results = {}

        npy_files = ["pred.npy", "gt.npy", "test_energy.npy", "threshold.npy"]
        for npy_file in npy_files:
            file_path = results_dir / npy_file
            if file_path.exists():
                key = npy_file.replace(".npy", "")
                results[key] = np.load(str(file_path))

        return results

    def load_all_predictions(self, experiment_dir: str, dataset: str = "PSM") -> Dict[str, Dict]:
        """
        加载实验中所有模型的预测结果

        Args:
            experiment_dir: 实验目录路径
            dataset: 数据集名称

        Returns:
            Dict: {模型名: {pred, gt, test_energy, threshold}}
        """
        exp_dir = Path(experiment_dir)
        all_results = {}

        # 从日志文件获取模型列表
        for log_file in exp_dir.glob("*.log"):
            model_name = log_file.stem
            test_dir = self.find_test_results_dir(model_name, dataset)

            if test_dir:
                try:
                    all_results[model_name] = self.load_npy_results(str(test_dir))
                except Exception as e:
                    print(f"加载 {model_name} 预测结果失败: {e}")

        return all_results

    # ========================================================================
    # 综合加载
    # ========================================================================

    def load_experiment_results(
        self,
        experiment_dir: str,
        dataset: str = "PSM",
        load_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        加载实验的完整结果

        Args:
            experiment_dir: 实验目录路径
            dataset: 数据集名称
            load_predictions: 是否加载预测数据

        Returns:
            Dict: 完整的实验结果
        """
        exp_dir = Path(experiment_dir)

        results = {
            "experiment_dir": str(exp_dir),
            "dataset": dataset,
            "logs": {},
            "metrics": {},
            "training_history": {},
            "predictions": {},
        }

        # 解析日志
        results["logs"] = self.parse_all_logs(str(exp_dir))

        # 提取指标和训练历史
        for model, log_data in results["logs"].items():
            if log_data.get("final_metrics"):
                results["metrics"][model] = log_data["final_metrics"]
            if log_data.get("training_history"):
                results["training_history"][model] = log_data["training_history"]

        # 加载预测数据
        if load_predictions:
            results["predictions"] = self.load_all_predictions(str(exp_dir), dataset)

        return results

    # ========================================================================
    # 结果保存
    # ========================================================================

    def save_metrics_json(
        self,
        metrics: Dict[str, Dict],
        output_path: str,
        experiment_info: Dict = None
    ):
        """
        保存指标到 JSON 文件

        Args:
            metrics: 模型指标字典
            output_path: 输出文件路径
            experiment_info: 额外的实验信息
        """
        data = {
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "实验信息": experiment_info or {},
            "模型结果": {}
        }

        # 转换指标名称为中文
        metric_names = {
            "accuracy": "准确率",
            "precision": "精确率",
            "recall": "召回率",
            "f1_score": "F1分数",
        }

        for model, model_metrics in metrics.items():
            data["模型结果"][model] = {
                "指标": {metric_names.get(k, k): v for k, v in model_metrics.items()}
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_ranked_models(
        self,
        metrics: Dict[str, Dict],
        sort_by: str = "f1_score",
        ascending: bool = False
    ) -> List[Tuple[str, Dict]]:
        """
        获取按指标排序的模型列表

        Args:
            metrics: 模型指标字典
            sort_by: 排序依据的指标名
            ascending: 是否升序

        Returns:
            List[Tuple]: [(模型名, 指标字典), ...]
        """
        items = [(name, m) for name, m in metrics.items() if sort_by in m]
        items.sort(key=lambda x: x[1][sort_by], reverse=not ascending)
        return items


# ============================================================================
# 便捷函数
# ============================================================================

def load_experiment(experiment_dir: str, dataset: str = "PSM") -> Dict[str, Any]:
    """
    快速加载实验结果

    Args:
        experiment_dir: 实验目录路径
        dataset: 数据集名称

    Returns:
        Dict: 实验结果
    """
    manager = ResultManager(str(Path(experiment_dir).parent))
    return manager.load_experiment_results(experiment_dir, dataset)


def find_latest_experiment(base_dir: str = "./results", pattern: str = "*") -> Optional[str]:
    """
    查找最新实验目录

    Args:
        base_dir: 结果根目录
        pattern: 匹配模式

    Returns:
        Optional[str]: 最新实验目录路径
    """
    manager = ResultManager(base_dir)
    result = manager.find_latest_experiment(pattern)
    return str(result) if result else None
