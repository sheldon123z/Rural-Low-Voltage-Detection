"""
分析模块

本模块提供时序异常检测实验的结果管理、统计分析和报告生成功能。

Modules:
    result_manager: 实验结果管理器（时间戳聚类、日志解析）
    report_generator: 科学分析报告生成器
    statistical_analysis: 统计显著性检验

Author: Rural Voltage Detection Project
Date: 2026
"""

from .result_manager import ResultManager
from .report_generator import ReportGenerator

__all__ = ["ResultManager", "ReportGenerator"]
