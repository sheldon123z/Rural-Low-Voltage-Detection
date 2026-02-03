"""
自定义检测标签页
农村低压配电网电压异常检测项目

本模块实现用户自定义数据的异常检测功能：
- CSV 文件上传
- 数据预览
- 模型选择和阈值调节
- 实时推理 (CPU)
- 检测结果可视化

Author: Rural Voltage Detection Project
Date: 2026
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import gradio as gr
import numpy as np
import pandas as pd

# Add parent directories to path for imports
DEMO_DIR = Path(__file__).parent.parent
CODE_DIR = DEMO_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# Import core modules
from core.inference import VoltageAnomalyDetector
from core.data_processor import DataProcessor
from visualization.detection_plots import (
    create_detection_timeline,
    create_score_distribution,
)
from config import (
    MODEL_DIR,
    INFERENCE_CONFIG,
    DEMO_DATA_CONFIG,
)


# ============================================================================
# 全局变量和配置
# ============================================================================

# 可用模型列表
AVAILABLE_MODELS = ["VoltageTimesNet_v2", "TimesNet", "DLinear"]

# 模型描述
MODEL_DESCRIPTIONS = {
    "VoltageTimesNet_v2": "推荐模型: 基于 TimesNet 的改进版本，针对召回率进行优化，适合农村电压异常检测",
    "TimesNet": "基线模型: 使用 FFT 发现周期性，结合 2D 卷积捕获时序模式",
    "DLinear": "轻量级模型: 基于线性分解的简单模型，速度快但精度略低",
}

# 检测器缓存
_detector_cache: Dict[str, VoltageAnomalyDetector] = {}


# ============================================================================
# 辅助函数
# ============================================================================

def get_detector(model_name: str) -> VoltageAnomalyDetector:
    """
    获取或创建检测器实例（带缓存）

    Args:
        model_name: 模型名称

    Returns:
        VoltageAnomalyDetector 实例
    """
    if model_name not in _detector_cache:
        # 查找模型检查点
        checkpoint_path = None
        model_file = MODEL_DIR / f"best_{model_name.lower()}.pth"
        if model_file.exists():
            checkpoint_path = str(model_file)

        # 创建检测器
        detector = VoltageAnomalyDetector(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=INFERENCE_CONFIG.get("device", "cpu"),
        )
        _detector_cache[model_name] = detector

    return _detector_cache[model_name]


def validate_csv_file(file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    验证 CSV 文件

    Args:
        file_path: CSV 文件路径

    Returns:
        (是否有效, 消息, DataFrame 或 None)
    """
    try:
        if file_path is None:
            return False, "请先上传 CSV 文件", None

        # 读取文件
        df = pd.read_csv(file_path)

        if df.empty:
            return False, "CSV 文件为空", None

        # 检查数据长度
        min_length = DEMO_DATA_CONFIG.get("window_size", 100) + 10
        if len(df) < min_length:
            return False, f"数据长度不足，至少需要 {min_length} 行数据（当前: {len(df)} 行）", None

        # 检查是否有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return False, "CSV 文件中没有找到数值列", None

        return True, f"文件验证成功: {len(df)} 行, {len(numeric_cols)} 个数值特征", df

    except pd.errors.EmptyDataError:
        return False, "CSV 文件为空或格式错误", None
    except pd.errors.ParserError as e:
        return False, f"CSV 解析错误: {str(e)}", None
    except Exception as e:
        return False, f"读取文件失败: {str(e)}", None


def format_preview_df(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    """
    格式化预览 DataFrame

    Args:
        df: 原始 DataFrame
        max_rows: 最大显示行数

    Returns:
        格式化后的 DataFrame
    """
    # 选择数值列
    exclude_cols = ["timestamp", "date", "time", "label", "index", "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 截取前 N 行
    preview_df = df[feature_cols].head(max_rows).copy()

    # 四舍五入数值
    for col in preview_df.columns:
        if preview_df[col].dtype in [np.float64, np.float32]:
            preview_df[col] = preview_df[col].round(4)

    return preview_df


def generate_detection_stats(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    total_samples: int,
) -> str:
    """
    生成检测统计信息

    Args:
        labels: 预测标签
        scores: 异常分数
        threshold: 阈值
        total_samples: 原始数据样本数

    Returns:
        Markdown 格式的统计信息
    """
    n_anomaly = int(np.sum(labels))
    n_normal = len(labels) - n_anomaly
    anomaly_ratio = n_anomaly / len(labels) * 100 if len(labels) > 0 else 0

    # 分数统计
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_max = float(np.max(scores))
    score_min = float(np.min(scores))

    stats_md = f"""
### 检测统计

| 指标 | 数值 |
|------|------|
| 原始数据样本数 | {total_samples} |
| 检测窗口数 | {len(labels)} |
| 检测到的异常 | {n_anomaly} ({anomaly_ratio:.2f}%) |
| 正常样本 | {n_normal} ({100-anomaly_ratio:.2f}%) |
| 使用阈值 | {threshold:.4f} |

### 异常分数统计

| 指标 | 数值 |
|------|------|
| 平均分数 | {score_mean:.4f} |
| 标准差 | {score_std:.4f} |
| 最大分数 | {score_max:.4f} |
| 最小分数 | {score_min:.4f} |

> 注: 异常分数表示重构误差，分数越高表示越可能是异常。
"""
    return stats_md


# ============================================================================
# 事件处理函数
# ============================================================================

def handle_file_upload(file) -> Tuple[str, pd.DataFrame, str]:
    """
    处理文件上传事件

    Args:
        file: 上传的文件对象

    Returns:
        (状态消息, 预览 DataFrame, 详细信息)
    """
    if file is None:
        return (
            "等待上传文件...",
            pd.DataFrame(),
            "请上传 CSV 格式的时序数据文件",
        )

    # 验证文件
    is_valid, message, df = validate_csv_file(file.name)

    if not is_valid:
        return (
            f"文件验证失败: {message}",
            pd.DataFrame(),
            message,
        )

    # 生成预览
    preview_df = format_preview_df(df, max_rows=10)

    # 生成详细信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["timestamp", "date", "time", "label", "index", "Unnamed: 0"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    detail_info = f"""
**文件信息:**
- 文件名: {Path(file.name).name}
- 数据行数: {len(df)}
- 特征数量: {len(feature_cols)}
- 特征列表: {', '.join(feature_cols[:8])}{'...' if len(feature_cols) > 8 else ''}
"""

    return (
        f"文件验证成功: {len(df)} 行数据",
        preview_df,
        detail_info,
    )


def handle_model_change(model_name: str) -> str:
    """
    处理模型选择变更

    Args:
        model_name: 选择的模型名称

    Returns:
        模型描述信息
    """
    desc = MODEL_DESCRIPTIONS.get(model_name, "未知模型")
    return f"**{model_name}**: {desc}"


def run_detection(
    file,
    model_name: str,
    threshold: float,
    progress=gr.Progress(),
) -> Tuple[Any, Any, str]:
    """
    执行异常检测

    Args:
        file: 上传的文件对象
        model_name: 模型名称
        threshold: 异常阈值
        progress: Gradio 进度条

    Returns:
        (时间线图, 分布图, 统计信息)
    """
    # 验证输入
    if file is None:
        return None, None, "请先上传 CSV 文件"

    try:
        progress(0, desc="验证文件...")
        is_valid, message, df = validate_csv_file(file.name)

        if not is_valid:
            return None, None, f"文件验证失败: {message}"

        progress(0.1, desc="加载数据...")

        # 提取特征数据
        exclude_cols = ["timestamp", "date", "time", "label", "index", "Unnamed: 0"]
        feature_cols = [c for c in df.columns if c not in exclude_cols
                       and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if len(feature_cols) == 0:
            return None, None, "未找到有效的数值特征列"

        data = df[feature_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        total_samples = len(data)

        progress(0.2, desc="创建数据窗口...")

        # 创建数据处理器
        window_size = DEMO_DATA_CONFIG.get("window_size", 100)
        step_size = DEMO_DATA_CONFIG.get("step_size", 1)

        processor = DataProcessor(
            seq_len=window_size,
            step=step_size,
            normalize=True,
        )

        # 处理数据
        windows = processor.fit_transform(data)
        n_windows = windows.shape[0]

        progress(0.3, desc=f"加载模型 {model_name}...")

        # 获取检测器
        detector = get_detector(model_name)

        # 加载模型（如果尚未加载）
        if not detector.is_loaded:
            detector.load_model()

        progress(0.5, desc="执行推理...")

        # 执行检测
        results = detector.predict(windows, threshold=threshold)

        scores = results["scores"]
        labels = results["labels"]

        progress(0.8, desc="生成可视化...")

        # 准备可视化数据
        # 使用原始数据的前 N 个点（与 scores 长度对齐）
        vis_data = data[:len(scores), :min(3, data.shape[1])]
        vis_feature_names = feature_cols[:min(3, len(feature_cols))]

        # 创建时间线图
        timeline_fig = create_detection_timeline(
            data=vis_data,
            scores=scores,
            labels=labels,
            threshold=threshold,
            feature_names=vis_feature_names,
            max_features=3,
            title="异常检测结果时间线",
        )

        progress(0.9, desc="生成统计信息...")

        # 创建分数分布图（使用预测标签）
        dist_fig = create_score_distribution(
            scores=scores,
            labels=labels,
            threshold=threshold,
            bins=50,
            title="异常分数分布",
        )

        # 生成统计信息
        stats_md = generate_detection_stats(
            labels=labels,
            scores=scores,
            threshold=threshold,
            total_samples=total_samples,
        )

        progress(1.0, desc="完成!")

        return timeline_fig, dist_fig, stats_md

    except ValueError as e:
        return None, None, f"数据处理错误: {str(e)}"
    except RuntimeError as e:
        return None, None, f"模型推理错误: {str(e)}"
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, f"检测失败: {str(e)}\n\n详细信息:\n```\n{error_detail}\n```"


# ============================================================================
# 创建标签页
# ============================================================================

def create_detection_tab():
    """
    创建自定义检测标签页

    Returns:
        Gradio Tab 组件
    """
    with gr.Tab("自定义检测") as tab:
        gr.Markdown("""
        ## 自定义数据异常检测

        上传您自己的时序数据文件，使用训练好的模型进行异常检测。

        **使用说明:**
        1. 上传 CSV 格式的时序数据文件
        2. 选择检测模型
        3. 调整异常阈值
        4. 点击「开始检测」按钮

        > 提示: 数据文件应包含数值型特征列，至少需要 100+ 行数据。
        """)

        with gr.Row():
            # 左侧: 输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 数据上传")

                # 文件上传
                file_input = gr.File(
                    label="上传 CSV 文件",
                    file_types=[".csv"],
                    file_count="single",
                )

                # 文件状态
                file_status = gr.Textbox(
                    label="文件状态",
                    value="等待上传文件...",
                    interactive=False,
                    lines=1,
                )

                # 文件详情
                file_detail = gr.Markdown(
                    value="请上传 CSV 格式的时序数据文件"
                )

                gr.Markdown("### 模型配置")

                # 模型选择
                model_selector = gr.Radio(
                    choices=AVAILABLE_MODELS,
                    value="VoltageTimesNet_v2",
                    label="选择模型",
                )

                # 模型描述
                model_desc = gr.Markdown(
                    value=f"**VoltageTimesNet_v2**: {MODEL_DESCRIPTIONS['VoltageTimesNet_v2']}"
                )

                # 阈值滑块
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="异常阈值",
                    info="阈值越低，检测越敏感（检出更多异常）；阈值越高，检测越保守",
                )

                # 检测按钮
                detect_btn = gr.Button(
                    "开始检测",
                    variant="primary",
                    size="lg",
                )

            # 右侧: 数据预览
            with gr.Column(scale=1):
                gr.Markdown("### 数据预览 (前 10 行)")

                preview_table = gr.Dataframe(
                    label="数据预览",
                    headers=None,
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("---")
        gr.Markdown("### 检测结果")

        # 结果区域
        with gr.Row():
            # 时间线图
            with gr.Column(scale=2):
                timeline_plot = gr.Plot(
                    label="异常检测时间线",
                )

        with gr.Row():
            # 分布图
            with gr.Column(scale=1):
                dist_plot = gr.Plot(
                    label="异常分数分布",
                )

            # 统计信息
            with gr.Column(scale=1):
                stats_output = gr.Markdown(
                    value="等待检测...",
                    label="检测统计",
                )

        # ====================================================================
        # 事件绑定
        # ====================================================================

        # 文件上传事件
        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[file_status, preview_table, file_detail],
        )

        # 模型选择事件
        model_selector.change(
            fn=handle_model_change,
            inputs=[model_selector],
            outputs=[model_desc],
        )

        # 检测按钮事件
        detect_btn.click(
            fn=run_detection,
            inputs=[file_input, model_selector, threshold_slider],
            outputs=[timeline_plot, dist_plot, stats_output],
        )

    return tab


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 创建测试应用
    with gr.Blocks(title="自定义检测测试", theme="soft") as demo:
        create_detection_tab()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
    )
