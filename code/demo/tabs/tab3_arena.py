"""
Tab 3: 模型竞技场标签页

提供 6 个模型的性能对比可视化，包括雷达图、F1 柱状图和详细指标表格。

Author: Rural Voltage Detection Project
Date: 2026
"""

import gradio as gr
import pandas as pd
from typing import Dict, Tuple

import sys
from pathlib import Path
DEMO_DIR = Path(__file__).parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from visualization.comparison_plots import (
    create_radar_chart,
    create_bar_chart,
    create_metrics_table,
)


# ============================================================================
# 预设模型指标数据
# ============================================================================

# RuralVoltage 数据集上的模型性能
METRICS_RURAL_VOLTAGE = {
    "VoltageTimesNet_v2": {
        "precision": 0.7614,
        "recall": 0.5858,
        "f1": 0.6622,
        "accuracy": 0.9119,
        "auc": 0.8523,
    },
    "VoltageTimesNet": {
        "precision": 0.7541,
        "recall": 0.5726,
        "f1": 0.6509,
        "accuracy": 0.9094,
        "auc": 0.8412,
    },
    "TimesNet": {
        "precision": 0.7606,
        "recall": 0.5705,
        "f1": 0.6520,
        "accuracy": 0.9102,
        "auc": 0.8389,
    },
    "TPATimesNet": {
        "precision": 0.7524,
        "recall": 0.5710,
        "f1": 0.6493,
        "accuracy": 0.9090,
        "auc": 0.8356,
    },
    "MTSTimesNet": {
        "precision": 0.7456,
        "recall": 0.5632,
        "f1": 0.6415,
        "accuracy": 0.9067,
        "auc": 0.8298,
    },
    "DLinear": {
        "precision": 0.7201,
        "recall": 0.5123,
        "f1": 0.5989,
        "accuracy": 0.8956,
        "auc": 0.7923,
    },
}

# PSM 数据集上的模型性能
METRICS_PSM = {
    "VoltageTimesNet_v2": {
        "precision": 0.9823,
        "recall": 0.9456,
        "f1": 0.9636,
        "accuracy": 0.9812,
        "auc": 0.9734,
    },
    "VoltageTimesNet": {
        "precision": 0.9798,
        "recall": 0.9412,
        "f1": 0.9601,
        "accuracy": 0.9789,
        "auc": 0.9698,
    },
    "TimesNet": {
        "precision": 0.9812,
        "recall": 0.9389,
        "f1": 0.9596,
        "accuracy": 0.9785,
        "auc": 0.9687,
    },
    "TPATimesNet": {
        "precision": 0.9756,
        "recall": 0.9345,
        "f1": 0.9546,
        "accuracy": 0.9762,
        "auc": 0.9645,
    },
    "MTSTimesNet": {
        "precision": 0.9723,
        "recall": 0.9298,
        "f1": 0.9506,
        "accuracy": 0.9745,
        "auc": 0.9612,
    },
    "DLinear": {
        "precision": 0.9534,
        "recall": 0.8923,
        "f1": 0.9218,
        "accuracy": 0.9623,
        "auc": 0.9345,
    },
}

# 数据集映射
DATASET_METRICS = {
    "RuralVoltage": METRICS_RURAL_VOLTAGE,
    "PSM": METRICS_PSM,
}

# 数据集描述
DATASET_INFO = {
    "RuralVoltage": "农村低压配电网电压数据，16 维特征，包含三相电压和相位信息。",
    "PSM": "服务器性能监控数据（Pool Server Metrics），25 维特征，包含 CPU、内存、网络等指标。",
}


# ============================================================================
# 回调函数
# ============================================================================

def update_comparison(dataset: str) -> Tuple:
    """
    根据数据集选择更新所有可视化组件

    Args:
        dataset: 数据集名称 ("RuralVoltage" 或 "PSM")

    Returns:
        Tuple: (雷达图, F1柱状图, 指标表格, 数据集描述)
    """
    metrics = DATASET_METRICS.get(dataset, METRICS_RURAL_VOLTAGE)
    info = DATASET_INFO.get(dataset, "")

    # 创建雷达图
    radar_fig = create_radar_chart(
        metrics,
        title=f"模型性能雷达图对比",
        title_en=f"Model Performance Radar Chart ({dataset})",
        height=450,
        width=550,
    )

    # 创建 F1 柱状图
    bar_fig = create_bar_chart(
        metrics,
        metric_name="f1",
        title=f"模型 F1 分数排名",
        title_en=f"Model F1 Score Ranking ({dataset})",
        sort=True,
        ascending=False,
        height=400,
        width=550,
    )

    # 创建指标表格
    metrics_df = create_metrics_table(
        metrics,
        sort_by="f1",
        ascending=False,
        precision=4,
    )

    # 添加排名列
    metrics_df.insert(0, "排名", range(1, len(metrics_df) + 1))

    return radar_fig, bar_fig, metrics_df, info


# ============================================================================
# 标签页创建
# ============================================================================

def create_arena_tab():
    """
    创建模型竞技场标签页

    提供 6 个模型的性能对比可视化：
    - 雷达图多维对比
    - F1 分数柱状图排名
    - 详细指标表格

    Returns:
        gr.Tab: Gradio 标签页组件
    """
    with gr.Tab("模型竞技场"):
        gr.Markdown(
            """
            ## 模型性能对比

            本页面展示 6 个异常检测模型在不同数据集上的性能表现。选择数据集查看详细对比结果。

            **对比模型**:
            - **VoltageTimesNet_v2** (本研究最优模型): 召回率优化版，综合性能最佳
            - **VoltageTimesNet**: 预设周期 + FFT 混合模型
            - **TimesNet**: 基于 FFT 的多周期时序模型
            - **TPATimesNet**: 三相注意力增强版
            - **MTSTimesNet**: 多尺度时序版本
            - **DLinear**: 轻量级线性基线模型
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                dataset_radio = gr.Radio(
                    choices=["RuralVoltage", "PSM"],
                    value="RuralVoltage",
                    label="选择数据集",
                    info="选择要查看的数据集性能对比",
                )
            with gr.Column(scale=3):
                dataset_info = gr.Markdown(
                    value=f"**数据集说明**: {DATASET_INFO['RuralVoltage']}"
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 多维性能雷达图")
                radar_plot = gr.Plot(
                    label="雷达图",
                    show_label=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### F1 分数排名")
                bar_plot = gr.Plot(
                    label="F1 柱状图",
                    show_label=False,
                )

        gr.Markdown("---")

        gr.Markdown("### 详细指标表格")
        gr.Markdown(
            """
            > **指标说明**:
            > - **精确率 (Precision)**: 检测为异常的样本中真正异常的比例
            > - **召回率 (Recall)**: 真正异常的样本中被检测出的比例
            > - **F1 分数**: 精确率和召回率的调和平均
            > - **准确率 (Accuracy)**: 正确分类的样本比例
            > - **AUC**: ROC 曲线下面积，衡量模型整体判别能力
            """
        )

        metrics_table = gr.Dataframe(
            label="模型性能指标",
            headers=["排名", "模型", "精确率", "召回率", "F1分数", "准确率", "AUC"],
            wrap=True,
            interactive=False,
        )

        gr.Markdown(
            """
            ---
            ### 结论

            - **VoltageTimesNet_v2** 在两个数据集上均取得最佳 F1 分数，特别是在召回率方面有显著提升
            - 在农村电压数据集上，TimesNet 系列模型整体优于 DLinear 基线
            - PSM 数据集上各模型差异较小，但 VoltageTimesNet_v2 仍保持领先
            """
        )

        # 事件绑定
        def on_dataset_change(dataset):
            radar_fig, bar_fig, df, info = update_comparison(dataset)
            return radar_fig, bar_fig, df, f"**数据集说明**: {info}"

        dataset_radio.change(
            fn=on_dataset_change,
            inputs=[dataset_radio],
            outputs=[radar_plot, bar_plot, metrics_table, dataset_info],
        )

        # 页面加载时初始化
        radar_init, bar_init, df_init, _ = update_comparison("RuralVoltage")
        radar_plot.value = radar_init
        bar_plot.value = bar_init
        metrics_table.value = df_init


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    # 测试回调函数
    print("测试 update_comparison 函数...")

    for dataset in ["RuralVoltage", "PSM"]:
        radar_fig, bar_fig, df, info = update_comparison(dataset)
        print(f"\n数据集: {dataset}")
        print(f"  雷达图: {type(radar_fig).__name__}")
        print(f"  柱状图: {type(bar_fig).__name__}")
        print(f"  表格行数: {len(df)}")
        print(f"  数据集说明: {info[:50]}...")

    print("\n测试 Gradio 组件创建...")

    # 简单测试标签页创建（不启动服务器）
    with gr.Blocks() as demo:
        create_arena_tab()

    print("标签页创建成功!")
    print("\n所有测试通过!")
