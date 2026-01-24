#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对性数据集实验结果分析

分析各模型在针对性数据集上的表现，验证模型设计假设

绘图风格: MATLAB 科研绘图风格
- 白色背景 + 黑色粗边框
- 灰色虚线网格
- 高对比度配色
- 中文字体支持
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 MATLAB 风格配置
from utils.plot_style import (
    apply_matlab_style,
    COLORS,
    MARKERS,
    set_axis_style,
    save_figure,
    get_color,
)

# 应用 MATLAB 科研绘图风格
apply_matlab_style()

# 数据集-模型对应关系
DATASET_MODEL_MAP = {
    "periodic_load": {
        "name": "周期性负荷",
        "target_model": "VoltageTimesNet",
        "description": "测试预设周期融合机制",
    },
    "three_phase": {
        "name": "三相不平衡",
        "target_model": "TPATimesNet",
        "description": "测试三相注意力机制",
    },
    "multi_scale": {
        "name": "多尺度复合",
        "target_model": "MTSTimesNet",
        "description": "测试多尺度时序建模",
    },
    "hybrid_period": {
        "name": "混合周期",
        "target_model": "HybridTimesNet",
        "description": "测试置信度融合机制",
    },
    "comprehensive": {
        "name": "综合评估",
        "target_model": None,
        "description": "公平对比所有模型",
    },
}

MODELS = ["TimesNet", "VoltageTimesNet", "TPATimesNet", "MTSTimesNet"]

MODEL_NAMES_CN = {
    "TimesNet": "TimesNet (基础)",
    "VoltageTimesNet": "VoltageTimesNet (预设周期)",
    "TPATimesNet": "TPATimesNet (三相注意力)",
    "MTSTimesNet": "MTSTimesNet (多尺度)",
    "HybridTimesNet": "HybridTimesNet (混合发现)",
}

# MATLAB 风格模型配色 (使用导入的 COLORS)
MODEL_COLORS = {
    "TimesNet": COLORS[0],        # 蓝色
    "VoltageTimesNet": COLORS[1],  # 橙色
    "TPATimesNet": COLORS[4],      # 绿色
    "MTSTimesNet": COLORS[3],      # 紫色
    "HybridTimesNet": COLORS[6],   # 深红
}


def parse_log_file(log_path):
    """解析训练日志"""
    results = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "training_time": None,
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 提取指标
    metrics_pattern = r"Accuracy:\s*([\d.]+).*?Precision:\s*([\d.]+).*?Recall:\s*([\d.]+).*?F1-score:\s*([\d.]+)"
    match = re.search(metrics_pattern, content)
    if match:
        results["accuracy"] = float(match.group(1))
        results["precision"] = float(match.group(2))
        results["recall"] = float(match.group(3))
        results["f1"] = float(match.group(4))

    # 提取训练时间
    time_pattern = r"cost time:\s*([\d.]+)"
    times = re.findall(time_pattern, content)
    if times:
        results["training_time"] = sum(float(t) for t in times)

    return results


def collect_results(result_dir):
    """收集所有实验结果"""
    all_results = {}

    for dataset in DATASET_MODEL_MAP.keys():
        all_results[dataset] = {}
        for model in MODELS:
            log_path = os.path.join(result_dir, f"{model}_{dataset}.log")
            results = parse_log_file(log_path)
            all_results[dataset][model] = results

    return all_results


def get_timestamp_dir(base_dir):
    """创建带时间戳的结果目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir, timestamp


def plot_heatmap(all_results, output_dir):
    """绘制 F1 分数热力图 (MATLAB 风格)"""
    datasets = list(DATASET_MODEL_MAP.keys())
    models = MODELS

    # 构建数据矩阵
    data = np.zeros((len(datasets), len(models)))
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            f1 = all_results[dataset][model].get("f1")
            data[i, j] = f1 if f1 is not None else 0

    fig, ax = plt.subplots(figsize=(12, 8))

    # 使用 MATLAB 风格的配色方案
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=1.0)

    # 设置标签
    dataset_labels = [DATASET_MODEL_MAP[d]["name"] for d in datasets]
    model_labels = [MODEL_NAMES_CN.get(m, m) for m in models]

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(dataset_labels, fontsize=11)

    # 添加数值标注
    for i in range(len(datasets)):
        for j in range(len(models)):
            value = data[i, j]
            color = "white" if value > 0.65 else "black"

            # 标记目标模型
            target = DATASET_MODEL_MAP[datasets[i]]["target_model"]
            if target == models[j]:
                text = f"{value:.3f}\n★"
                fontweight = "bold"
            else:
                text = f"{value:.3f}"
                fontweight = "normal"

            ax.text(j, i, text, ha="center", va="center", color=color,
                   fontsize=11, fontweight=fontweight)

    ax.set_title("模型-数据集 F1 分数热力图\n(★ 表示预期优势模型)",
                fontsize=16, fontweight="bold", pad=15)

    # MATLAB 风格边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("F1 分数", rotation=-90, va="bottom", fontsize=12)
    cbar.outline.set_linewidth(1.5)

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "F1热力图"))
    plt.close()

    print("已保存: F1热力图.png/pdf")


def plot_advantage_analysis(all_results, output_dir):
    """绘制模型优势分析图 (MATLAB 风格)"""
    datasets = list(DATASET_MODEL_MAP.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        info = DATASET_MODEL_MAP[dataset]

        f1_scores = []
        model_names = []
        bar_colors = []

        for model in MODELS:
            f1 = all_results[dataset][model].get("f1", 0)
            f1_scores.append(f1 if f1 else 0)
            model_names.append(model.replace("TimesNet", "\nTimesNet"))

            # 高亮目标模型，使用 MATLAB 配色
            if info["target_model"] == model:
                bar_colors.append(MODEL_COLORS[model])
            else:
                bar_colors.append("#B0B0B0")  # 浅灰色

        bars = ax.bar(
            model_names, f1_scores, color=bar_colors,
            edgecolor="black", linewidth=1.5, width=0.7
        )

        # 添加数值标注
        for bar, score in zip(bars, f1_scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{score:.3f}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_title(f"{info['name']}数据集\n{info['description']}",
                    fontsize=13, fontweight="bold")
        ax.set_ylabel("F1 分数", fontsize=11)
        ax.set_ylim(0, 1.15)

        # MATLAB 风格: 粗边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 网格线
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
        ax.set_axisbelow(True)

        # 标注目标阈值线
        if info["target_model"]:
            ax.axhline(y=0.8, color=COLORS[6], linestyle="--",
                      linewidth=1.5, alpha=0.6, label="目标阈值 0.8")

    # 隐藏多余的子图
    if len(datasets) < 6:
        axes[-1].axis("off")

    plt.suptitle(
        "各数据集上的模型性能对比\n(彩色柱表示该数据集的目标优势模型)",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "模型优势分析"))
    plt.close()

    print("已保存: 模型优势分析.png/pdf")


def plot_radar_comparison(all_results, output_dir):
    """绘制综合评估雷达图 (MATLAB 风格)"""
    # 使用综合数据集的结果
    comp_results = all_results.get("comprehensive", {})

    if not any(comp_results.get(m, {}).get("f1") for m in MODELS):
        print("警告: 综合数据集结果不完整，跳过雷达图")
        return

    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for idx, model in enumerate(MODELS):
        values = []
        for key in metric_keys:
            v = comp_results.get(model, {}).get(key, 0)
            values.append(v if v else 0)
        values += values[:1]  # 闭合

        # 使用 MATLAB 风格配色和标记
        color = MODEL_COLORS[model]
        marker = MARKERS[idx]

        ax.plot(
            angles,
            values,
            marker=marker,
            linestyle="-",
            linewidth=2.5,
            markersize=10,
            markeredgewidth=1.5,
            markeredgecolor="black",
            label=MODEL_NAMES_CN.get(model, model),
            color=color,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)

    # 设置雷达图网格
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7, color="#666666")

    ax.set_title("综合数据集上的多维度性能对比",
                fontsize=16, fontweight="bold", y=1.08)

    # MATLAB 风格图例
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.0),
        fontsize=11,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "综合雷达图"))
    plt.close()

    print("已保存: 综合雷达图.png/pdf")


def plot_metrics_comparison(all_results, output_dir):
    """绘制各数据集上的多指标对比图 (MATLAB 风格)"""
    datasets = [d for d in DATASET_MODEL_MAP.keys() if d != "hybrid_period"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["准确率", "精确率", "召回率", "F1分数"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    bar_width = 0.18
    x = np.arange(len(datasets))

    for ax_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[ax_idx]

        for model_idx, model in enumerate(MODELS):
            values = []
            for dataset in datasets:
                v = all_results[dataset][model].get(metric, 0)
                values.append(v if v else 0)

            offset = (model_idx - len(MODELS)/2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, values,
                bar_width * 0.9,
                label=model if ax_idx == 0 else "",
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=1.2,
            )

            # 添加数值标注 (仅对非零值)
            for bar, val in zip(bars, values):
                if val > 0.1:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}",
                        ha="center",
                        fontsize=8,
                        rotation=90,
                    )

        ax.set_title(metric_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("分数", fontsize=11)
        ax.set_ylim(0, 1.2)

        dataset_labels = [DATASET_MODEL_MAP[d]["name"] for d in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=10)

        # MATLAB 风格
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
        ax.set_axisbelow(True)

    # 添加共享图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(MODELS),
        fontsize=11,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    plt.suptitle("各数据集上的多指标性能对比",
                fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(fig, os.path.join(output_dir, "多指标对比"))
    plt.close()

    print("已保存: 多指标对比.png/pdf")


def generate_report(all_results, output_dir, timestamp):
    """生成实验分析报告"""
    report = []
    report.append("# 针对性数据集实验分析报告\n")
    report.append(f"**实验时间戳**: {timestamp}\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 实验概述
    report.append("## 1. 实验概述\n\n")
    report.append("本实验旨在验证各 TimesNet 变体模型的设计假设。\n")
    report.append("通过针对性设计的数据集，测试每个模型在其擅长领域的表现。\n\n")

    report.append("| 数据集 | 目标模型 | 测试目标 |\n")
    report.append("|--------|----------|----------|\n")
    for dataset, info in DATASET_MODEL_MAP.items():
        target = info["target_model"] if info["target_model"] else "所有模型"
        report.append(f"| {info['name']} | {target} | {info['description']} |\n")

    # 结果汇总
    report.append("\n## 2. F1 分数汇总\n\n")
    report.append(
        "| 数据集 | TimesNet | VoltageTimesNet | TPATimesNet | MTSTimesNet | 最佳模型 |\n"
    )
    report.append(
        "|--------|----------|-----------------|-------------|-------------|----------|\n"
    )

    for dataset, info in DATASET_MODEL_MAP.items():
        row = [info["name"]]
        best_f1 = 0
        best_model = ""

        for model in MODELS:
            f1 = all_results[dataset][model].get("f1", 0)
            f1 = f1 if f1 else 0
            row.append(f"{f1:.4f}" if f1 > 0 else "N/A")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        row.append(f"**{best_model}**" if best_model else "N/A")
        report.append("| " + " | ".join(row) + " |\n")

    # 假设验证
    report.append("\n## 3. 设计假设验证\n\n")

    for dataset, info in DATASET_MODEL_MAP.items():
        target = info["target_model"]
        if not target:
            continue

        report.append(f"### {info['name']}数据集\n\n")
        report.append(f"**目标模型**: {target}\n\n")
        report.append(f"**测试假设**: {info['description']}\n\n")

        # 获取各模型 F1
        f1_scores = {m: all_results[dataset][m].get("f1", 0) for m in MODELS}
        f1_scores = {k: v for k, v in f1_scores.items() if v}

        if f1_scores:
            best_model = max(f1_scores, key=f1_scores.get)
            target_f1 = f1_scores.get(target, 0)
            best_f1 = f1_scores[best_model]

            if best_model == target:
                report.append(
                    f"✅ **验证通过**: {target} 以 F1={target_f1:.4f} 取得最佳性能\n\n"
                )
            else:
                diff = best_f1 - target_f1
                report.append(
                    f"⚠️ **部分验证**: 最佳模型为 {best_model} (F1={best_f1:.4f})，"
                    f"{target} F1={target_f1:.4f}，差距 {diff:.4f}\n\n"
                )
        else:
            report.append("❌ **数据不足**: 未获取到有效结果\n\n")

    # 结论
    report.append("## 4. 结论与建议\n\n")
    report.append("### 4.1 模型选型建议\n\n")
    report.append("| 应用场景 | 推荐模型 | 说明 |\n")
    report.append("|----------|----------|------|\n")
    report.append("| 常规电压监测 | TimesNet | 通用性强，基准性能可靠 |\n")
    report.append("| 周期性负荷明显 | VoltageTimesNet | 预设电网周期，针对性强 |\n")
    report.append("| 三相系统监测 | TPATimesNet | 三相注意力，捕捉不平衡 |\n")
    report.append("| 复杂混合异常 | MTSTimesNet | 多尺度建模，综合能力强 |\n")
    report.append("| 高噪声环境 | HybridTimesNet | 置信度融合，鲁棒性强 |\n\n")

    report.append("### 4.2 可视化结果\n\n")
    report.append("- `F1热力图.png/pdf`: 模型-数据集 F1 分数热力图\n")
    report.append("- `模型优势分析.png/pdf`: 各数据集上的模型性能对比\n")
    report.append("- `综合雷达图.png/pdf`: 综合数据集多维度对比\n")

    # 保存报告
    report_path = os.path.join(output_dir, "针对性实验报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)

    print(f"已保存: 针对性实验报告.md")

    # 保存 JSON 结果
    json_data = {
        "实验时间戳": timestamp,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据集信息": {k: v["name"] for k, v in DATASET_MODEL_MAP.items()},
        "模型结果": {},
    }

    for dataset in DATASET_MODEL_MAP.keys():
        json_data["模型结果"][dataset] = {}
        for model in MODELS:
            r = all_results[dataset][model]
            json_data["模型结果"][dataset][model] = {
                "准确率": r.get("accuracy"),
                "精确率": r.get("precision"),
                "召回率": r.get("recall"),
                "F1分数": r.get("f1"),
                "训练时间": r.get("training_time"),
            }

    json_path = os.path.join(output_dir, "针对性实验结果.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"已保存: 针对性实验结果.json")


def main():
    parser = argparse.ArgumentParser(description="分析针对性数据集实验结果")
    parser.add_argument("--result_dir", type=str, required=True, help="结果日志目录")
    parser.add_argument(
        "--no_timestamp", action="store_true", help="不使用时间戳子目录"
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("针对性数据集实验结果分析")
    print("=" * 50)

    # 创建输出目录
    if args.no_timestamp:
        output_dir = args.result_dir
        timestamp = "无时间戳"
    else:
        output_dir, timestamp = get_timestamp_dir(args.result_dir)

    print(f"实验时间戳: {timestamp}")
    print(f"结果保存目录: {output_dir}\n")

    # 收集结果
    all_results = collect_results(args.result_dir)

    # 打印结果概要
    for dataset, info in DATASET_MODEL_MAP.items():
        print(f"\n{info['name']}数据集:")
        for model in MODELS:
            f1 = all_results[dataset][model].get("f1")
            if f1:
                marker = " ★" if info["target_model"] == model else ""
                print(f"  {model}: F1={f1:.4f}{marker}")
            else:
                print(f"  {model}: 无数据")

    # 生成可视化
    print("\n" + "=" * 50)
    print("生成可视化图表...")
    print("=" * 50)

    plot_heatmap(all_results, output_dir)
    plot_advantage_analysis(all_results, output_dir)
    plot_metrics_comparison(all_results, output_dir)
    plot_radar_comparison(all_results, output_dir)

    # 生成报告
    print("\n" + "=" * 50)
    print("生成分析报告...")
    print("=" * 50)

    generate_report(all_results, output_dir, timestamp)

    print(f"\n分析完成！所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
