#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimesNet 变体模型对比结果分析与可视化

使用 MATLAB 风格的科研绑图配置
结果按时间戳分组保存
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义绑图样式
try:
    from utils.plot_style import (
        apply_matlab_style,
        MATLAB_COLORS_MODERN as COLORS,
        MARKERS,
        LINE_STYLES,
        save_figure,
        set_axis_style,
    )

    apply_matlab_style()
except ImportError:
    print("警告: 无法导入自定义绑图样式，使用默认配置")
    # 备用配置
    COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE"]
    MARKERS = ["o", "s", "^", "D", "v", "<"]
    LINE_STYLES = ["-", "--", "-.", ":"]

    # 设置中文字体
    mpl.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.alpha"] = 0.7
    mpl.rcParams["savefig.dpi"] = 300


def get_timestamp_dir(base_dir):
    """创建带时间戳的结果目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir, timestamp


def parse_log_file(log_path):
    """解析训练日志文件，提取指标"""
    results = {
        "train_loss": [],
        "vali_loss": [],
        "test_loss": [],
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "training_time": None,
    }

    if not os.path.exists(log_path):
        print(f"警告: 日志文件不存在 {log_path}")
        return results

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 提取每个 epoch 的损失
    epoch_pattern = r"Epoch:\s*(\d+).*?Train Loss:\s*([\d.]+).*?Vali Loss:\s*([\d.]+).*?Test Loss:\s*([\d.]+)"
    for match in re.finditer(epoch_pattern, content, re.DOTALL):
        results["train_loss"].append(float(match.group(2)))
        results["vali_loss"].append(float(match.group(3)))
        results["test_loss"].append(float(match.group(4)))

    # 提取最终测试指标
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


def plot_training_curves(all_results, output_dir):
    """绘制训练曲线对比图 - MATLAB 风格"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (model_name, results) in enumerate(all_results.items()):
        if results["train_loss"]:
            epochs = range(1, len(results["train_loss"]) + 1)
            color = COLORS[idx % len(COLORS)]
            marker = MARKERS[idx % len(MARKERS)]

            # 训练损失
            axes[0].plot(
                epochs,
                results["train_loss"],
                linestyle="-",
                marker=marker,
                markevery=2,
                label=model_name,
                color=color,
                linewidth=2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1,
            )
            # 验证损失
            axes[1].plot(
                epochs,
                results["vali_loss"],
                linestyle="-",
                marker=marker,
                markevery=2,
                label=model_name,
                color=color,
                linewidth=2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1,
            )
            # 测试损失
            axes[2].plot(
                epochs,
                results["test_loss"],
                linestyle="-",
                marker=marker,
                markevery=2,
                label=model_name,
                color=color,
                linewidth=2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1,
            )

    titles = ["训练损失", "验证损失", "测试损失"]
    for i, ax in enumerate(axes):
        ax.set_xlabel("训练轮次", fontweight="normal")
        ax.set_ylabel("损失值", fontweight="normal")
        ax.set_title(titles[i], fontweight="bold")
        ax.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(left=0.5)

        # MATLAB 风格边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "训练曲线对比.png"), dpi=300, facecolor="white")
    plt.savefig(os.path.join(output_dir, "训练曲线对比.pdf"), facecolor="white")
    plt.close()

    print("已保存: 训练曲线对比.png/pdf")


def plot_metrics_comparison(all_results, output_dir):
    """绘制性能指标对比图 - MATLAB 风格"""
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    models = list(all_results.keys())
    n_models = len(models)
    n_metrics = len(metrics)

    # 准备数据
    data = np.zeros((n_models, n_metrics))
    for i, model in enumerate(models):
        for j, key in enumerate(metric_keys):
            val = all_results[model].get(key)
            data[i, j] = val if val is not None else 0

    # 绑制分组柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for i, model in enumerate(models):
        bars = ax.bar(
            x + offsets[i],
            data[i],
            width * 0.9,
            label=model,
            color=COLORS[i % len(COLORS)],
            edgecolor="black",
            linewidth=1.2,
        )
        # 添加数值标签
        for bar, val in zip(bars, data[i]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="normal",
                )

    ax.set_ylabel("指标值", fontweight="normal")
    ax.set_title("模型性能指标对比", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # MATLAB 风格边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "性能指标对比.png"), dpi=300, facecolor="white")
    plt.savefig(os.path.join(output_dir, "性能指标对比.pdf"), facecolor="white")
    plt.close()

    print("已保存: 性能指标对比.png/pdf")


def plot_radar_chart(all_results, output_dir):
    """绘制雷达图对比 - MATLAB 风格"""
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    models = list(all_results.keys())
    n_metrics = len(metrics)

    # 准备数据
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for idx, model in enumerate(models):
        values = []
        for key in metric_keys:
            val = all_results[model].get(key)
            values.append(val if val is not None else 0)
        values += values[:1]  # 闭合

        color = COLORS[idx % len(COLORS)]
        ax.plot(
            angles,
            values,
            linestyle="-",
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2,
            markersize=8,
            label=model,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_title("模型性能雷达图对比", fontweight="bold", y=1.08)

    # 网格样式
    ax.grid(True, linestyle="--", alpha=0.7)

    # 图例
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "雷达图对比.png"), dpi=300, facecolor="white")
    plt.savefig(os.path.join(output_dir, "雷达图对比.pdf"), facecolor="white")
    plt.close()

    print("已保存: 雷达图对比.png/pdf")


def plot_f1_bar_chart(all_results, output_dir):
    """绘制 F1 分数排名柱状图 - MATLAB 风格"""
    # 提取 F1 分数并排序
    f1_scores = []
    for model, results in all_results.items():
        f1 = results.get("f1")
        if f1 is not None:
            f1_scores.append((model, f1))

    f1_scores.sort(key=lambda x: x[1], reverse=True)

    if not f1_scores:
        print("警告: 没有有效的 F1 分数数据")
        return

    models, scores = zip(*f1_scores)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用渐变色
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    bars = ax.barh(
        range(len(models)),
        scores,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        height=0.6,
    )

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel("F1 分数", fontweight="normal")
    ax.set_title("模型 F1 分数排名", fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # MATLAB 风格边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 添加最佳模型标注
    ax.axvline(x=scores[0], color="#D95319", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        scores[0] - 0.01,
        len(models) - 0.5,
        f"最佳: {scores[0]:.4f}",
        ha="right",
        fontsize=10,
        color="#D95319",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "F1分数对比.png"), dpi=300, facecolor="white")
    plt.savefig(os.path.join(output_dir, "F1分数对比.pdf"), facecolor="white")
    plt.close()

    print("已保存: F1分数对比.png/pdf")


def generate_report(all_results, output_dir, timestamp):
    """生成分析报告"""
    report = []
    report.append("# TimesNet 变体模型 PSM 数据集对比实验报告\n\n")
    report.append(f"**实验时间戳**: {timestamp}\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 实验配置
    report.append("## 1. 实验配置\n\n")
    report.append("| 参数 | 值 |\n")
    report.append("|------|-----|\n")
    report.append("| 数据集 | PSM (服务器机器数据集) |\n")
    report.append("| 序列长度 | 100 |\n")
    report.append("| 隐藏维度 | 64 |\n")
    report.append("| 编码器层数 | 2 |\n")
    report.append("| Top-K 周期数 | 3 |\n")
    report.append("| 批次大小 | 128 |\n")
    report.append("| 训练轮数 | 10 |\n")
    report.append("| 学习率 | 0.0001 |\n")
    report.append("| 异常比例 | 1% |\n\n")

    # 性能对比
    report.append("## 2. 模型性能对比\n\n")
    report.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间(秒) |\n")
    report.append("|------|--------|--------|--------|--------|-------------|\n")

    best_f1 = 0
    best_model = ""

    for model, results in all_results.items():
        acc = results.get("accuracy", 0)
        prec = results.get("precision", 0)
        rec = results.get("recall", 0)
        f1 = results.get("f1", 0)
        time_cost = results.get("training_time", 0)

        if f1 and f1 > best_f1:
            best_f1 = f1
            best_model = model

        report.append(
            f"| {model} | {acc:.4f if acc else 'N/A'} | "
            f"{prec:.4f if prec else 'N/A'} | {rec:.4f if rec else 'N/A'} | "
            f"{f1:.4f if f1 else 'N/A'} | {time_cost:.1f if time_cost else 'N/A'} |\n"
        )

    report.append(f"\n**最佳模型**: {best_model} (F1分数: {best_f1:.4f})\n\n")

    # 模型特点分析
    report.append("## 3. 模型特点分析\n\n")

    model_descriptions = {
        "TimesNet": (
            "基础模型",
            "FFT 自动发现周期 + 2D卷积时序建模",
            "通用性强，无需先验知识",
            "通用时序异常检测",
        ),
        "VoltageTimesNet": (
            "电压优化版",
            "预设电网周期 (60/300/900/3600 采样点) + FFT 混合",
            "融合领域知识，对电压信号更敏感",
            "电力系统电压监测",
        ),
        "TPATimesNet": (
            "三相注意力版",
            "三相交叉注意力机制，建模 Va/Vb/Vc 相位关系",
            "捕获三相不平衡异常",
            "三相电力系统异常检测",
        ),
        "MTSTimesNet": (
            "多尺度版",
            "短期/中期/长期三路并行 + 自适应融合",
            "同时捕获瞬态事件和长期趋势",
            "复杂多尺度异常模式",
        ),
        "HybridTimesNet": (
            "混合发现版",
            "置信度融合预设周期与 FFT 发现周期",
            "鲁棒性强，可解释性好",
            "需要可靠周期检测的场景",
        ),
    }

    for model in all_results.keys():
        desc = model_descriptions.get(
            model, ("未知", "未知", "未知", "未知")
        )
        report.append(f"\n### 3.{list(all_results.keys()).index(model)+1} {model} ({desc[0]})\n")
        report.append(f"- **核心机制**: {desc[1]}\n")
        report.append(f"- **优点**: {desc[2]}\n")
        report.append(f"- **适用场景**: {desc[3]}\n")

    # 结论
    report.append("\n## 4. 结论\n\n")
    report.append(
        f"在 PSM 数据集上的对比实验中，**{best_model}** 取得了最佳 F1 分数 ({best_f1:.4f})。\n\n"
    )
    report.append("各模型的表现差异反映了不同设计理念的优劣：\n")
    report.append("- 基础 TimesNet 提供了可靠的基准性能\n")
    report.append("- 变体模型通过引入领域知识或多尺度机制进一步提升检测能力\n\n")

    # 可视化结果
    report.append("## 5. 可视化结果\n\n")
    report.append("- `训练曲线对比.png/pdf`: 训练曲线对比图\n")
    report.append("- `性能指标对比.png/pdf`: 性能指标柱状图\n")
    report.append("- `雷达图对比.png/pdf`: 雷达图综合对比\n")
    report.append("- `F1分数对比.png/pdf`: F1 分数排名图\n")

    # 保存报告
    report_path = os.path.join(output_dir, "实验分析报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)

    print("已保存: 实验分析报告.md")

    # 保存 JSON 格式结果
    json_data = {
        "实验时间戳": timestamp,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型结果": {},
    }

    for model, results in all_results.items():
        json_data["模型结果"][model] = {
            "准确率": results.get("accuracy"),
            "精确率": results.get("precision"),
            "召回率": results.get("recall"),
            "F1分数": results.get("f1"),
            "训练时间": results.get("training_time"),
            "训练轮次": len(results.get("train_loss", [])),
        }

    json_path = os.path.join(output_dir, "实验结果.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("已保存: 实验结果.json")


def main():
    parser = argparse.ArgumentParser(description="分析 TimesNet 变体模型对比实验结果")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/psm_comparison",
        help="结果目录路径",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["TimesNet", "VoltageTimesNet", "TPATimesNet", "MTSTimesNet", "HybridTimesNet"],
        help="要分析的模型列表",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="不使用时间戳子目录",
    )

    args = parser.parse_args()

    # 创建输出目录
    if args.no_timestamp:
        output_dir = args.result_dir
        timestamp = "无时间戳"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir, timestamp = get_timestamp_dir(args.result_dir)

    print("\n" + "=" * 50)
    print("TimesNet 变体模型对比结果分析")
    print("=" * 50)
    print(f"实验时间戳: {timestamp}")
    print(f"结果保存目录: {output_dir}\n")

    # 收集所有模型结果
    all_results = {}
    for model in args.models:
        log_path = os.path.join(args.result_dir, f"{model}_PSM.log")
        if not os.path.exists(log_path):
            # 尝试其他可能的日志名称
            alt_paths = [
                os.path.join(args.result_dir, f"{model}.log"),
                os.path.join(args.result_dir, f"train_{model}.log"),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    log_path = alt_path
                    break

        results = parse_log_file(log_path)

        # 打印解析结果
        print(f"解析模型: {model}")
        if results["f1"]:
            print(f"  准确率: {results['accuracy']:.4f}")
            print(f"  精确率: {results['precision']:.4f}")
            print(f"  召回率: {results['recall']:.4f}")
            print(f"  F1分数: {results['f1']:.4f}")
        else:
            print("  无有效结果")
        print()

        all_results[model] = results

    # 检查是否有有效数据
    has_valid_data = any(r.get("f1") for r in all_results.values())
    if not has_valid_data:
        print("警告: 没有找到有效的实验结果")
        return

    # 生成可视化
    print("=" * 50)
    print("生成可视化图表...")
    print("=" * 50)

    plot_training_curves(all_results, output_dir)
    plot_metrics_comparison(all_results, output_dir)
    plot_radar_chart(all_results, output_dir)
    plot_f1_bar_chart(all_results, output_dir)

    # 生成报告
    print("\n" + "=" * 50)
    print("生成分析报告...")
    print("=" * 50)

    generate_report(all_results, output_dir, timestamp)

    print(f"\n分析完成！所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
