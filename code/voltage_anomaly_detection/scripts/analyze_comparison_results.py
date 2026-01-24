#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimesNet 变体模型对比结果分析与可视化
解析训练日志，生成对比图表和分析报告
结果按时间戳分组保存
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置中文字体 (优先使用系统可用的中文字体)
matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 学术论文风格设置
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


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
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (model_name, results) in enumerate(all_results.items()):
        if results["train_loss"]:
            epochs = range(1, len(results["train_loss"]) + 1)
            axes[0].plot(
                epochs,
                results["train_loss"],
                "-o",
                label=model_name,
                color=colors[idx],
                markersize=4,
            )
            axes[1].plot(
                epochs,
                results["vali_loss"],
                "-s",
                label=model_name,
                color=colors[idx],
                markersize=4,
            )
            axes[2].plot(
                epochs,
                results["test_loss"],
                "-^",
                label=model_name,
                color=colors[idx],
                markersize=4,
            )

    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失值")
    axes[0].set_title("训练损失")
    axes[0].legend(loc="upper right")

    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("损失值")
    axes[1].set_title("验证损失")
    axes[1].legend(loc="upper right")

    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("损失值")
    axes[2].set_title("测试损失")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "训练曲线对比.png"))
    plt.savefig(os.path.join(output_dir, "训练曲线对比.pdf"))
    plt.close()
    print("已保存: 训练曲线对比.png/pdf")


def plot_metrics_comparison(all_results, output_dir):
    """绘制性能指标对比柱状图"""
    models = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["准确率", "精确率", "召回率", "F1分数"]

    # 准备数据
    data = {m: [] for m in metrics}
    valid_models = []

    for model in models:
        results = all_results[model]
        if results["accuracy"] is not None:
            valid_models.append(model)
            for m in metrics:
                data[m].append(results[m])

    if not valid_models:
        print("警告: 没有有效的指标数据")
        return

    # 绘制分组柱状图
    x = np.arange(len(valid_models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=label, color=colors[i])
        # 在柱状图上标注数值
        for bar, val in zip(bars, data[metric]):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_xlabel("模型")
    ax.set_ylabel("分数")
    ax.set_title("TimesNet 变体模型在 PSM 数据集上的性能对比")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "性能指标对比.png"))
    plt.savefig(os.path.join(output_dir, "性能指标对比.pdf"))
    plt.close()
    print("已保存: 性能指标对比.png/pdf")


def plot_radar_chart(all_results, output_dir):
    """绘制雷达图对比各模型综合性能"""
    models = []
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["准确率", "精确率", "召回率", "F1分数"]

    data = []
    for model, results in all_results.items():
        if results["accuracy"] is not None:
            models.append(model)
            data.append([results[m] for m in metrics])

    if not models:
        return

    # 雷达图设置
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for idx, (model, values) in enumerate(zip(models, data)):
        values = values + values[:1]  # 闭合
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("多维度性能综合对比", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "雷达图对比.png"))
    plt.savefig(os.path.join(output_dir, "雷达图对比.pdf"))
    plt.close()
    print("已保存: 雷达图对比.png/pdf")


def plot_f1_bar_chart(all_results, output_dir):
    """绘制 F1-Score 单独对比图（用于论文）"""
    models = []
    f1_scores = []

    for model, results in all_results.items():
        if results["f1"] is not None:
            models.append(model)
            f1_scores.append(results["f1"])

    if not models:
        return

    # 排序
    sorted_pairs = sorted(zip(f1_scores, models), reverse=True)
    f1_scores, models = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))[::-1]

    bars = ax.barh(models, f1_scores, color=colors, edgecolor="navy", linewidth=1.2)

    # 标注数值
    for bar, score in zip(bars, f1_scores):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("F1 分数")
    ax.set_title("TimesNet 变体模型 F1 分数对比 (PSM 数据集)")
    ax.set_xlim(0, max(f1_scores) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    # 高亮最佳模型
    best_idx = 0
    bars[best_idx].set_color("#e74c3c")
    bars[best_idx].set_edgecolor("darkred")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "F1分数对比.png"))
    plt.savefig(os.path.join(output_dir, "F1分数对比.pdf"))
    plt.close()
    print("已保存: F1分数对比.png/pdf")


def generate_report(all_results, output_dir, timestamp):
    """生成 Markdown 分析报告"""
    report = []
    report.append("# TimesNet 变体模型 PSM 数据集对比实验报告\n")
    report.append(f"**实验时间戳**: {timestamp}")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("\n## 1. 实验配置\n")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append("| 数据集 | PSM (服务器机器数据集) |")
    report.append("| 序列长度 | 100 |")
    report.append("| 隐藏维度 | 64 |")
    report.append("| 编码器层数 | 2 |")
    report.append("| Top-K 周期数 | 3 |")
    report.append("| 批次大小 | 128 |")
    report.append("| 训练轮数 | 10 |")
    report.append("| 学习率 | 0.0001 |")
    report.append("| 异常比例 | 1% |")

    report.append("\n## 2. 模型性能对比\n")
    report.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间(秒) |")
    report.append("|------|--------|--------|--------|--------|-------------|")

    best_f1 = 0
    best_model = ""

    for model, results in all_results.items():
        if results["accuracy"] is not None:
            acc = f"{results['accuracy']:.4f}"
            prec = f"{results['precision']:.4f}"
            rec = f"{results['recall']:.4f}"
            f1 = f"{results['f1']:.4f}"
            time_str = (
                f"{results['training_time']:.1f}" if results["training_time"] else "N/A"
            )

            if results["f1"] > best_f1:
                best_f1 = results["f1"]
                best_model = model

            report.append(f"| {model} | {acc} | {prec} | {rec} | {f1} | {time_str} |")
        else:
            report.append(f"| {model} | N/A | N/A | N/A | N/A | N/A |")

    report.append(f"\n**最佳模型**: {best_model} (F1分数: {best_f1:.4f})")

    report.append("\n## 3. 模型特点分析\n")
    report.append(
        """
### 3.1 TimesNet (基础模型)
- **核心机制**: FFT 自动发现周期 + 2D卷积时序建模
- **优点**: 通用性强，无需先验知识
- **适用场景**: 通用时序异常检测

### 3.2 VoltageTimesNet (电压优化版)
- **创新点**: 预设电网周期 (60/300/900/3600 采样点) + FFT 混合
- **优点**: 融合领域知识，对电压信号更敏感
- **适用场景**: 电力系统电压监测

### 3.3 TPATimesNet (三相注意力版)
- **创新点**: 三相交叉注意力机制，建模 Va/Vb/Vc 相位关系
- **优点**: 捕获三相不平衡异常
- **适用场景**: 三相电力系统异常检测

### 3.4 MTSTimesNet (多尺度版)
- **创新点**: 短期/中期/长期三路并行 + 自适应融合
- **优点**: 同时捕获瞬态事件和长期趋势
- **适用场景**: 复杂多尺度异常模式

### 3.5 HybridTimesNet (混合发现版)
- **创新点**: 置信度融合预设周期与 FFT 发现周期
- **优点**: 鲁棒性强，可解释性好
- **适用场景**: 需要可靠周期检测的场景
"""
    )

    report.append("\n## 4. 结论\n")
    if best_model:
        report.append(
            f"在 PSM 数据集上的对比实验中，**{best_model}** 取得了最佳 F1 分数 ({best_f1:.4f})。"
        )
        report.append("\n各模型的表现差异反映了不同设计理念的优劣：")
        report.append("- 基础 TimesNet 提供了可靠的基准性能")
        report.append("- 变体模型通过引入领域知识或多尺度机制进一步提升检测能力")

    report.append("\n## 5. 可视化结果\n")
    report.append("- `训练曲线对比.png/pdf`: 训练曲线对比图")
    report.append("- `性能指标对比.png/pdf`: 性能指标柱状图")
    report.append("- `雷达图对比.png/pdf`: 雷达图综合对比")
    report.append("- `F1分数对比.png/pdf`: F1 分数排名图")

    # 保存报告
    report_path = os.path.join(output_dir, "实验分析报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("已保存: 实验分析报告.md")

    # 同时保存 JSON 格式结果
    json_results = {
        "实验时间戳": timestamp,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型结果": {},
    }
    for model, results in all_results.items():
        json_results["模型结果"][model] = {
            "准确率": results["accuracy"],
            "精确率": results["precision"],
            "召回率": results["recall"],
            "F1分数": results["f1"],
            "训练时间": results["training_time"],
            "训练轮次": len(results["train_loss"]),
        }

    json_path = os.path.join(output_dir, "实验结果.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print("已保存: 实验结果.json")


def main():
    parser = argparse.ArgumentParser(description="分析 TimesNet 变体模型对比结果")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/psm_comparison",
        help="结果目录路径",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "TimesNet",
            "VoltageTimesNet",
            "TPATimesNet",
            "MTSTimesNet",
            "HybridTimesNet",
        ],
        help="要分析的模型列表",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="不使用时间戳子目录",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("TimesNet 变体模型对比结果分析")
    print("=" * 50)

    # 创建带时间戳的结果目录
    if args.no_timestamp:
        output_dir = args.result_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir, timestamp = get_timestamp_dir(args.result_dir)

    print(f"实验时间戳: {timestamp}")
    print(f"结果保存目录: {output_dir}")

    # 解析所有模型的日志
    all_results = {}
    for model in args.models:
        # 优先查找当前目录，然后查找父目录
        log_path = os.path.join(args.result_dir, f"{model}_log.txt")
        if not os.path.exists(log_path):
            log_path = os.path.join(os.path.dirname(args.result_dir), f"{model}_log.txt")

        print(f"\n解析模型: {model}")
        results = parse_log_file(log_path)
        all_results[model] = results

        if results["f1"] is not None:
            print(f"  准确率: {results['accuracy']:.4f}")
            print(f"  精确率: {results['precision']:.4f}")
            print(f"  召回率: {results['recall']:.4f}")
            print(f"  F1分数: {results['f1']:.4f}")
        else:
            print("  (无有效结果)")

    # 生成可视化
    print("\n" + "=" * 50)
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
