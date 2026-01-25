#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整实验分析脚本

一键生成所有图表和分析报告。

使用方法：
    python scripts/generate_full_analysis.py --result_dir ./results/PSM_comparison_XXXXXX

    # 指定数据集
    python scripts/generate_full_analysis.py --result_dir ./results/PSM_comparison_XXXXXX --dataset PSM

    # 包含交互式图表
    python scripts/generate_full_analysis.py --result_dir ./results/PSM_comparison_XXXXXX --include_interactive

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from analysis.result_manager import ResultManager
from analysis.report_generator import ReportGenerator
from visualization.plot_factory import PlotFactory
from utils.font_config import setup_matplotlib_thesis_style


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="完整实验分析脚本")

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="实验结果目录路径"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PSM",
        help="数据集名称 (default: PSM)"
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=4,
        help="论文章节号 (default: 4)"
    )
    parser.add_argument(
        "--include_interactive",
        action="store_true",
        help="是否生成交互式图表"
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="不使用时间戳创建分析目录"
    )

    return parser.parse_args()


def generate_all_figures(
    results: dict,
    output_dir: str,
    chapter: int = 4
) -> dict:
    """
    生成所有静态图表

    Args:
        results: 实验结果字典
        output_dir: 输出目录
        chapter: 章节号

    Returns:
        dict: 生成的图表路径字典
    """
    print("\n" + "=" * 60)
    print("开始生成静态图表...")
    print("=" * 60)

    # 创建图表工厂
    factory = PlotFactory(output_dir=output_dir, chapter=chapter)

    generated = {}
    metrics = results.get("metrics", {})
    training_history = results.get("training_history", {})
    predictions = results.get("predictions", {})

    # 1. 训练过程可视化
    if training_history:
        print("\n[1/8] 生成训练曲线...")
        try:
            path = factory.training_loss_curve(training_history)
            generated["训练曲线对比"] = path
            print(f"  ✓ 训练曲线对比: {path}")
        except Exception as e:
            print(f"  ✗ 训练曲线对比失败: {e}")

    # 2. 性能指标柱状图
    if metrics:
        print("\n[2/8] 生成性能指标柱状图...")
        try:
            path = factory.metrics_bar_chart(metrics)
            generated["性能指标对比"] = path
            print(f"  ✓ 性能指标对比: {path}")
        except Exception as e:
            print(f"  ✗ 性能指标对比失败: {e}")

    # 3. 雷达图
    if metrics:
        print("\n[3/8] 生成雷达图...")
        try:
            path = factory.radar_chart(metrics)
            generated["雷达图对比"] = path
            print(f"  ✓ 雷达图对比: {path}")
        except Exception as e:
            print(f"  ✗ 雷达图对比失败: {e}")

    # 4. 性能热力图
    if metrics:
        print("\n[4/8] 生成性能热力图...")
        try:
            path = factory.performance_heatmap(metrics)
            generated["性能热力图"] = path
            print(f"  ✓ 性能热力图: {path}")
        except Exception as e:
            print(f"  ✗ 性能热力图失败: {e}")

    # 5. F1分数排名
    if metrics:
        print("\n[5/8] 生成F1分数排名图...")
        try:
            path = factory.f1_ranking_chart(metrics)
            generated["F1分数对比"] = path
            print(f"  ✓ F1分数对比: {path}")
        except Exception as e:
            print(f"  ✗ F1分数对比失败: {e}")

    # 6. ROC曲线
    if predictions:
        print("\n[6/8] 生成ROC曲线...")
        try:
            path = factory.roc_curve(predictions)
            generated["ROC曲线"] = path
            print(f"  ✓ ROC曲线: {path}")
        except Exception as e:
            print(f"  ✗ ROC曲线失败: {e}")

    # 7. PR曲线
    if predictions:
        print("\n[7/8] 生成PR曲线...")
        try:
            path = factory.pr_curve(predictions)
            generated["PR曲线"] = path
            print(f"  ✓ PR曲线: {path}")
        except Exception as e:
            print(f"  ✗ PR曲线失败: {e}")

    # 8. 混淆矩阵（为最佳模型生成）
    if predictions and metrics:
        print("\n[8/8] 生成混淆矩阵...")
        best_model = max(metrics.items(), key=lambda x: x[1].get("f1_score", 0))[0]
        if best_model in predictions:
            try:
                pred_data = predictions[best_model]
                if "pred" in pred_data and "gt" in pred_data:
                    path = factory.confusion_matrix(
                        pred_data["gt"],
                        pred_data["pred"],
                        model_name=best_model,
                        filename=f"混淆矩阵_{best_model}"
                    )
                    generated[f"混淆矩阵_{best_model}"] = path
                    print(f"  ✓ 混淆矩阵_{best_model}: {path}")
            except Exception as e:
                print(f"  ✗ 混淆矩阵失败: {e}")

    print("\n" + "-" * 60)
    print(f"静态图表生成完成，共 {len(generated)} 个")
    print("-" * 60)

    return generated


def generate_interactive_figures(
    results: dict,
    output_dir: str
) -> dict:
    """
    生成交互式图表

    Args:
        results: 实验结果字典
        output_dir: 输出目录

    Returns:
        dict: 生成的图表路径字典
    """
    print("\n" + "=" * 60)
    print("开始生成交互式图表...")
    print("=" * 60)

    try:
        from visualization.interactive_plots import InteractivePlotter, PLOTLY_AVAILABLE
    except ImportError:
        print("  ✗ Plotly 未安装，跳过交互式图表生成")
        print("  提示: 运行 'pip install plotly kaleido' 安装")
        return {}

    if not PLOTLY_AVAILABLE:
        print("  ✗ Plotly 不可用，跳过交互式图表生成")
        return {}

    plotter = InteractivePlotter(output_dir=output_dir)
    generated = {}

    training_history = results.get("training_history", {})
    metrics = results.get("metrics", {})

    # 1. 训练仪表盘
    if training_history:
        print("\n[1/3] 生成训练仪表盘...")
        try:
            path = plotter.training_dashboard(training_history)
            generated["训练仪表盘"] = path
            print(f"  ✓ 训练仪表盘: {path}")
        except Exception as e:
            print(f"  ✗ 训练仪表盘失败: {e}")

    # 2. 性能雷达图
    if metrics:
        print("\n[2/3] 生成交互式雷达图...")
        try:
            path = plotter.metrics_radar(metrics)
            generated["性能雷达图"] = path
            print(f"  ✓ 性能雷达图: {path}")
        except Exception as e:
            print(f"  ✗ 性能雷达图失败: {e}")

    # 3. 平行坐标图
    if metrics:
        print("\n[3/3] 生成平行坐标图...")
        try:
            path = plotter.metrics_parallel_coordinates(metrics)
            generated["平行坐标图"] = path
            print(f"  ✓ 平行坐标图: {path}")
        except Exception as e:
            print(f"  ✗ 平行坐标图失败: {e}")

    print("\n" + "-" * 60)
    print(f"交互式图表生成完成，共 {len(generated)} 个")
    print("-" * 60)

    return generated


def generate_report(
    results: dict,
    output_path: str,
    figure_dir: str,
    experiment_info: dict
) -> str:
    """
    生成分析报告

    Args:
        results: 实验结果字典
        output_path: 输出文件路径
        figure_dir: 图表目录
        experiment_info: 实验信息

    Returns:
        str: 报告内容
    """
    print("\n" + "=" * 60)
    print("生成分析报告...")
    print("=" * 60)

    generator = ReportGenerator(template="thesis")
    content = generator.generate_markdown(
        results=results,
        output_path=output_path,
        figure_dir=figure_dir,
        experiment_info=experiment_info
    )

    print(f"  ✓ 分析报告已保存至: {output_path}")

    return content


def save_metrics_json(
    results: dict,
    output_path: str,
    experiment_info: dict
):
    """保存指标到 JSON 文件"""
    manager = ResultManager()
    manager.save_metrics_json(
        metrics=results.get("metrics", {}),
        output_path=output_path,
        experiment_info=experiment_info
    )
    print(f"  ✓ 指标 JSON 已保存至: {output_path}")


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("时序异常检测实验完整分析")
    print("=" * 60)
    print(f"实验目录: {args.result_dir}")
    print(f"数据集: {args.dataset}")
    print(f"章节号: {args.chapter}")
    print(f"交互式图表: {'是' if args.include_interactive else '否'}")

    # 初始化样式
    setup_matplotlib_thesis_style(chapter=args.chapter, verbose=True)

    # 加载实验结果
    print("\n加载实验结果...")
    manager = ResultManager(str(Path(args.result_dir).parent))
    results = manager.load_experiment_results(
        args.result_dir,
        dataset=args.dataset,
        load_predictions=True
    )

    metrics = results.get("metrics", {})
    print(f"  加载了 {len(metrics)} 个模型的结果")

    if not metrics:
        print("  ✗ 未找到有效的实验结果，请检查目录")
        return

    # 创建分析目录
    if args.no_timestamp:
        analysis_dir = Path(args.result_dir) / "analysis"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path(args.result_dir) / f"analysis_{timestamp}"

    figures_dir = analysis_dir / "figures"
    interactive_dir = analysis_dir / "interactive"

    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.include_interactive:
        interactive_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n分析结果将保存至: {analysis_dir}")

    # 准备实验信息
    experiment_info = {
        "dataset": args.dataset,
        "type": "comparison",
        "result_dir": args.result_dir,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 生成静态图表
    static_figures = generate_all_figures(
        results=results,
        output_dir=str(figures_dir),
        chapter=args.chapter
    )

    # 生成交互式图表
    interactive_figures = {}
    if args.include_interactive:
        interactive_figures = generate_interactive_figures(
            results=results,
            output_dir=str(interactive_dir)
        )

    # 生成分析报告
    report_path = analysis_dir / "实验分析报告.md"
    generate_report(
        results=results,
        output_path=str(report_path),
        figure_dir="./figures",
        experiment_info=experiment_info
    )

    # 保存指标 JSON
    json_path = analysis_dir / "实验结果.json"
    save_metrics_json(
        results=results,
        output_path=str(json_path),
        experiment_info=experiment_info
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出目录: {analysis_dir}")
    print(f"\n生成的文件:")
    print(f"  - 静态图表: {len(static_figures)} 个")
    print(f"  - 交互式图表: {len(interactive_figures)} 个")
    print(f"  - 分析报告: {report_path}")
    print(f"  - 结果 JSON: {json_path}")

    # 打印最佳模型
    if metrics:
        best_model = max(metrics.items(), key=lambda x: x[1].get("f1_score", 0))
        print(f"\n最佳模型: {best_model[0]}")
        print(f"  - F1分数: {best_model[1].get('f1_score', 0):.4f}")
        print(f"  - 准确率: {best_model[1].get('accuracy', 0):.4f}")
        print(f"  - 精确率: {best_model[1].get('precision', 0):.4f}")
        print(f"  - 召回率: {best_model[1].get('recall', 0):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
