#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键生成所有图表 / Generate All Plots
======================================

统一调用所有绘图脚本，一键生成完整的实验结果图表。

Usage:
    # 使用示例数据生成所有图表
    python bindgenerate_bindall.py --output ./figures/

    # 使用自定义数据生成所有图表
    python bindgenerate_bindall.py --data results.json --output ./figures/

    # 仅生成指定图表
    python bindgenerate_bindall.py --plots f1_ranking,radar --output ./figures/

Author: Rural Low-Voltage Detection Project
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from bindstyle_config import SAMPLE_DATA, apply_bindstyle

# 导入各绘图模块
from bindplot_f1_bindranking import plot_f1_ranking
from bindplot_bindmetrics_bindbar import plot_metrics_bar
from bindplot_bindradar import plot_radar
from bindplot_bindheatmap import plot_heatmap
from bindplot_bindroc import plot_roc, generate_sample_roc_data
from bindplot_bindpr import plot_pr, generate_sample_pr_data
from bindplot_bindconfusion import plot_confusion_matrix, generate_sample_confusion_matrix
from bindplot_bindtraining import plot_training_curve, generate_sample_training_data


# 可用图表列表
AVAILABLE_PLOTS = {
    'f1_ranking': {
        'name': 'F1 分数排名图',
        'name_en': 'F1 Score Ranking',
        'filename': 'f1_score_ranking',
        'function': 'plot_f1_ranking'
    },
    'metrics_bar': {
        'name': '性能指标柱状图',
        'name_en': 'Metrics Bar Chart',
        'filename': 'metrics_comparison',
        'function': 'plot_metrics_bar'
    },
    'radar': {
        'name': '雷达图',
        'name_en': 'Radar Chart',
        'filename': 'radar_chart',
        'function': 'plot_radar'
    },
    'heatmap': {
        'name': '性能热力图',
        'name_en': 'Performance Heatmap',
        'filename': 'performance_heatmap',
        'function': 'plot_heatmap'
    },
    'roc': {
        'name': 'ROC 曲线',
        'name_en': 'ROC Curve',
        'filename': 'roc_curve',
        'function': 'plot_roc'
    },
    'pr': {
        'name': 'PR 曲线',
        'name_en': 'PR Curve',
        'filename': 'pr_curve',
        'function': 'plot_pr'
    },
    'confusion': {
        'name': '混淆矩阵',
        'name_en': 'Confusion Matrix',
        'filename': 'confusion_matrix',
        'function': 'plot_confusion_matrix'
    },
    'training': {
        'name': '训练曲线',
        'name_en': 'Training Curve',
        'filename': 'training_curve',
        'function': 'plot_training_curve'
    }
}


def load_data(data_path=None):
    """加载数据"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"已加载数据文件: {data_path}")
        return data
    else:
        print("使用示例数据 / Using sample data")
        return SAMPLE_DATA.copy()


def prepare_complete_data(data):
    """
    准备完整的数据集（包括派生数据）
    Prepare complete dataset including derived data
    """
    complete_data = data.copy()

    models = data.get('models', SAMPLE_DATA['models'])
    f1_scores = data.get('f1_scores', SAMPLE_DATA['f1_scores'])
    precision = data.get('precision', SAMPLE_DATA['precision'])
    recall = data.get('recall', SAMPLE_DATA['recall'])

    # 生成 ROC 数据（如果不存在）
    if 'roc_data' not in complete_data:
        complete_data['roc_data'] = generate_sample_roc_data(models, f1_scores)

    # 生成 PR 数据（如果不存在）
    if 'pr_data' not in complete_data:
        complete_data['pr_data'] = generate_sample_pr_data(models, precision, recall)

    # 生成混淆矩阵（如果不存在）
    if 'confusion_matrix' not in complete_data:
        complete_data['confusion_matrix'] = generate_sample_confusion_matrix(
            precision[0], recall[0]
        )
        complete_data['model_name'] = models[0]

    # 生成训练数据（如果不存在）
    if 'training_data' not in complete_data:
        complete_data['training_data'] = generate_sample_training_data(models, f1_scores)

    return complete_data


def generate_all_plots(data, output_dir, plots=None, top_n=5):
    """
    生成所有图表
    Generate all plots

    Args:
        data: 数据字典
        output_dir: 输出目录
        plots: 要生成的图表列表（None 表示全部）
        top_n: 显示前 n 个模型
    """
    # 应用统一样式
    apply_bindstyle()

    # 准备完整数据
    complete_data = prepare_complete_data(data)

    # 确定要生成的图表
    if plots is None:
        plots_to_generate = list(AVAILABLE_PLOTS.keys())
    else:
        plots_to_generate = [p.strip() for p in plots.split(',')]

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 提取基础数据
    models = complete_data.get('models', SAMPLE_DATA['models'])
    f1_scores = complete_data.get('f1_scores', SAMPLE_DATA['f1_scores'])
    metrics_dict = {
        'f1_scores': complete_data.get('f1_scores', SAMPLE_DATA['f1_scores']),
        'precision': complete_data.get('precision', SAMPLE_DATA['precision']),
        'recall': complete_data.get('recall', SAMPLE_DATA['recall']),
        'accuracy': complete_data.get('accuracy', SAMPLE_DATA['accuracy']),
    }

    # 记录生成结果
    results = []
    print("\n" + "=" * 60)
    print("开始生成图表 / Starting plot generation")
    print("=" * 60)

    for plot_key in plots_to_generate:
        if plot_key not in AVAILABLE_PLOTS:
            print(f"警告: 未知图表类型 '{plot_key}'，跳过")
            continue

        plot_info = AVAILABLE_PLOTS[plot_key]
        print(f"\n正在生成: {plot_info['name']} ({plot_info['name_en']})")

        try:
            if plot_key == 'f1_ranking':
                plot_f1_ranking(
                    models=models,
                    f1_scores=f1_scores,
                    output_dir=output_dir,
                    filename=plot_info['filename']
                )

            elif plot_key == 'metrics_bar':
                plot_metrics_bar(
                    models=models,
                    metrics_dict=metrics_dict,
                    output_dir=output_dir,
                    filename=plot_info['filename']
                )

            elif plot_key == 'radar':
                plot_radar(
                    models=models,
                    metrics_dict=metrics_dict,
                    output_dir=output_dir,
                    filename=plot_info['filename'],
                    top_n=top_n
                )

            elif plot_key == 'heatmap':
                plot_heatmap(
                    models=models,
                    metrics_dict=metrics_dict,
                    output_dir=output_dir,
                    filename=plot_info['filename']
                )

            elif plot_key == 'roc':
                plot_roc(
                    roc_data=complete_data['roc_data'],
                    output_dir=output_dir,
                    filename=plot_info['filename'],
                    top_n=top_n
                )

            elif plot_key == 'pr':
                plot_pr(
                    pr_data=complete_data['pr_data'],
                    output_dir=output_dir,
                    filename=plot_info['filename'],
                    top_n=top_n
                )

            elif plot_key == 'confusion':
                plot_confusion_matrix(
                    confusion_matrix=complete_data['confusion_matrix'],
                    model_name=complete_data.get('model_name'),
                    output_dir=output_dir,
                    filename=plot_info['filename']
                )

            elif plot_key == 'training':
                plot_training_curve(
                    training_data=complete_data['training_data'],
                    output_dir=output_dir,
                    filename=plot_info['filename'],
                    top_n=top_n,
                    show_val=True
                )

            results.append({
                'plot': plot_key,
                'name': plot_info['name'],
                'filename': f"{plot_info['filename']}.png",
                'status': 'success'
            })
            print(f"  ✓ 成功保存: {plot_info['filename']}.png")

        except Exception as e:
            results.append({
                'plot': plot_key,
                'name': plot_info['name'],
                'filename': f"{plot_info['filename']}.png",
                'status': 'failed',
                'error': str(e)
            })
            print(f"  ✗ 生成失败: {e}")

    # 打印总结
    print("\n" + "=" * 60)
    print("生成完成 / Generation Complete")
    print("=" * 60)

    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)

    print(f"\n成功: {success_count}/{total_count}")
    print(f"输出目录: {output_path.absolute()}")

    # 列出生成的文件
    print("\n生成的文件:")
    for r in results:
        status = "✓" if r['status'] == 'success' else "✗"
        print(f"  {status} {r['filename']} - {r['name']}")

    # 保存生成报告
    report = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_directory': str(output_path.absolute()),
        'total_plots': total_count,
        'successful': success_count,
        'failed': total_count - success_count,
        'results': results
    }

    report_path = output_path / 'generation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n生成报告已保存: {report_path}")

    return results


def list_available_plots():
    """列出所有可用的图表类型"""
    print("\n可用图表类型 / Available Plot Types:")
    print("-" * 50)
    for key, info in AVAILABLE_PLOTS.items():
        print(f"  {key:15} - {info['name']} ({info['name_en']})")
    print("-" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='一键生成所有图表 / Generate All Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  # 使用示例数据生成所有图表
  python bindgenerate_bindall.py --output ./figures/

  # 使用自定义数据
  python bindgenerate_bindall.py --data results.json --output ./figures/

  # 仅生成指定图表
  python bindgenerate_bindall.py --plots f1_ranking,radar,roc --output ./figures/

  # 列出可用图表类型
  python bindgenerate_bindall.py --list
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='JSON 数据文件路径 / Path to JSON data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./figures',
        help='输出目录 / Output directory (default: ./figures)'
    )
    parser.add_argument(
        '--plots',
        type=str,
        default=None,
        help='要生成的图表（逗号分隔）/ Plots to generate (comma-separated)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='显示前 n 个模型 / Show top n models (default: 5)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出可用图表类型 / List available plot types'
    )

    args = parser.parse_args()

    # 列出可用图表
    if args.list:
        list_available_plots()
        return

    # 加载数据
    data = load_data(args.data)

    # 生成图表
    generate_all_plots(
        data=data,
        output_dir=args.output,
        plots=args.plots,
        top_n=args.top_n
    )


if __name__ == '__main__':
    main()
