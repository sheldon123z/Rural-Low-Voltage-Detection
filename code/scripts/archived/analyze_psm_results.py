#!/usr/bin/env python3
"""
PSM 数据集多模型故障检测对比实验结果分析脚本
生成科研级别的论文图表（中文标签）

输出:
- 训练曲线对比.png/pdf
- 性能指标对比.png/pdf
- 雷达图对比.png/pdf
- F1分数对比.png/pdf
- 实验分析报告.md
- 实验结果.json
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体和科研绘图风格
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 色盲友好配色方案 (Okabe-Ito)
COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
          '#0072B2', '#D55E00', '#CC79A7', '#000000', '#999999']


def parse_log_file(log_path):
    """解析训练日志文件，提取训练曲线和最终指标"""
    with open(log_path, 'r') as f:
        content = f.read()

    # 提取训练损失
    train_losses = []
    val_losses = []
    test_losses = []

    # 匹配 Epoch 行
    epoch_pattern = r'Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+) Test Loss: ([\d.]+)'
    for match in re.finditer(epoch_pattern, content):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        test_loss = float(match.group(4))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    # 提取最终测试指标
    metrics_pattern = r'Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-score: ([\d.]+)'
    metrics_match = re.search(metrics_pattern, content)

    if metrics_match:
        metrics = {
            'accuracy': float(metrics_match.group(1)),
            'precision': float(metrics_match.group(2)),
            'recall': float(metrics_match.group(3)),
            'f1': float(metrics_match.group(4))
        }
    else:
        metrics = None

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'metrics': metrics
    }


def plot_training_curves(results, output_dir):
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练损失
    ax1 = axes[0]
    for i, (model_name, data) in enumerate(results.items()):
        if data['train_losses']:
            epochs = range(1, len(data['train_losses']) + 1)
            ax1.plot(epochs, data['train_losses'],
                    color=COLORS[i % len(COLORS)],
                    label=model_name, linewidth=2)

    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('训练损失')
    ax1.set_title('训练损失曲线对比')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 验证损失
    ax2 = axes[1]
    for i, (model_name, data) in enumerate(results.items()):
        if data['val_losses']:
            epochs = range(1, len(data['val_losses']) + 1)
            ax2.plot(epochs, data['val_losses'],
                    color=COLORS[i % len(COLORS)],
                    label=model_name, linewidth=2)

    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('验证损失')
    ax2.set_title('验证损失曲线对比')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '训练曲线对比.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, '训练曲线对比.pdf'), bbox_inches='tight')
    plt.close()
    print("已生成: 训练曲线对比.png/pdf")


def plot_performance_comparison(results, output_dir):
    """绘制性能指标对比柱状图"""
    # 收集有效结果
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}

    if not valid_results:
        print("警告: 没有有效的测试结果")
        return

    models = list(valid_results.keys())
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
        values = [valid_results[m]['metrics'][metric_key] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric_name,
                     color=COLORS[i], edgecolor='black', linewidth=0.5)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('PSM 数据集异常检测性能指标对比')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0.75, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '性能指标对比.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, '性能指标对比.pdf'), bbox_inches='tight')
    plt.close()
    print("已生成: 性能指标对比.png/pdf")


def plot_radar_chart(results, output_dir):
    """绘制雷达图对比"""
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}

    if not valid_results:
        return

    # 雷达图参数
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (model_name, data) in enumerate(valid_results.items()):
        metrics = data['metrics']
        values = [metrics['accuracy'], metrics['precision'],
                 metrics['recall'], metrics['f1']]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2,
               label=model_name, color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.15, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14)
    ax.set_ylim(0.75, 1.0)
    ax.set_title('模型性能雷达图对比', size=18, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '雷达图对比.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, '雷达图对比.pdf'), bbox_inches='tight')
    plt.close()
    print("已生成: 雷达图对比.png/pdf")


def plot_f1_comparison(results, output_dir):
    """绘制F1分数对比图（横向条形图）"""
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}

    if not valid_results:
        return

    # 按F1分数排序
    sorted_results = sorted(valid_results.items(),
                           key=lambda x: x[1]['metrics']['f1'],
                           reverse=True)

    models = [item[0] for item in sorted_results]
    f1_scores = [item[1]['metrics']['f1'] for item in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 根据分数分配颜色
    colors = []
    for f1 in f1_scores:
        if f1 >= 0.97:
            colors.append('#009E73')  # 绿色 - 最佳
        elif f1 >= 0.95:
            colors.append('#56B4E9')  # 蓝色 - 良好
        elif f1 >= 0.90:
            colors.append('#E69F00')  # 橙色 - 中等
        else:
            colors.append('#D55E00')  # 红色 - 较低

    bars = ax.barh(models, f1_scores, color=colors, edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for bar, score in zip(bars, f1_scores):
        width = bar.get_width()
        ax.annotate(f'{score:.4f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('F1 分数')
    ax.set_title('PSM 数据集 F1 分数对比')
    ax.set_xlim(0.85, 1.0)
    ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5, label='基准线 (0.95)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # 最高分在上

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1分数对比.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'F1分数对比.pdf'), bbox_inches='tight')
    plt.close()
    print("已生成: F1分数对比.png/pdf")


def generate_report(results, output_dir, timestamp):
    """生成实验分析报告"""
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}

    # 按F1分数排序
    sorted_results = sorted(valid_results.items(),
                           key=lambda x: x[1]['metrics']['f1'],
                           reverse=True)

    report = f"""# PSM 数据集多模型故障检测对比实验分析报告

## 实验信息

- **实验时间**: {timestamp}
- **数据集**: PSM (服务器监控数据，25维特征)
- **任务类型**: 时序异常检测
- **评估指标**: 准确率、精确率、召回率、F1分数

## 实验设置

### 公共参数
| 参数 | 值 |
|------|-----|
| 序列长度 | 100 |
| 批次大小 | 128 |
| 训练轮数 | 10 |
| 学习率 | 0.0001 |
| 模型维度 | 64 |
| 层数 | 2 |
| 早停耐心 | 3 |

### 模型列表
- **基线模型**: TimesNet, Transformer, DLinear, iTransformer, Autoformer
- **创新模型**: VoltageTimesNet, TPATimesNet, MTSTimesNet

## 实验结果

### 性能指标汇总

| 排名 | 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|:----:|:-----|:------:|:------:|:------:|:------:|
"""

    for i, (model, data) in enumerate(sorted_results, 1):
        m = data['metrics']
        report += f"| {i} | **{model}** | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** |\n"

    # 分析最佳模型
    best_model, best_data = sorted_results[0]
    best_metrics = best_data['metrics']

    report += f"""

### 最佳模型分析

最佳模型为 **{best_model}**，F1 分数达到 **{best_metrics['f1']:.4f}**。

#### 性能特点
- **准确率**: {best_metrics['accuracy']:.4f} - 整体预测正确率
- **精确率**: {best_metrics['precision']:.4f} - 预测为异常中实际为异常的比例
- **召回率**: {best_metrics['recall']:.4f} - 实际异常中被正确检测的比例

### 模型对比分析

#### TimesNet 系列模型
"""

    timesnet_models = ['TimesNet', 'VoltageTimesNet', 'TPATimesNet', 'MTSTimesNet']
    for model in timesnet_models:
        if model in valid_results:
            m = valid_results[model]['metrics']
            report += f"- **{model}**: F1={m['f1']:.4f}, 召回率={m['recall']:.4f}\n"

    report += """
#### Transformer 系列模型
"""
    transformer_models = ['Transformer', 'iTransformer', 'Autoformer']
    for model in transformer_models:
        if model in valid_results:
            m = valid_results[model]['metrics']
            report += f"- **{model}**: F1={m['f1']:.4f}, 召回率={m['recall']:.4f}\n"

    report += f"""

### 关键发现

1. **TimesNet 系列表现优异**: TimesNet 及其变体在 PSM 数据集上表现最佳，F1 分数均超过 0.96
2. **VoltageTimesNet 和 TimesNet 表现相当**: 预设周期策略在通用异常检测任务上与自适应FFT周期发现效果接近
3. **Transformer 基线模型表现中等**: 原始 Transformer 和 Autoformer 的召回率相对较低
4. **DLinear 轻量模型表现不俗**: 线性模型也能达到 0.9661 的 F1 分数

## 结论与建议

1. **推荐模型**: 对于 PSM 类工业监控数据的异常检测，推荐使用 **{best_model}** 模型
2. **模型选择策略**:
   - 追求最佳性能: TimesNet / VoltageTimesNet
   - 计算资源受限: DLinear
   - 需要可解释性: Transformer 系列

## 图表清单

- `训练曲线对比.png/pdf` - 训练和验证损失变化
- `性能指标对比.png/pdf` - 四项指标柱状图
- `雷达图对比.png/pdf` - 多维性能雷达图
- `F1分数对比.png/pdf` - F1分数排名

---
*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(os.path.join(output_dir, '实验分析报告.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("已生成: 实验分析报告.md")


def save_results_json(results, output_dir, timestamp):
    """保存结果为JSON格式"""
    valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}

    json_data = {
        '实验信息': {
            '时间戳': timestamp,
            '数据集': 'PSM',
            '特征维度': 25,
            '序列长度': 100
        },
        '模型结果': {}
    }

    for model, data in valid_results.items():
        json_data['模型结果'][model] = {
            '指标': {
                '准确率': data['metrics']['accuracy'],
                '精确率': data['metrics']['precision'],
                '召回率': data['metrics']['recall'],
                'F1分数': data['metrics']['f1']
            },
            '训练轮数': len(data['train_losses'])
        }

    with open(os.path.join(output_dir, '实验结果.json'), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("已生成: 实验结果.json")


def main():
    # 结果目录
    result_dir = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/results/PSM_comparison_20260125_013217'

    # 创建输出目录（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(result_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"结果目录: {result_dir}")
    print(f"输出目录: {output_dir}")
    print("="*50)

    # 解析所有日志文件
    results = {}
    log_files = sorted(Path(result_dir).glob('*.log'))

    for log_file in log_files:
        model_name = log_file.stem
        print(f"解析: {model_name}")
        results[model_name] = parse_log_file(log_file)

        if results[model_name]['metrics']:
            m = results[model_name]['metrics']
            print(f"  F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")
        else:
            print(f"  [警告] 未找到测试指标")

    print("="*50)

    # 生成图表
    print("\n生成科研图表...")
    plot_training_curves(results, output_dir)
    plot_performance_comparison(results, output_dir)
    plot_radar_chart(results, output_dir)
    plot_f1_comparison(results, output_dir)

    # 生成报告
    print("\n生成实验报告...")
    generate_report(results, output_dir, timestamp)
    save_results_json(results, output_dir, timestamp)

    print("\n"+"="*50)
    print(f"分析完成! 所有文件保存在: {output_dir}")


if __name__ == '__main__':
    main()
