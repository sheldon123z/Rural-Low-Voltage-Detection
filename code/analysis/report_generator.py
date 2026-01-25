"""
科学分析报告生成器

本模块提供 Markdown 格式的科学分析报告自动生成功能。

Author: Rural Voltage Detection Project
Date: 2026
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class ReportGenerator:
    """
    科学分析报告生成器

    生成符合学术规范的 Markdown 格式实验分析报告。
    """

    def __init__(self, template: str = "thesis"):
        """
        初始化报告生成器

        Args:
            template: 报告模板类型 ("thesis", "simple")
        """
        self.template = template

    def generate_markdown(
        self,
        results: Dict[str, Any],
        output_path: str,
        figure_dir: str = "./figures",
        experiment_info: Dict = None
    ) -> str:
        """
        生成 Markdown 格式报告

        Args:
            results: 实验结果字典
            output_path: 输出文件路径
            figure_dir: 图表目录（相对路径）
            experiment_info: 额外的实验信息

        Returns:
            str: 生成的报告内容
        """
        info = experiment_info or {}
        metrics = results.get("metrics", {})
        training_history = results.get("training_history", {})

        # 生成报告内容
        content = self._generate_header(info)
        content += self._generate_experiment_info(info, results)
        content += self._generate_experiment_settings(info)
        content += self._generate_results_section(metrics)
        content += self._generate_best_model_analysis(metrics)
        content += self._generate_model_comparison(metrics, training_history)
        content += self._generate_key_findings(metrics)
        content += self._generate_visualizations_section(figure_dir)
        content += self._generate_conclusions(metrics)
        content += self._generate_footer()

        # 保存报告
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content

    def _generate_header(self, info: Dict) -> str:
        """生成报告标题"""
        dataset = info.get("dataset", "未知数据集")
        exp_type = info.get("type", "对比实验")

        return f"""# {dataset}数据集{exp_type}实验分析报告

---

"""

    def _generate_experiment_info(self, info: Dict, results: Dict) -> str:
        """生成实验信息部分"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset = info.get("dataset", "未知")
        num_models = len(results.get("metrics", {}))

        return f"""## 实验信息

| 项目 | 内容 |
|------|------|
| **实验时间** | {timestamp} |
| **数据集** | {dataset} |
| **任务类型** | 时序异常检测 |
| **模型数量** | {num_models} |
| **评估指标** | 准确率、精确率、召回率、F1分数 |

---

"""

    def _generate_experiment_settings(self, info: Dict) -> str:
        """生成实验设置部分"""
        settings = info.get("settings", {})

        # 默认参数
        default_settings = {
            "seq_len": 100,
            "batch_size": 128,
            "train_epochs": 10,
            "learning_rate": 0.0001,
            "d_model": 64,
            "e_layers": 2,
            "patience": 3,
        }
        settings = {**default_settings, **settings}

        return f"""## 实验设置

### 公共参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 序列长度 | {settings.get('seq_len')} | 输入序列长度 |
| 批次大小 | {settings.get('batch_size')} | 训练批次大小 |
| 训练轮数 | {settings.get('train_epochs')} | 最大训练轮数 |
| 学习率 | {settings.get('learning_rate')} | 初始学习率 |
| 模型维度 | {settings.get('d_model')} | 隐藏层维度 |
| 编码器层数 | {settings.get('e_layers')} | Transformer编码器层数 |
| 早停耐心 | {settings.get('patience')} | 验证损失不下降的容忍轮数 |

---

"""

    def _generate_results_section(self, metrics: Dict) -> str:
        """生成结果汇总部分"""
        if not metrics:
            return "## 实验结果\n\n暂无结果数据。\n\n---\n\n"

        # 按 F1 分数排序
        sorted_models = sorted(
            metrics.items(),
            key=lambda x: x[1].get("f1_score", 0),
            reverse=True
        )

        content = "## 实验结果\n\n### 性能指标汇总\n\n"
        content += "| 排名 | 模型 | 准确率 | 精确率 | 召回率 | F1分数 |\n"
        content += "|:----:|:-----|:------:|:------:|:------:|:------:|\n"

        for i, (model, m) in enumerate(sorted_models, 1):
            acc = m.get("accuracy", 0)
            prec = m.get("precision", 0)
            rec = m.get("recall", 0)
            f1 = m.get("f1_score", 0)
            content += f"| {i} | {model} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"

        content += "\n---\n\n"
        return content

    def _generate_best_model_analysis(self, metrics: Dict) -> str:
        """生成最佳模型分析"""
        if not metrics:
            return ""

        # 找出最佳模型
        best_model = max(metrics.items(), key=lambda x: x[1].get("f1_score", 0))
        model_name, m = best_model

        content = f"""### 最佳模型分析

最佳模型为 **{model_name}**，F1 分数达到 **{m.get('f1_score', 0):.4f}**。

#### 性能特点

- **准确率**: {m.get('accuracy', 0):.4f} - 整体预测正确率，表示模型在所有样本上的预测准确程度
- **精确率**: {m.get('precision', 0):.4f} - 预测为异常中实际为异常的比例，反映模型的异常检测准确性
- **召回率**: {m.get('recall', 0):.4f} - 实际异常中被正确检测的比例，反映模型的异常覆盖能力
- **F1分数**: {m.get('f1_score', 0):.4f} - 精确率和召回率的调和平均，综合评估模型性能

"""
        return content

    def _generate_model_comparison(self, metrics: Dict, training_history: Dict) -> str:
        """生成模型对比分析"""
        if len(metrics) < 2:
            return ""

        content = "### 模型对比分析\n\n"

        # 按系列分组
        timesnet_models = [m for m in metrics.keys() if "TimesNet" in m]
        other_models = [m for m in metrics.keys() if "TimesNet" not in m]

        if timesnet_models:
            content += "#### TimesNet 系列模型\n\n"
            for model in timesnet_models:
                m = metrics[model]
                content += f"- **{model}**: F1={m.get('f1_score', 0):.4f}, "
                content += f"Precision={m.get('precision', 0):.4f}, "
                content += f"Recall={m.get('recall', 0):.4f}\n"
            content += "\n"

        if other_models:
            content += "#### 其他基线模型\n\n"
            for model in other_models:
                m = metrics[model]
                content += f"- **{model}**: F1={m.get('f1_score', 0):.4f}, "
                content += f"Precision={m.get('precision', 0):.4f}, "
                content += f"Recall={m.get('recall', 0):.4f}\n"
            content += "\n"

        content += "---\n\n"
        return content

    def _generate_key_findings(self, metrics: Dict) -> str:
        """生成关键发现"""
        if not metrics:
            return ""

        sorted_models = sorted(
            metrics.items(),
            key=lambda x: x[1].get("f1_score", 0),
            reverse=True
        )

        best_model, best_metrics = sorted_models[0]
        worst_model, worst_metrics = sorted_models[-1]

        f1_diff = best_metrics.get("f1_score", 0) - worst_metrics.get("f1_score", 0)

        # 找出精确率和召回率最高的模型
        best_precision_model = max(metrics.items(), key=lambda x: x[1].get("precision", 0))
        best_recall_model = max(metrics.items(), key=lambda x: x[1].get("recall", 0))

        content = f"""### 关键发现

1. **最佳性能模型**: {best_model} 在 F1 分数上表现最优，达到 {best_metrics.get('f1_score', 0):.4f}

2. **性能差距**: 最佳模型与最差模型（{worst_model}）的 F1 分数差距为 {f1_diff:.4f}

3. **精确率最优**: {best_precision_model[0]} 的精确率最高（{best_precision_model[1].get('precision', 0):.4f}），适合对误报敏感的场景

4. **召回率最优**: {best_recall_model[0]} 的召回率最高（{best_recall_model[1].get('recall', 0):.4f}），适合对漏报敏感的场景

5. **综合建议**: 对于农村电压异常检测任务，推荐使用 {best_model} 模型

---

"""
        return content

    def _generate_visualizations_section(self, figure_dir: str) -> str:
        """生成可视化结果部分"""
        return f"""## 可视化结果

### 训练过程

- `{figure_dir}/训练曲线对比.png` - 训练和验证损失变化曲线

### 性能对比

- `{figure_dir}/性能指标对比.png` - 四项指标分组柱状图
- `{figure_dir}/雷达图对比.png` - 多维性能雷达图
- `{figure_dir}/F1分数对比.png` - F1分数排名柱状图
- `{figure_dir}/性能热力图.png` - 模型×指标热力图

### 分类评估

- `{figure_dir}/ROC曲线.png` - ROC曲线对比
- `{figure_dir}/PR曲线.png` - PR曲线对比
- `{figure_dir}/混淆矩阵.png` - 混淆矩阵热力图

### 异常检测分析

- `{figure_dir}/重构误差分布.png` - 训练集和测试集误差分布
- `{figure_dir}/阈值敏感性分析.png` - 阈值对性能的影响

---

"""

    def _generate_conclusions(self, metrics: Dict) -> str:
        """生成结论与建议"""
        if not metrics:
            return "## 结论与建议\n\n暂无足够数据生成结论。\n\n"

        best_model = max(metrics.items(), key=lambda x: x[1].get("f1_score", 0))
        model_name, m = best_model

        content = f"""## 结论与建议

### 主要结论

1. 在本次实验中，**{model_name}** 模型表现最优，F1 分数达到 {m.get('f1_score', 0):.4f}

2. 基于重构误差的异常检测方法在电压异常检测任务上表现良好

3. TimesNet 系列模型通过 FFT 周期发现和 2D 卷积建模，能够有效捕捉电压信号的周期性模式

### 建议

1. **模型选择**: 推荐在实际应用中使用 {model_name} 模型进行电压异常检测

2. **阈值调优**: 根据具体业务场景对误报和漏报的容忍度，适当调整异常检测阈值

3. **数据增强**: 可考虑增加更多异常类型的训练数据，以提升模型对复杂异常的识别能力

4. **模型融合**: 可尝试集成多个模型的预测结果，进一步提升检测性能

"""
        return content

    def _generate_footer(self) -> str:
        """生成报告页脚"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
---

*报告自动生成于 {timestamp}*
*生成工具: Rural Voltage Detection Analysis System*
"""

    def generate_summary(self, metrics: Dict) -> Dict[str, Any]:
        """
        生成结构化摘要

        Args:
            metrics: 模型指标字典

        Returns:
            Dict: 摘要信息
        """
        if not metrics:
            return {"error": "无有效指标数据"}

        sorted_models = sorted(
            metrics.items(),
            key=lambda x: x[1].get("f1_score", 0),
            reverse=True
        )

        best_model, best_metrics = sorted_models[0]

        return {
            "best_model": best_model,
            "best_f1": best_metrics.get("f1_score", 0),
            "best_accuracy": best_metrics.get("accuracy", 0),
            "num_models": len(metrics),
            "all_f1_scores": {m: metrics[m].get("f1_score", 0) for m in metrics},
            "ranking": [m for m, _ in sorted_models],
        }

    def generate_latex_table(self, metrics: Dict, caption: str = "模型性能对比") -> str:
        """
        生成 LaTeX 三线表

        Args:
            metrics: 模型指标字典
            caption: 表格标题

        Returns:
            str: LaTeX 表格代码
        """
        sorted_models = sorted(
            metrics.items(),
            key=lambda x: x[1].get("f1_score", 0),
            reverse=True
        )

        latex = r"""\begin{table}[htbp]
\centering
\caption{""" + caption + r"""}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
模型 & 准确率 & 精确率 & 召回率 & F1分数 \\
\midrule
"""
        for model, m in sorted_models:
            acc = m.get("accuracy", 0)
            prec = m.get("precision", 0)
            rec = m.get("recall", 0)
            f1 = m.get("f1_score", 0)
            latex += f"{model} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex
