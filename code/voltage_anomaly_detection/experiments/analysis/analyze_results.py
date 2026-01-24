"""
Results Analysis and Visualization for Voltage Anomaly Detection Experiments

This module provides utilities for:
1. Parsing experiment results from log files
2. Computing statistical comparisons
3. Generating tables and figures for the thesis

Usage:
    python analyze_results.py --results_dir ./experiment_results
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ResultParser:
    """Parse experiment results from log files and result directories."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        
    def parse_log_file(self, log_path: str) -> Dict:
        """Parse a single log file for metrics."""
        metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'threshold': None,
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
        }
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Parse final metrics
        acc_match = re.search(r'Accuracy:\s*([\d.]+)', content)
        prec_match = re.search(r'Precision:\s*([\d.]+)', content)
        rec_match = re.search(r'Recall:\s*([\d.]+)', content)
        f1_match = re.search(r'F1-score:\s*([\d.]+)', content)
        thresh_match = re.search(r'Threshold:\s*([\d.]+)', content)
        
        if acc_match:
            metrics['accuracy'] = float(acc_match.group(1))
        if prec_match:
            metrics['precision'] = float(prec_match.group(1))
        if rec_match:
            metrics['recall'] = float(rec_match.group(1))
        if f1_match:
            metrics['f1'] = float(f1_match.group(1))
        if thresh_match:
            metrics['threshold'] = float(thresh_match.group(1))
        
        # Parse epoch losses
        loss_pattern = r'Epoch:\s*\d+.*Train Loss:\s*([\d.]+).*Vali Loss:\s*([\d.]+).*Test Loss:\s*([\d.]+)'
        for match in re.finditer(loss_pattern, content):
            metrics['train_loss'].append(float(match.group(1)))
            metrics['val_loss'].append(float(match.group(2)))
            metrics['test_loss'].append(float(match.group(3)))
        
        return metrics
    
    def parse_all_results(self) -> pd.DataFrame:
        """Parse all results and return as DataFrame."""
        results = []
        
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if file.endswith('_log.txt'):
                    log_path = os.path.join(root, file)
                    
                    # Extract experiment info from filename
                    exp_name = file.replace('_log.txt', '')
                    
                    metrics = self.parse_log_file(log_path)
                    metrics['experiment'] = exp_name
                    metrics['category'] = os.path.basename(root)
                    
                    results.append(metrics)
        
        return pd.DataFrame(results)


class StatisticalAnalysis:
    """Statistical analysis of experiment results."""
    
    @staticmethod
    def compute_summary_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Compute summary statistics for a metric grouped by experiment."""
        summary = df.groupby('experiment')[metric].agg(['mean', 'std', 'min', 'max'])
        return summary.round(4)
    
    @staticmethod
    def paired_t_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
        """Perform paired t-test between two sets of scores."""
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        return t_stat, p_value
    
    @staticmethod
    def rank_models(df: pd.DataFrame, metric: str = 'f1', ascending: bool = False) -> pd.DataFrame:
        """Rank models by a metric."""
        ranked = df.sort_values(metric, ascending=ascending)
        ranked['rank'] = range(1, len(ranked) + 1)
        return ranked


class TableGenerator:
    """Generate tables for thesis."""
    
    @staticmethod
    def generate_baseline_comparison(df: pd.DataFrame) -> str:
        """Generate LaTeX table comparing baseline models."""
        # Filter baseline experiments
        baseline_df = df[df['category'] == 'baseline'].copy()
        
        # Select columns
        cols = ['experiment', 'accuracy', 'precision', 'recall', 'f1']
        table_df = baseline_df[cols].round(4)
        
        # Generate LaTeX
        latex = table_df.to_latex(
            index=False,
            caption='基线模型在农村电压数据集上的性能对比',
            label='tab:baseline_comparison',
            column_format='lcccc',
            escape=False
        )
        
        return latex
    
    @staticmethod
    def generate_ablation_table(df: pd.DataFrame, ablation_type: str) -> str:
        """Generate LaTeX table for ablation study."""
        # Filter ablation experiments
        ablation_df = df[df['experiment'].str.contains(ablation_type)].copy()
        
        cols = ['experiment', 'accuracy', 'precision', 'recall', 'f1']
        table_df = ablation_df[cols].round(4)
        
        latex = table_df.to_latex(
            index=False,
            caption=f'{ablation_type}消融实验结果',
            label=f'tab:ablation_{ablation_type}',
            escape=False
        )
        
        return latex


class FigureGenerator:
    """Generate figures for thesis."""
    
    def __init__(self, output_dir: str = './figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, metrics: Dict, save_name: str):
        """Plot training loss curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(metrics['train_loss']) + 1)
            
            ax.plot(epochs, metrics['train_loss'], label='训练损失', marker='o')
            ax.plot(epochs, metrics['val_loss'], label='验证损失', marker='s')
            ax.plot(epochs, metrics['test_loss'], label='测试损失', marker='^')
            
            ax.set_xlabel('轮次')
            ax.set_ylabel('损失')
            ax.set_title('训练过程损失曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.pdf'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.png'), dpi=300)
            plt.close()
            
        except ImportError:
            print("matplotlib not installed, skipping plot generation")
    
    def plot_model_comparison(self, df: pd.DataFrame, metric: str, save_name: str):
        """Plot bar chart comparing models."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = df['experiment'].values
            values = df[metric].values
            
            bars = ax.bar(range(len(models)), values, color='steelblue')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'模型{metric.upper()}对比')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.pdf'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.png'), dpi=300)
            plt.close()
            
        except ImportError:
            print("matplotlib not installed, skipping plot generation")
    
    def plot_ablation_heatmap(self, df: pd.DataFrame, save_name: str):
        """Plot heatmap for ablation study."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Pivot the data for heatmap
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            heatmap_data = df.set_index('experiment')[metrics]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax)
            
            ax.set_title('消融实验结果热力图')
            ax.set_xlabel('评价指标')
            ax.set_ylabel('实验配置')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.pdf'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.png'), dpi=300)
            plt.close()
            
        except ImportError:
            print("matplotlib/seaborn not installed, skipping plot generation")


def generate_thesis_tables(df: pd.DataFrame, output_dir: str):
    """Generate all tables for thesis."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = TableGenerator()
    
    # Baseline comparison table
    baseline_latex = generator.generate_baseline_comparison(df)
    with open(os.path.join(output_dir, 'baseline_comparison.tex'), 'w') as f:
        f.write(baseline_latex)
    
    # Ablation tables
    for ablation_type in ['seqlen', 'dmodel', 'layers', 'topk', 'anomaly']:
        ablation_latex = generator.generate_ablation_table(df, ablation_type)
        with open(os.path.join(output_dir, f'ablation_{ablation_type}.tex'), 'w') as f:
            f.write(ablation_latex)
    
    print(f"Tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                        help='Output directory for tables and figures')
    parser.add_argument('--generate_figures', action='store_true',
                        help='Generate visualization figures')
    
    args = parser.parse_args()
    
    # Parse results
    print(f"Parsing results from {args.results_dir}...")
    parser = ResultParser(args.results_dir)
    df = parser.parse_all_results()
    
    if df.empty:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Found {len(df)} experiment results")
    
    # Generate summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"\n{metric.upper()}:")
        print(df[['experiment', metric]].dropna().to_string(index=False))
    
    # Generate tables
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results to CSV
    df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)
    print(f"\nResults saved to {args.output_dir}/all_results.csv")
    
    # Generate LaTeX tables
    tables_dir = os.path.join(args.output_dir, 'tables')
    generate_thesis_tables(df, tables_dir)
    
    # Generate figures if requested
    if args.generate_figures:
        fig_gen = FigureGenerator(os.path.join(args.output_dir, 'figures'))
        
        # Model comparison
        fig_gen.plot_model_comparison(df, 'f1', 'model_f1_comparison')
        
        print(f"Figures saved to {args.output_dir}/figures")
    
    # Rank models
    print("\n" + "="*60)
    print("Model Rankings (by F1-score)")
    print("="*60)
    
    ranked = StatisticalAnalysis.rank_models(df, 'f1')
    print(ranked[['rank', 'experiment', 'f1']].to_string(index=False))


if __name__ == '__main__':
    main()
