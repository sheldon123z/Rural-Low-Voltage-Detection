#!/usr/bin/env python3
"""
VoltageTimesNet_v2 超参数优化脚本
使用 Optuna 框架进行贝叶斯优化

用法:
    python scripts/optuna_optimize_voltage.py --n-trials 50 --study-name voltage_opt
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection


class VoltageTimesNetOptimizer:
    """VoltageTimesNet_v2 超参数优化器"""

    def __init__(self, data_path, device='cuda:0'):
        self.data_path = data_path
        self.device = device
        self.best_f1 = 0.0
        self.best_params = {}

    def create_args(self, trial):
        """根据 Optuna trial 创建模型参数"""

        # ============================================================
        # 搜索空间定义
        # ============================================================

        # 模型架构参数
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        e_layers = trial.suggest_int("e_layers", 1, 3)
        d_ff = trial.suggest_categorical("d_ff", [64, 128, 256, 512])
        top_k = trial.suggest_int("top_k", 3, 7)
        num_kernels = trial.suggest_categorical("num_kernels", [4, 6, 8])

        # 训练参数
        learning_rate = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        # 序列长度
        seq_len = trial.suggest_categorical("seq_len", [50, 100, 200])

        # 异常检测阈值比例
        anomaly_ratio = trial.suggest_float("anomaly_ratio", 1.0, 5.0)

        # 创建参数命名空间
        class Args:
            pass

        args = Args()

        # 基础设置
        args.task_name = 'anomaly_detection'  # 关键：必须设置
        args.is_training = 1
        args.model_id = f'optuna_trial_{trial.number}'
        args.model = 'VoltageTimesNet_v2'
        args.data = 'RuralVoltage'
        args.root_path = self.data_path
        args.data_path = ''
        args.features = 'M'
        args.target = 'OT'
        args.freq = 'h'
        args.checkpoints = './checkpoints/'
        args.seed = 42

        # 模型参数
        args.seq_len = seq_len
        args.label_len = 48
        args.pred_len = 0
        args.enc_in = 16
        args.dec_in = 16
        args.c_out = 16
        args.d_model = d_model
        args.n_heads = 8
        args.e_layers = e_layers
        args.d_layers = 1
        args.d_ff = d_ff
        args.factor = 3
        args.embed = 'timeF'
        args.distil = True
        args.dropout = dropout
        args.activation = 'gelu'
        args.output_attention = False

        # TimesNet 特定参数
        args.top_k = top_k
        args.num_kernels = num_kernels

        # 训练参数
        args.num_workers = 0
        args.itr = 1
        args.train_epochs = 5  # 快速评估
        args.batch_size = batch_size
        args.patience = 3
        args.learning_rate = learning_rate
        args.des = f'optuna_trial_{trial.number}'
        args.loss = 'MSE'
        args.lradj = 'type1'
        args.use_amp = False

        # 异常检测参数
        args.anomaly_ratio = anomaly_ratio
        args.win_size = seq_len
        args.step = 1

        # GPU 设置
        args.use_gpu = True
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0'

        # 额外必需参数
        args.moving_avg = 25
        args.preset_weight = 0.3

        return args

    def objective(self, trial):
        """Optuna 目标函数"""

        try:
            # 创建参数
            args = self.create_args(trial)

            # 创建实验
            exp = Exp_Anomaly_Detection(args)

            # 训练
            print(f"\n[Trial {trial.number}] 开始训练...")
            print(f"  参数: d_model={args.d_model}, e_layers={args.e_layers}, "
                  f"lr={args.learning_rate:.6f}, batch_size={args.batch_size}")

            setting = f'{args.model}_{args.data}_trial{trial.number}'
            exp.train(setting)

            # 测试并获取 F1 分数
            metrics = exp.test(setting, test=1)

            # 获取 F1 分数
            if isinstance(metrics, dict):
                f1 = metrics.get('f1', metrics.get('f_score', 0.0))
            elif isinstance(metrics, (list, tuple)):
                # 假设返回格式为 [accuracy, precision, recall, f1]
                f1 = metrics[3] if len(metrics) > 3 else metrics[-1]
            else:
                f1 = float(metrics) if metrics else 0.0

            print(f"[Trial {trial.number}] F1 = {f1:.4f}")

            # 记录最佳结果
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_params = {
                    'd_model': args.d_model,
                    'e_layers': args.e_layers,
                    'd_ff': args.d_ff,
                    'top_k': args.top_k,
                    'num_kernels': args.num_kernels,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'dropout': args.dropout,
                    'seq_len': args.seq_len,
                    'anomaly_ratio': args.anomaly_ratio,
                }
                print(f"[Trial {trial.number}] ★ 新的最佳结果! F1 = {f1:.4f}")

            # 清理 GPU 内存
            del exp
            torch.cuda.empty_cache()

            return f1

        except Exception as e:
            print(f"[Trial {trial.number}] 错误: {e}")
            torch.cuda.empty_cache()
            return 0.0

    def optimize(self, n_trials=50, study_name='voltage_optimization'):
        """运行超参数优化"""

        # 创建 Optuna study
        sampler = TPESampler(seed=42, multivariate=True)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # 最大化 F1
            sampler=sampler,
            pruner=pruner,
        )

        print("=" * 60)
        print("VoltageTimesNet_v2 超参数优化")
        print("=" * 60)
        print(f"优化目标: F1 分数 (最大化)")
        print(f"试验次数: {n_trials}")
        print(f"采样器: TPE (Tree-structured Parzen Estimator)")
        print("=" * 60)

        # 运行优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # 输出结果
        print("\n" + "=" * 60)
        print("优化完成!")
        print("=" * 60)
        print(f"最佳 F1 分数: {study.best_value:.4f}")
        print(f"最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study


def main():
    parser = argparse.ArgumentParser(description='VoltageTimesNet_v2 超参数优化')
    parser.add_argument('--n-trials', type=int, default=50, help='优化试验次数')
    parser.add_argument('--study-name', type=str, default='voltage_opt', help='Study 名称')
    parser.add_argument('--data-path', type=str,
                        default='./dataset/RuralVoltage/realistic_v2/',
                        help='数据集路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    parser.add_argument('--output-dir', type=str, default='./results/optuna',
                        help='结果输出目录')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建优化器
    optimizer = VoltageTimesNetOptimizer(
        data_path=args.data_path,
        device=args.device,
    )

    # 运行优化
    study = optimizer.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name,
    )

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(args.output_dir, f'optuna_results_{timestamp}.json')

    results = {
        'best_f1': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials,
        'timestamp': timestamp,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ]
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {result_file}")

    # 生成可视化 (如果安装了 plotly)
    try:
        import plotly

        # 优化历史
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(args.output_dir, f'optimization_history_{timestamp}.html'))

        # 参数重要性
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(args.output_dir, f'param_importances_{timestamp}.html'))

        # 参数关系
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(args.output_dir, f'parallel_coordinate_{timestamp}.html'))

        print(f"可视化图表已保存到: {args.output_dir}")

    except ImportError:
        print("提示: 安装 plotly 可生成交互式可视化图表")

    return study


if __name__ == '__main__':
    main()
