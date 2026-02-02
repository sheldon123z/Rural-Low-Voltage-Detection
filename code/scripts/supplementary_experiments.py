#!/usr/bin/env python3
"""
Supplementary Experiments for Rural Voltage Anomaly Detection Paper

This script runs the following experiments:
1. Classic baseline models comparison (LSTM-AE, Isolation Forest, One-Class SVM)
2. Statistical significance tests (paired t-test, Wilcoxon)
3. Hyperparameter optimization using Optuna
4. Save best models

Author: Rural Voltage Anomaly Detection Project
Date: 2026-02
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM as SklearnOCSVM

from data_provider.data_factory import data_provider
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.tools import adjustment

warnings.filterwarnings("ignore")


class ExperimentConfig:
    """Configuration for experiments."""
    
    def __init__(self):
        # Basic settings
        self.data = "RuralVoltage"
        self.root_path = "./dataset/RuralVoltage/realistic_v2/"
        self.enc_in = 16
        self.c_out = 16
        self.seq_len = 100
        self.d_model = 64
        self.e_layers = 2
        self.top_k = 5
        self.num_kernels = 6
        self.d_ff = 256
        self.dropout = 0.1
        self.embed = "timeF"
        self.freq = "h"
        
        # Training settings
        self.train_epochs = 10
        self.batch_size = 128
        self.patience = 3
        self.learning_rate = 0.0001
        self.anomaly_ratio = 3.0
        
        # Device settings
        self.use_gpu = torch.cuda.is_available()
        self.gpu = 0
        self.use_multi_gpu = False
        self.device_ids = [0]
        
        # Task settings
        self.task_name = "anomaly_detection"
        self.features = "M"
        self.target = "OT"
        self.checkpoints = "./checkpoints/"
        self.num_workers = 0  # Avoid deadlock
        
        # Moving average for decomposition
        self.moving_avg = 25


def create_args_from_config(config, model_name, model_id=None):
    """Create argparse namespace from config."""
    args = argparse.Namespace()
    for key, value in vars(config).items():
        setattr(args, key, value)
    args.model = model_name
    args.model_id = model_id or f"{model_name}_{config.data}"
    args.is_training = 1
    return args


class SupplementaryExperiments:
    """Main class for running supplementary experiments."""
    
    def __init__(self, output_dir="./results/supplementary_experiments"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.config = ExperimentConfig()
        self.results = {}
        
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # Also write to log file
        log_file = os.path.join(self.results_dir, "experiment_log.txt")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def run_single_experiment(self, model_name, seed=42, epochs=None):
        """
        Run a single experiment for a given model.
        
        Returns:
            dict: Metrics dictionary with accuracy, precision, recall, f1
        """
        self.log(f"Running experiment: {model_name} (seed={seed})")
        
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create args
        args = create_args_from_config(self.config, model_name)
        args.seed = seed
        if epochs:
            args.train_epochs = epochs
        
        # Create experiment
        setting = f"{model_name}_{self.config.data}_seed{seed}_{self.timestamp}"
        
        try:
            exp = Exp_Anomaly_Detection(args)
            
            # Train
            self.log(f"  Training {model_name}...")
            exp.train(setting)
            
            # Test
            self.log(f"  Testing {model_name}...")
            metrics = exp.test(setting, test=0)
            
            self.log(f"  Results: Acc={metrics['accuracy']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                    f"F1={metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.log(f"  ERROR: {str(e)}")
            return None
    
    def run_sklearn_baseline(self, model_type, seed=42):
        """
        Run sklearn-based baseline (Isolation Forest or One-Class SVM).
        
        These models require special handling as they don't use reconstruction.
        """
        self.log(f"Running sklearn baseline: {model_type} (seed={seed})")
        
        # Set seed
        np.random.seed(seed)
        
        # Create args for data loading
        args = create_args_from_config(self.config, "DLinear")  # Dummy model
        
        # Load data
        train_data, train_loader = data_provider(args, "train")
        test_data, test_loader = data_provider(args, "test")
        
        # Collect training data
        self.log(f"  Collecting training data...")
        train_features = []
        for batch_x, _ in train_loader:
            train_features.append(batch_x.numpy())
        train_features = np.concatenate(train_features, axis=0)
        train_features = train_features.reshape(train_features.shape[0], -1)
        
        # Collect test data
        self.log(f"  Collecting test data...")
        test_features = []
        test_labels = []
        for batch_x, batch_y in test_loader:
            test_features.append(batch_x.numpy())
            test_labels.append(batch_y.numpy())
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_features = test_features.reshape(test_features.shape[0], -1)
        test_labels = test_labels.reshape(-1)
        
        # Convert multi-class labels to binary
        test_labels_binary = (test_labels > 0).astype(int)
        
        # Fit and predict
        contamination = self.config.anomaly_ratio / 100.0
        contamination = min(contamination, 0.5)  # sklearn limit
        
        if model_type == "IsolationForest":
            self.log(f"  Fitting Isolation Forest...")
            model = SklearnIF(
                n_estimators=100,
                contamination=contamination,
                random_state=seed,
                n_jobs=-1
            )
        else:  # OneClassSVM
            self.log(f"  Fitting One-Class SVM...")
            # Subsample for SVM (memory intensive)
            max_samples = min(5000, train_features.shape[0])
            indices = np.random.choice(train_features.shape[0], max_samples, replace=False)
            train_features_subset = train_features[indices]
            
            model = SklearnOCSVM(
                kernel='rbf',
                gamma='scale',
                nu=contamination
            )
            train_features = train_features_subset
        
        # Fit
        model.fit(train_features)
        
        # Predict (-1 for anomaly, 1 for normal in sklearn)
        self.log(f"  Predicting...")
        predictions = model.predict(test_features)
        predictions = (predictions == -1).astype(int)  # Convert to 0/1
        
        # Apply point adjustment
        gt, pred = adjustment(test_labels_binary, predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(gt, pred)
        precision = precision_score(gt, pred, zero_division=0)
        recall = recall_score(gt, pred, zero_division=0)
        f1 = f1_score(gt, pred, zero_division=0)
        
        self.log(f"  Results: Acc={accuracy:.4f}, P={precision:.4f}, "
                f"R={recall:.4f}, F1={f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def experiment_1_baseline_comparison(self):
        """
        Experiment 1: Classic baseline models comparison.
        
        Models: LSTM-AutoEncoder, Isolation Forest, One-Class SVM
        Plus existing models: DLinear, TimesNet, VoltageTimesNet_v2
        """
        self.log("=" * 60)
        self.log("EXPERIMENT 1: Classic Baseline Models Comparison")
        self.log("=" * 60)
        
        results = {}
        
        # Deep learning baselines
        dl_models = ["LSTMAutoEncoder", "DLinear", "TimesNet", "VoltageTimesNet_v2"]
        for model in dl_models:
            metrics = self.run_single_experiment(model, seed=42)
            if metrics:
                results[model] = metrics
        
        # sklearn baselines
        for model in ["IsolationForest", "OneClassSVM"]:
            metrics = self.run_sklearn_baseline(model, seed=42)
            if metrics:
                results[model] = metrics
        
        # Save results
        self.results["baseline_comparison"] = results
        self._save_results("baseline_comparison.json", results)
        
        # Create comparison table
        self._create_comparison_table(results, "baseline_comparison")
        
        return results
    
    def experiment_2_statistical_significance(self, n_runs=5):
        """
        Experiment 2: Statistical significance tests.
        
        Run each model multiple times and perform paired t-test.
        """
        self.log("=" * 60)
        self.log(f"EXPERIMENT 2: Statistical Significance Tests (n={n_runs})")
        self.log("=" * 60)
        
        models = ["VoltageTimesNet_v2", "TimesNet", "DLinear"]
        all_results = {model: [] for model in models}
        
        # Run multiple times
        for seed in range(42, 42 + n_runs):
            self.log(f"\n--- Run {seed - 41}/{n_runs} (seed={seed}) ---")
            for model in models:
                metrics = self.run_single_experiment(model, seed=seed, epochs=5)
                if metrics:
                    all_results[model].append(metrics["f1"])
        
        # Statistical tests
        significance_results = {}
        
        # VoltageTimesNet_v2 vs TimesNet
        if len(all_results["VoltageTimesNet_v2"]) > 1 and len(all_results["TimesNet"]) > 1:
            t_stat, p_value = stats.ttest_rel(
                all_results["VoltageTimesNet_v2"],
                all_results["TimesNet"]
            )
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                all_results["VoltageTimesNet_v2"],
                all_results["TimesNet"]
            )
            significance_results["VoltageTimesNet_v2_vs_TimesNet"] = {
                "t_statistic": t_stat,
                "t_p_value": p_value,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "significant_005": p_value < 0.05
            }
            self.log(f"\nVoltageTimesNet_v2 vs TimesNet:")
            self.log(f"  t-test: t={t_stat:.4f}, p={p_value:.4f}")
            self.log(f"  Wilcoxon: stat={wilcoxon_stat:.4f}, p={wilcoxon_p:.4f}")
        
        # VoltageTimesNet_v2 vs DLinear
        if len(all_results["VoltageTimesNet_v2"]) > 1 and len(all_results["DLinear"]) > 1:
            t_stat, p_value = stats.ttest_rel(
                all_results["VoltageTimesNet_v2"],
                all_results["DLinear"]
            )
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                all_results["VoltageTimesNet_v2"],
                all_results["DLinear"]
            )
            significance_results["VoltageTimesNet_v2_vs_DLinear"] = {
                "t_statistic": t_stat,
                "t_p_value": p_value,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "significant_005": p_value < 0.05
            }
            self.log(f"\nVoltageTimesNet_v2 vs DLinear:")
            self.log(f"  t-test: t={t_stat:.4f}, p={p_value:.4f}")
            self.log(f"  Wilcoxon: stat={wilcoxon_stat:.4f}, p={wilcoxon_p:.4f}")
        
        # Summary statistics
        summary = {}
        for model in models:
            if all_results[model]:
                summary[model] = {
                    "mean_f1": np.mean(all_results[model]),
                    "std_f1": np.std(all_results[model]),
                    "min_f1": np.min(all_results[model]),
                    "max_f1": np.max(all_results[model]),
                    "all_f1": all_results[model]
                }
                self.log(f"\n{model}: F1 = {summary[model]['mean_f1']:.4f} +/- {summary[model]['std_f1']:.4f}")
        
        results = {
            "significance_tests": significance_results,
            "summary": summary
        }
        
        self.results["statistical_significance"] = results
        self._save_results("statistical_significance.json", results)
        
        return results
    
    def experiment_3_hyperparameter_optimization(self, n_trials=20):
        """
        Experiment 3: Hyperparameter optimization using Optuna.
        """
        self.log("=" * 60)
        self.log(f"EXPERIMENT 3: Hyperparameter Optimization (n_trials={n_trials})")
        self.log("=" * 60)
        
        try:
            import optuna
        except ImportError:
            self.log("ERROR: Optuna not installed. Installing...")
            os.system("pip install optuna")
            import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            d_model = trial.suggest_categorical("d_model", [32, 64, 128])
            e_layers = trial.suggest_int("e_layers", 1, 3)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            
            # Create config
            args = create_args_from_config(self.config, "VoltageTimesNet_v2")
            args.d_model = d_model
            args.e_layers = e_layers
            args.learning_rate = learning_rate
            args.batch_size = batch_size
            args.train_epochs = 5  # Reduced for speed
            args.d_ff = d_model * 4
            
            # Run experiment
            setting = f"optuna_trial{trial.number}_{self.timestamp}"
            
            try:
                exp = Exp_Anomaly_Detection(args)
                exp.train(setting)
                metrics = exp.test(setting, test=0)
                return metrics["f1"]
            except Exception as e:
                self.log(f"Trial {trial.number} failed: {str(e)}")
                return 0.0
        
        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Log results
        self.log(f"\nBest trial:")
        self.log(f"  F1 Score: {study.best_trial.value:.4f}")
        self.log(f"  Params: {study.best_trial.params}")
        
        results = {
            "best_f1": study.best_trial.value,
            "best_params": study.best_trial.params,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params
                }
                for t in study.trials
            ]
        }
        
        self.results["hyperparameter_optimization"] = results
        self._save_results("hyperparameter_optimization.json", results)
        
        return results
    
    def experiment_4_save_best_model(self):
        """
        Experiment 4: Train and save the best model.
        """
        self.log("=" * 60)
        self.log("EXPERIMENT 4: Save Best Model")
        self.log("=" * 60)
        
        # Use best hyperparameters if available
        best_params = self.results.get("hyperparameter_optimization", {}).get("best_params", {})
        
        args = create_args_from_config(self.config, "VoltageTimesNet_v2")
        args.d_model = best_params.get("d_model", 64)
        args.e_layers = best_params.get("e_layers", 2)
        args.learning_rate = best_params.get("learning_rate", 0.0001)
        args.batch_size = best_params.get("batch_size", 128)
        args.d_ff = args.d_model * 4
        args.train_epochs = 20  # Full training
        
        setting = f"best_model_{self.timestamp}"
        
        # Train
        self.log("Training best model with optimized parameters...")
        exp = Exp_Anomaly_Detection(args)
        exp.train(setting)
        
        # Test
        metrics = exp.test(setting, test=0)
        self.log(f"Final metrics: Acc={metrics['accuracy']:.4f}, "
                f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}")
        
        # Save model to newest_models directory
        save_dir = "./newest_models"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "best_voltagetimesnet_v2.pth")
        torch.save(exp.model.state_dict(), model_path)
        self.log(f"Model saved to: {model_path}")
        
        # Save model config
        config_path = os.path.join(save_dir, "best_model_config.json")
        config_dict = {
            "model": "VoltageTimesNet_v2",
            "d_model": args.d_model,
            "e_layers": args.e_layers,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "metrics": metrics
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        self.log(f"Config saved to: {config_path}")
        
        return metrics
    
    def _save_results(self, filename, data):
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.log(f"Results saved to: {filepath}")
    
    def _create_comparison_table(self, results, name):
        """Create a comparison table from results."""
        rows = []
        for model, metrics in results.items():
            rows.append({
                "Model": model,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}"
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("F1", ascending=False)
        
        # Save as CSV
        csv_path = os.path.join(self.results_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        
        # Print table
        self.log(f"\nComparison Table ({name}):")
        self.log(df.to_string(index=False))
    
    def generate_report(self):
        """Generate final experiment report."""
        self.log("=" * 60)
        self.log("Generating Experiment Report")
        self.log("=" * 60)
        
        report = []
        report.append("# Supplementary Experiment Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nResults Directory: {self.results_dir}")
        
        # Experiment 1: Baseline Comparison
        if "baseline_comparison" in self.results:
            report.append("\n## 1. Classic Baseline Models Comparison")
            report.append("\n### Results")
            report.append("\n| Model | Accuracy | Precision | Recall | F1 |")
            report.append("|-------|----------|-----------|--------|-----|")
            
            sorted_results = sorted(
                self.results["baseline_comparison"].items(),
                key=lambda x: x[1]["f1"],
                reverse=True
            )
            for model, metrics in sorted_results:
                report.append(
                    f"| {model} | {metrics['accuracy']:.4f} | "
                    f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} |"
                )
        
        # Experiment 2: Statistical Significance
        if "statistical_significance" in self.results:
            report.append("\n## 2. Statistical Significance Tests")
            
            sig_results = self.results["statistical_significance"]
            
            if "significance_tests" in sig_results:
                report.append("\n### Paired t-test and Wilcoxon Results")
                for comparison, test_results in sig_results["significance_tests"].items():
                    report.append(f"\n**{comparison}**")
                    report.append(f"- t-statistic: {test_results['t_statistic']:.4f}")
                    report.append(f"- t-test p-value: {test_results['t_p_value']:.4f}")
                    report.append(f"- Wilcoxon p-value: {test_results['wilcoxon_p_value']:.4f}")
                    report.append(f"- Significant (p<0.05): {test_results['significant_005']}")
            
            if "summary" in sig_results:
                report.append("\n### Summary Statistics")
                report.append("\n| Model | Mean F1 | Std F1 |")
                report.append("|-------|---------|--------|")
                for model, stats in sig_results["summary"].items():
                    report.append(
                        f"| {model} | {stats['mean_f1']:.4f} | {stats['std_f1']:.4f} |"
                    )
        
        # Experiment 3: Hyperparameter Optimization
        if "hyperparameter_optimization" in self.results:
            report.append("\n## 3. Hyperparameter Optimization")
            hp_results = self.results["hyperparameter_optimization"]
            report.append(f"\n**Best F1 Score:** {hp_results['best_f1']:.4f}")
            report.append("\n**Best Parameters:**")
            for param, value in hp_results["best_params"].items():
                report.append(f"- {param}: {value}")
        
        report.append("\n## 4. Best Model")
        report.append("\nThe best model has been saved to `newest_models/best_voltagetimesnet_v2.pth`")
        
        # Write report
        report_path = os.path.join(self.results_dir, "experiment_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        # Also save to docs directory
        docs_path = "./docs/supplementary_experiment_report.md"
        os.makedirs("./docs", exist_ok=True)
        with open(docs_path, "w") as f:
            f.write("\n".join(report))
        
        self.log(f"Report saved to: {report_path}")
        self.log(f"Report also saved to: {docs_path}")
        
        return report_path
    
    def run_all(self, skip_hp_opt=False, n_significance_runs=5, n_hp_trials=20):
        """Run all experiments."""
        self.log("=" * 60)
        self.log("STARTING SUPPLEMENTARY EXPERIMENTS")
        self.log("=" * 60)
        
        start_time = time.time()
        
        # Experiment 1: Baseline comparison
        self.experiment_1_baseline_comparison()
        
        # Experiment 2: Statistical significance
        self.experiment_2_statistical_significance(n_runs=n_significance_runs)
        
        # Experiment 3: Hyperparameter optimization (optional)
        if not skip_hp_opt:
            self.experiment_3_hyperparameter_optimization(n_trials=n_hp_trials)
        
        # Experiment 4: Save best model
        self.experiment_4_save_best_model()
        
        # Generate report
        self.generate_report()
        
        total_time = time.time() - start_time
        self.log(f"\nTotal time: {total_time/60:.2f} minutes")
        self.log("=" * 60)
        self.log("ALL EXPERIMENTS COMPLETED")
        self.log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Supplementary Experiments")
    parser.add_argument(
        "--skip-hp-opt",
        action="store_true",
        help="Skip hyperparameter optimization"
    )
    parser.add_argument(
        "--n-significance-runs",
        type=int,
        default=5,
        help="Number of runs for significance tests"
    )
    parser.add_argument(
        "--n-hp-trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter optimization"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer runs and trials"
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only specified experiment (1-4)"
    )
    
    args = parser.parse_args()
    
    experiments = SupplementaryExperiments()
    
    if args.quick:
        args.n_significance_runs = 3
        args.n_hp_trials = 10
    
    if args.exp:
        if args.exp == 1:
            experiments.experiment_1_baseline_comparison()
        elif args.exp == 2:
            experiments.experiment_2_statistical_significance(n_runs=args.n_significance_runs)
        elif args.exp == 3:
            experiments.experiment_3_hyperparameter_optimization(n_trials=args.n_hp_trials)
        elif args.exp == 4:
            experiments.experiment_4_save_best_model()
        experiments.generate_report()
    else:
        experiments.run_all(
            skip_hp_opt=args.skip_hp_opt,
            n_significance_runs=args.n_significance_runs,
            n_hp_trials=args.n_hp_trials
        )


if __name__ == "__main__":
    main()
