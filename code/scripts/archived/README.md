# Archived Scripts / 已归档脚本

本目录包含已弃用或被更新版本替代的脚本，保留供参考。

**注意**: 这些脚本可能无法直接运行，仅供代码参考。

## 归档说明

| 脚本 | 归档原因 | 替代方案 |
|------|----------|----------|
| `analyze_experiment_results.py` | v1 版本，已有更完善的分析工具 | `analyze_comparison_results.py` |
| `analyze_experiment_results_v2.py` | v2 版本，功能已整合到主分析脚本 | `analyze_comparison_results.py` |
| `analyze_psm_results.py` | PSM 专用分析，功能已整合 | `analyze_comparison_results.py` |
| `analyze_targeted_results.py` | 针对性数据集分析，功能已整合 | `analyze_comparison_results.py` |
| `generate_full_analysis.py` | 完整分析生成，功能已整合 | `analyze_comparison_results.py` |
| `auto_analyze.sh` | 自动分析脚本，功能已整合到主实验脚本 | `full_experiment.sh` |
| `compare_timesnet_psm.sh` | PSM 对比实验，已整合到完整实验 | `full_experiment.sh` |
| `monitor_training.sh` | 训练监控，不再需要 | 直接查看日志文件 |
| `run_all_experiments.sh` | 早期全量实验脚本 | `full_experiment.sh` |
| `train_msl.sh` | MSL 数据集训练 | `MSL/` 目录下脚本 |
| `train_optimized.sh` | 优化版训练脚本，已整合 | `full_experiment.sh` |
| `train_parallel.sh` | 并行训练脚本，已整合 | `full_experiment.sh` |
| `train_psm.sh` | PSM 数据集训练 | `PSM/` 目录下脚本 |
| `quick_comparison.sh` | 快速对比实验 | `full_experiment.sh` |
| `run_adaptive_voltage_comparison.sh` | 自适应电压对比 | `full_experiment.sh` |
| `quick_test_fixes.py` | 快速修复测试 | 已完成，不再需要 |

## 归档时间

2026-02-02

## 如需恢复

如需恢复某个脚本，可将其移回上级目录：
```bash
mv archived/script_name.sh ../
```

但建议优先使用当前维护的脚本。
