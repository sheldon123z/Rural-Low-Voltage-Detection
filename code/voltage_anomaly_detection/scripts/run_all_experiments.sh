#!/bin/bash

# =============================================================================
# 农网低电压异常检测实验 - 完整实验运行脚本
# Complete Experiment Runner for Rural Low-Voltage Anomaly Detection
# =============================================================================

# 用法:
#   bash run_all_experiments.sh          # 运行全部实验
#   bash run_all_experiments.sh baseline # 仅运行基线
#   bash run_all_experiments.sh innovative # 仅运行创新模型
#   bash run_all_experiments.sh ablation # 运行消融实验

cd "$(dirname "$0")" || exit

EXPERIMENT_TYPE=${1:-all}

echo "=============================================="
echo "农网低电压异常检测实验"
echo "Rural Low-Voltage Anomaly Detection Experiments"
echo "=============================================="
echo "实验类型: $EXPERIMENT_TYPE"
echo "开始时间: $(date)"
echo "=============================================="

# 创建结果目录
mkdir -p ../test_results/experiment_summary

run_baseline() {
    echo ""
    echo ">>> 运行基线模型实验..."
    bash RuralVoltage/run_baselines.sh 2>&1 | tee ../test_results/experiment_summary/baselines.log
}

run_innovative() {
    echo ""
    echo ">>> 运行创新模型实验..."
    
    echo "--- TPATimesNet (三相注意力) ---"
    bash RuralVoltage/TPATimesNet.sh 2>&1 | tee ../test_results/experiment_summary/TPATimesNet.log
    
    echo "--- MTSTimesNet (多尺度时序) ---"
    bash RuralVoltage/MTSTimesNet.sh 2>&1 | tee ../test_results/experiment_summary/MTSTimesNet.log
    
    echo "--- HybridTimesNet (混合周期) ---"
    bash RuralVoltage/HybridTimesNet.sh 2>&1 | tee ../test_results/experiment_summary/HybridTimesNet.log
}

run_ablation() {
    echo ""
    echo ">>> 运行消融实验..."
    bash RuralVoltage/run_ablation.sh 2>&1 | tee ../test_results/experiment_summary/ablation.log
}

case $EXPERIMENT_TYPE in
    baseline)
        run_baseline
        ;;
    innovative)
        run_innovative
        ;;
    ablation)
        run_ablation
        ;;
    all)
        run_baseline
        run_innovative
        ;;
    *)
        echo "未知的实验类型: $EXPERIMENT_TYPE"
        echo "可选: all, baseline, innovative, ablation"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "实验完成!"
echo "结束时间: $(date)"
echo "结果保存在: test_results/experiment_summary/"
echo "=============================================="
