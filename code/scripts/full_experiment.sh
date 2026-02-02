#!/bin/bash
# Full Experiment Script for Thesis
# Runs comprehensive model comparison on RuralVoltage dataset
# Date: 2026-02-02

set -e
cd /home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code

RESULT_DIR="results/full_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR
mkdir -p logs

echo "=============================================="
echo "Full Experiment Suite for Thesis"
echo "Results will be saved to: $RESULT_DIR"
echo "=============================================="

# Common parameters
EPOCHS=10
BATCH_SIZE=64
PATIENCE=3
LR=0.0001

# ============================================
# Part 1: RuralVoltage Dataset (16 features)
# ============================================
echo ""
echo "=============================================="
echo "Part 1: RuralVoltage Dataset Experiments"
echo "=============================================="

RURAL_ARGS="--data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 100 \
    --d_model 64 \
    --d_ff 128 \
    --e_layers 2 \
    --top_k 5 \
    --train_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE \
    --learning_rate $LR \
    --num_workers 0"

# Models to test on RuralVoltage
RURAL_MODELS=("TimesNet" "VoltageTimesNet" "VoltageTimesNet_v2" "TPATimesNet" "DLinear" "PatchTST")

for model in "${RURAL_MODELS[@]}"; do
    echo ""
    echo ">>> Training $model on RuralVoltage..."
    python -u run.py --is_training 1 --model $model $RURAL_ARGS --des full_rural 2>&1 | tee logs/full_${model}_rural.log

    # Extract results
    grep -E "(Accuracy|Precision|Recall|F1-score)" logs/full_${model}_rural.log | tail -1 >> $RESULT_DIR/rural_results.txt
    echo "$model: $(grep -E 'F1-score' logs/full_${model}_rural.log | tail -1)" >> $RESULT_DIR/rural_summary.txt
done

echo ""
echo "RuralVoltage experiments completed!"
echo "Results saved to: $RESULT_DIR/rural_results.txt"

# ============================================
# Part 2: Threshold Sensitivity Analysis
# ============================================
echo ""
echo "=============================================="
echo "Part 2: Threshold Sensitivity Analysis"
echo "=============================================="

echo "VoltageTimesNet_v2 Threshold Analysis on RuralVoltage:" > $RESULT_DIR/threshold_analysis.txt
for ratio in 1.0 1.5 2.0 2.5 3.0 4.0 5.0; do
    echo "Testing anomaly_ratio=$ratio..."
    result=$(python run.py --is_training 0 --model VoltageTimesNet_v2 $RURAL_ARGS --anomaly_ratio $ratio --des threshold_test 2>&1 | grep -E "(Threshold|Accuracy|Precision|Recall|F1)")
    echo "anomaly_ratio=$ratio: $result" >> $RESULT_DIR/threshold_analysis.txt
done

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results directory: $RESULT_DIR"
echo ""
echo "=== RuralVoltage Results ==="
cat $RESULT_DIR/rural_summary.txt
echo ""
echo "=== Threshold Analysis ==="
cat $RESULT_DIR/threshold_analysis.txt

# Save completion marker
echo "Experiment completed at $(date)" > $RESULT_DIR/COMPLETED
echo ""
echo "All experiments finished successfully!"
