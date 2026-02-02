#!/bin/bash
# Quick comparison script for model validation
# Tests: TimesNet vs VoltageTimesNet vs VoltageTimesNet_v2
# Target: Recall improvement from 62% to 70%+

set -e

cd /home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code

# Quick training configuration (~3-5 min per model)
COMMON_ARGS="--is_training 1 \
    --data RuralVoltage \
    --root_path ./dataset/RuralVoltage/realistic_v2/ \
    --enc_in 16 --c_out 16 \
    --seq_len 100 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 2 \
    --top_k 3 \
    --train_epochs 3 \
    --batch_size 64 \
    --patience 2 \
    --learning_rate 0.0001 \
    --num_workers 0 \
    --des quick_compare"

echo "=========================================="
echo "Quick Model Comparison - Recall Focus"
echo "=========================================="
echo ""

# 1. TimesNet (baseline)
echo "[1/3] Testing TimesNet (baseline)..."
python -u run.py --model TimesNet $COMMON_ARGS 2>&1 | tee logs/quick_TimesNet.log
echo ""

# 2. VoltageTimesNet (original)
echo "[2/3] Testing VoltageTimesNet..."
python -u run.py --model VoltageTimesNet $COMMON_ARGS 2>&1 | tee logs/quick_VoltageTimesNet.log
echo ""

# 3. VoltageTimesNet_v2 (improved)
echo "[3/3] Testing VoltageTimesNet_v2 (improved)..."
python -u run.py --model VoltageTimesNet_v2 $COMMON_ARGS 2>&1 | tee logs/quick_VoltageTimesNet_v2.log
echo ""

echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Extracting F-score and metrics from logs..."
echo ""

for model in TimesNet VoltageTimesNet VoltageTimesNet_v2; do
    echo "=== $model ==="
    grep -E "(Precision|Recall|F-score|Accuracy)" logs/quick_${model}.log | tail -4
    echo ""
done

echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
