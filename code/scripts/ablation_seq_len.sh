#!/bin/bash

# =============================================================================
# 序列长度消融实验 - 输入窗口长度参数敏感性分析
# Sequence Length Ablation Study
# =============================================================================

# 实验说明:
# 测试不同序列长度对模型性能的影响
# 基础实验：TimesNet 在 RuralVoltage 数据集上
# 扩展实验：VoltageTimesNet 在 RuralVoltage 数据集上（验证长序列下的潜力）

cd "$(dirname "$0")/.." || exit

export CUDA_VISIBLE_DEVICES=0

DATA=RuralVoltage
ROOT_PATH=./dataset/RuralVoltage/comprehensive
DATA_PATH=train.csv
ENC_IN=16
C_OUT=16

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=./results/seq_len_ablation_${TIMESTAMP}
mkdir -p $RESULT_DIR

echo "=========================================="
echo "Sequence Length Ablation Study"
echo "Result directory: $RESULT_DIR"
echo "=========================================="

# =============================================================================
# Part 1: TimesNet 序列长度消融实验
# =============================================================================
echo "=========================================="
echo "Part 1: TimesNet Sequence Length Ablation"
echo "=========================================="

for SEQ_LEN in 50 100 200 500; do
    echo "=========================================="
    echo "Running TimesNet with seq_len=$SEQ_LEN"
    echo "=========================================="

    python -u run.py \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id ablation_seqlen_TimesNet_${SEQ_LEN} \
        --model TimesNet \
        --data $DATA \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len 0 \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model 64 \
        --d_ff 64 \
        --e_layers 2 \
        --top_k 5 \
        --num_kernels 6 \
        --dropout 0.1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --anomaly_ratio 1.0 \
        2>&1 | tee $RESULT_DIR/TimesNet_seq${SEQ_LEN}_log.txt

    echo "Completed TimesNet seq_len=$SEQ_LEN"
    echo ""
done

# =============================================================================
# Part 2: VoltageTimesNet 序列长度消融实验（验证长序列下的潜力）
# =============================================================================
echo "=========================================="
echo "Part 2: VoltageTimesNet Sequence Length Ablation"
echo "Note: Testing longer sequences to verify preset period utilization"
echo "=========================================="

# VoltageTimesNet 预设周期为 [60, 300, 900, 3600]
# seq_len 需要 >= 2 * preset_period 才能有效利用预设周期
# seq_len=100: 只能利用 60（1分钟周期）
# seq_len=360: 可以利用 60 和 300（5分钟周期）
# seq_len=720: 可以利用 60、300（5分钟周期）
# seq_len=1800: 可以利用 60、300、900（15分钟周期）

for SEQ_LEN in 100 360 720; do
    echo "=========================================="
    echo "Running VoltageTimesNet with seq_len=$SEQ_LEN"
    echo "=========================================="

    python -u run.py \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id ablation_seqlen_VoltageTimesNet_${SEQ_LEN} \
        --model VoltageTimesNet \
        --data $DATA \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len 0 \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model 64 \
        --d_ff 64 \
        --e_layers 2 \
        --top_k 5 \
        --num_kernels 6 \
        --preset_weight 0.3 \
        --dropout 0.1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --anomaly_ratio 1.0 \
        2>&1 | tee $RESULT_DIR/VoltageTimesNet_seq${SEQ_LEN}_log.txt

    echo "Completed VoltageTimesNet seq_len=$SEQ_LEN"
    echo ""
done

echo "=========================================="
echo "Sequence Length Ablation Experiments Completed!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="

# 生成汇总报告
cat << 'EOF' >> $RESULT_DIR/summary.txt
Sequence Length Ablation Study Summary
======================================

Part 1: TimesNet Configurations:
  - seq_len=50:  Short window
  - seq_len=100: Default window [baseline]
  - seq_len=200: Medium window
  - seq_len=500: Long window

Part 2: VoltageTimesNet Configurations:
  - seq_len=100: Can utilize 60s preset period
  - seq_len=360: Can utilize 60s, 300s preset periods
  - seq_len=720: Can utilize 60s, 300s preset periods (full 5min coverage)

Preset Period Requirements:
  - 60s (1min):   seq_len >= 120
  - 300s (5min):  seq_len >= 600
  - 900s (15min): seq_len >= 1800
  - 3600s (1h):   seq_len >= 7200

Dataset: RuralVoltage (16 features)
Training epochs: 10
EOF

echo "Summary saved to $RESULT_DIR/summary.txt"
