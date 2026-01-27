#!/bin/bash

# =============================================================================
# α 参数消融实验 - VoltageTimesNet 混合周期比例参数敏感性分析
# Alpha Parameter Ablation Study for VoltageTimesNet
# =============================================================================

# 实验说明:
# 测试不同 α 值对模型性能的影响
# α = 1 - preset_weight (FFT发现周期的权重)
# preset_weight = 0.1, 0.2, 0.3, 0.4, 0.5 对应 α = 0.9, 0.8, 0.7, 0.6, 0.5

cd "$(dirname "$0")/.." || exit

export CUDA_VISIBLE_DEVICES=0

DATA=PSM
ROOT_PATH=./dataset/PSM
DATA_PATH=data.csv
SEQ_LEN=100
ENC_IN=25
C_OUT=25

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=./results/alpha_ablation_${TIMESTAMP}
mkdir -p $RESULT_DIR

echo "=========================================="
echo "Alpha Parameter Ablation Study"
echo "Result directory: $RESULT_DIR"
echo "=========================================="

# α 参数消融实验
# preset_weight = 1 - α
for PRESET_WEIGHT in 0.1 0.2 0.3 0.4 0.5; do
    ALPHA=$(echo "1 - $PRESET_WEIGHT" | bc)
    echo "=========================================="
    echo "Running VoltageTimesNet with alpha=$ALPHA (preset_weight=$PRESET_WEIGHT)"
    echo "=========================================="

    python -u run.py \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id ablation_alpha_${ALPHA} \
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
        --preset_weight $PRESET_WEIGHT \
        --dropout 0.1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --anomaly_ratio 1.0 \
        2>&1 | tee $RESULT_DIR/alpha_${ALPHA}_log.txt

    echo "Completed alpha=$ALPHA"
    echo ""
done

# 对照实验：原始TimesNet (仅FFT，无预设周期)
echo "=========================================="
echo "Running baseline TimesNet (FFT only)"
echo "=========================================="

python -u run.py \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_baseline_timesnet \
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
    2>&1 | tee $RESULT_DIR/baseline_timesnet_log.txt

echo "=========================================="
echo "Alpha ablation experiments completed!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="

# 生成汇总报告
echo "" >> $RESULT_DIR/summary.txt
echo "Alpha Parameter Ablation Study Summary" >> $RESULT_DIR/summary.txt
echo "======================================" >> $RESULT_DIR/summary.txt
echo "Date: $TIMESTAMP" >> $RESULT_DIR/summary.txt
echo "" >> $RESULT_DIR/summary.txt
echo "Configurations tested:" >> $RESULT_DIR/summary.txt
echo "  - alpha=0.5 (preset_weight=0.5)" >> $RESULT_DIR/summary.txt
echo "  - alpha=0.6 (preset_weight=0.4)" >> $RESULT_DIR/summary.txt
echo "  - alpha=0.7 (preset_weight=0.3) [default]" >> $RESULT_DIR/summary.txt
echo "  - alpha=0.8 (preset_weight=0.2)" >> $RESULT_DIR/summary.txt
echo "  - alpha=0.9 (preset_weight=0.1)" >> $RESULT_DIR/summary.txt
echo "  - baseline: TimesNet (FFT only)" >> $RESULT_DIR/summary.txt
echo "" >> $RESULT_DIR/summary.txt
echo "Dataset: PSM" >> $RESULT_DIR/summary.txt
echo "Sequence length: 100" >> $RESULT_DIR/summary.txt
echo "Training epochs: 10" >> $RESULT_DIR/summary.txt
