#!/bin/bash

# =============================================================================
# 农网低电压异常检测实验 - MTSTimesNet (多尺度时序模型)
# Multi-scale Temporal TimesNet for Rural Low-Voltage Anomaly Detection
# =============================================================================

# 实验说明:
# MTSTimesNet 采用并行多尺度时序分支，同时捕获不同时间尺度的模式
# 核心创新: 短期(波动)、中期(趋势)、长期(周期)三个并行分支 + 自适应融合门
# 适用场景: 复杂多尺度异常模式识别

cd "$(dirname "$0")/.." || exit

# 基础配置
export CUDA_VISIBLE_DEVICES=0

# 数据集配置
DATA=RuralVoltage
ROOT_PATH=./dataset/RuralVoltage
DATA_PATH=train.csv

# 模型配置
MODEL=MTSTimesNet
SEQ_LEN=100
ENC_IN=16
C_OUT=16

# =============================================================================
# 实验1: 标准配置
# =============================================================================
echo "=========================================="
echo "Experiment 1: MTSTimesNet Standard Config"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id MTSTimesNet_${DATA}_standard \
    --model $MODEL \
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
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

# =============================================================================
# 实验2: 更多层数配置
# =============================================================================
echo "=========================================="
echo "Experiment 2: MTSTimesNet Deep Config"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id MTSTimesNet_${DATA}_deep \
    --model $MODEL \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 3 \
    --top_k 5 \
    --num_kernels 6 \
    --n_heads 8 \
    --dropout 0.2 \
    --batch_size 32 \
    --learning_rate 0.00005 \
    --train_epochs 15 \
    --patience 5 \
    --anomaly_ratio 1.0

# =============================================================================
# 实验3: 长序列配置 (捕获更长期模式)
# =============================================================================
echo "=========================================="
echo "Experiment 3: MTSTimesNet Long Sequence"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id MTSTimesNet_${DATA}_long_seq \
    --model $MODEL \
    --data $DATA \
    --features M \
    --seq_len 200 \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --top_k 8 \
    --num_kernels 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

echo "=========================================="
echo "MTSTimesNet experiments completed!"
echo "=========================================="
