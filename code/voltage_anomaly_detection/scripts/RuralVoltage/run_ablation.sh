#!/bin/bash

# =============================================================================
# 消融实验 - 验证各创新组件的贡献
# Ablation Studies for Innovative Components
# =============================================================================

# 实验说明:
# 通过逐步移除或简化创新组件，验证每个组件的贡献
# 1. TPATimesNet 消融: 无三相注意力 vs 完整模型
# 2. MTSTimesNet 消融: 单尺度 vs 多尺度
# 3. HybridTimesNet 消融: 仅FFT vs 仅预设 vs 混合

cd "$(dirname "$0")/.." || exit

export CUDA_VISIBLE_DEVICES=0

DATA=RuralVoltage
ROOT_PATH=./dataset/RuralVoltage
DATA_PATH=train.csv
SEQ_LEN=100
ENC_IN=16
C_OUT=16

# =============================================================================
# 消融实验组1: 三相注意力的影响
# =============================================================================
echo "=========================================="
echo "Ablation 1: Three-Phase Attention Impact"
echo "=========================================="

# 完整TPATimesNet (含三相注意力)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_TPA_full \
    --model TPATimesNet \
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

# 对照: 原始TimesNet (无三相注意力)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_TPA_baseline \
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
    --anomaly_ratio 1.0

# =============================================================================
# 消融实验组2: 多尺度的影响
# =============================================================================
echo "=========================================="
echo "Ablation 2: Multi-scale Impact"
echo "=========================================="

# 完整MTSTimesNet (多尺度)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_MTS_full \
    --model MTSTimesNet \
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

# 对照: 原始TimesNet (单尺度)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_MTS_single_scale \
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
    --anomaly_ratio 1.0

# =============================================================================
# 消融实验组3: 混合周期发现的影响
# =============================================================================
echo "=========================================="
echo "Ablation 3: Hybrid Period Discovery Impact"
echo "=========================================="

# 完整HybridTimesNet (FFT + 预设)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_Hybrid_full \
    --model HybridTimesNet \
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

# 对照: 原始TimesNet (仅FFT)
python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id ablation_Hybrid_fft_only \
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
    --anomaly_ratio 1.0

# =============================================================================
# 消融实验组4: 超参数敏感性
# =============================================================================
echo "=========================================="
echo "Ablation 4: Hyperparameter Sensitivity"
echo "=========================================="

# top_k 敏感性
for TOP_K in 3 5 8; do
    python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id ablation_topk_${TOP_K} \
        --model TPATimesNet \
        --data $DATA \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len 0 \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model 64 \
        --d_ff 64 \
        --e_layers 2 \
        --top_k $TOP_K \
        --num_kernels 6 \
        --n_heads 8 \
        --dropout 0.1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --anomaly_ratio 1.0
done

# d_model 敏感性
for D_MODEL in 32 64 128; do
    python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id ablation_dmodel_${D_MODEL} \
        --model TPATimesNet \
        --data $DATA \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len 0 \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_MODEL \
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
done

echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
