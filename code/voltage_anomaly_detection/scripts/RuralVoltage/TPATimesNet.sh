#!/bin/bash

# =============================================================================
# 农网低电压异常检测实验 - TPATimesNet (三相注意力机制)
# Three-Phase Attention TimesNet for Rural Low-Voltage Anomaly Detection
# =============================================================================

# 实验说明:
# TPATimesNet 是针对农网三相电压数据设计的创新模型
# 核心创新: 引入三相注意力机制，显式建模Va/Vb/Vc之间的相位关系
# 适用场景: 三相不平衡检测、跨相位异常传播分析

cd "$(dirname "$0")/.." || exit

# 基础配置
export CUDA_VISIBLE_DEVICES=0

# 数据集配置
DATA=RuralVoltage
ROOT_PATH=./dataset/RuralVoltage
DATA_PATH=train.csv

# 模型配置
MODEL=TPATimesNet
SEQ_LEN=100
ENC_IN=16      # 16个特征: Va,Vb,Vc,Ia,Ib,Ic,P,Q,S,PF,THD_Va,THD_Vb,THD_Vc,Freq,V_unbal,I_unbal
C_OUT=16

# =============================================================================
# 实验1: 标准配置
# =============================================================================
echo "=========================================="
echo "Experiment 1: TPATimesNet Standard Config"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id TPATimesNet_${DATA}_standard \
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
# 实验2: 深层配置 (更多层数)
# =============================================================================
echo "=========================================="
echo "Experiment 2: TPATimesNet Deep Config"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id TPATimesNet_${DATA}_deep \
    --model $MODEL \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 4 \
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
# 实验3: 长序列配置
# =============================================================================
echo "=========================================="
echo "Experiment 3: TPATimesNet Long Sequence"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id TPATimesNet_${DATA}_long_seq \
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
echo "TPATimesNet experiments completed!"
echo "=========================================="
