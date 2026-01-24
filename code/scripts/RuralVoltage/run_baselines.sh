#!/bin/bash

# =============================================================================
# 农网低电压异常检测实验 - 基线模型对比实验
# Baseline Model Comparison for Rural Low-Voltage Anomaly Detection
# =============================================================================

# 实验说明:
# 运行多个基线模型作为对比，评估创新模型的性能提升
# 基线模型: TimesNet, DLinear, PatchTST, Transformer, Autoformer

cd "$(dirname "$0")/.." || exit

# 基础配置
export CUDA_VISIBLE_DEVICES=0

# 数据集配置
DATA=RuralVoltage
ROOT_PATH=./dataset/RuralVoltage
DATA_PATH=train.csv
SEQ_LEN=100
ENC_IN=16
C_OUT=16

# =============================================================================
# 基线1: TimesNet (原始模型)
# =============================================================================
echo "=========================================="
echo "Baseline 1: TimesNet"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id TimesNet_${DATA}_baseline \
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
# 基线2: DLinear
# =============================================================================
echo "=========================================="
echo "Baseline 2: DLinear"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id DLinear_${DATA}_baseline \
    --model DLinear \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

# =============================================================================
# 基线3: PatchTST
# =============================================================================
echo "=========================================="
echo "Baseline 3: PatchTST"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id PatchTST_${DATA}_baseline \
    --model PatchTST \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

# =============================================================================
# 基线4: Transformer
# =============================================================================
echo "=========================================="
echo "Baseline 4: Transformer"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id Transformer_${DATA}_baseline \
    --model Transformer \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

# =============================================================================
# 基线5: Autoformer
# =============================================================================
echo "=========================================="
echo "Baseline 5: Autoformer"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id Autoformer_${DATA}_baseline \
    --model Autoformer \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len 0 \
    --enc_in $ENC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

# =============================================================================
# 基线6: VoltageTimesNet (已有的电压优化版本)
# =============================================================================
echo "=========================================="
echo "Baseline 6: VoltageTimesNet"
echo "=========================================="

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id VoltageTimesNet_${DATA}_baseline \
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
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --anomaly_ratio 1.0

echo "=========================================="
echo "All baseline experiments completed!"
echo "=========================================="
