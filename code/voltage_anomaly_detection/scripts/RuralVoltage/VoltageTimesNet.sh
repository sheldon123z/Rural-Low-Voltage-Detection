#!/bin/bash

# Rural Voltage Anomaly Detection with VoltageTimesNet
# Dataset: RuralVoltage (17 features)
# Task: Anomaly Detection
# Model: VoltageTimesNet (TimesNet variant optimized for voltage signals)

export CUDA_VISIBLE_DEVICES=0

# Generate sample data if not exists
if [ ! -f "./dataset/RuralVoltage/train.csv" ]; then
    echo "Generating sample data..."
    cd ./dataset/RuralVoltage
    python generate_sample_data.py --train_samples 10000 --test_samples 2000 --anomaly_ratio 0.1
    cd ../..
fi

model_name=VoltageTimesNet
seq_len=100

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/RuralVoltage/ \
  --model_id RuralVoltage_VoltageTimesNet_${seq_len} \
  --model $model_name \
  --data RuralVoltage \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 17 \
  --c_out 17 \
  --top_k 5 \
  --num_kernels 6 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1.0 \
  --des 'VoltageTimesNet_Exp'
