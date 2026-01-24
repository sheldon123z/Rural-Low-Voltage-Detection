#!/bin/bash

# PSM Dataset Anomaly Detection with TimesNet
# Server Machine Dataset - 25 features

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --model_id PSM \
  --model TimesNet \
  --data PSM \
  --root_path ./dataset/PSM/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --batch_size 128 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1 \
  --des Exp
