#!/bin/bash

# MSL Dataset Anomaly Detection with TimesNet
# Mars Science Laboratory Dataset - 55 features

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --model_id MSL \
  --model TimesNet \
  --data MSL \
  --root_path ./dataset/MSL/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --batch_size 128 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1 \
  --des Exp
