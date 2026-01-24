#!/bin/bash

# Rural Voltage Anomaly Detection with TimesNet
# Custom Rural Power Grid Dataset
# 
# Expected features (16 dimensions):
#   - Va, Vb, Vc: Three-phase voltage (V)
#   - Ia, Ib, Ic: Three-phase current (A)
#   - P, Q, S, PF: Power metrics
#   - THD_Va, THD_Vb, THD_Vc: Harmonic distortion (%)
#   - Freq: Frequency (Hz)
#   - V_unbalance, I_unbalance: Unbalance ratio (%)

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --model_id RuralVoltage \
  --model TimesNet \
  --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 16 \
  --c_out 16 \
  --top_k 3 \
  --batch_size 64 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1 \
  --des Exp
