#!/bin/bash
# PSM 数据集公共配置
# 用于多模型故障检测对比实验

# 数据集配置
export DATASET=PSM
export ROOT_PATH=./dataset/PSM/
export ENC_IN=25
export C_OUT=25
export FEATURES=M

# 模型架构参数
export SEQ_LEN=100
export PRED_LEN=0
export D_MODEL=64
export D_FF=64
export E_LAYERS=2
export N_HEADS=8
export DROPOUT=0.1

# TimesNet 系列参数
export TOP_K=3
export NUM_KERNELS=6

# 训练参数
export BATCH_SIZE=128
export LEARNING_RATE=0.0001
export TRAIN_EPOCHS=10
export PATIENCE=3
export NUM_WORKERS=4

# 异常检测参数
export ANOMALY_RATIO=1.0
export SEED=2021

# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 结果目录（时间戳）
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RESULT_DIR="./results/PSM_comparison_${TIMESTAMP}"

# 公共训练参数
export COMMON_ARGS="
  --is_training 1
  --data ${DATASET}
  --root_path ${ROOT_PATH}
  --features ${FEATURES}
  --seq_len ${SEQ_LEN}
  --pred_len ${PRED_LEN}
  --enc_in ${ENC_IN}
  --c_out ${C_OUT}
  --d_model ${D_MODEL}
  --d_ff ${D_FF}
  --e_layers ${E_LAYERS}
  --n_heads ${N_HEADS}
  --dropout ${DROPOUT}
  --batch_size ${BATCH_SIZE}
  --learning_rate ${LEARNING_RATE}
  --train_epochs ${TRAIN_EPOCHS}
  --patience ${PATIENCE}
  --num_workers ${NUM_WORKERS}
  --anomaly_ratio ${ANOMALY_RATIO}
  --des PSM_Comparison
"
