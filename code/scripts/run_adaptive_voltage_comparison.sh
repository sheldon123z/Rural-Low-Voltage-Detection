#!/bin/bash
# AdaptiveVoltageTimesNet 对比实验
# 在RuralVoltage comprehensive数据集上对比各模型性能

# 设置实验参数
DATA_ROOT="./dataset/RuralVoltage/comprehensive"
SEQ_LEN=100
D_MODEL=64
D_FF=64
E_LAYERS=2
TOP_K=5
BATCH_SIZE=32
TRAIN_EPOCHS=10
ENC_IN=16
C_OUT=16
NUM_WORKERS=0

# 创建结果目录
RESULT_DIR="./results/adaptive_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

echo "=== AdaptiveVoltageTimesNet 对比实验 ===" | tee $RESULT_DIR/experiment_log.txt
echo "开始时间: $(date)" | tee -a $RESULT_DIR/experiment_log.txt
echo "" | tee -a $RESULT_DIR/experiment_log.txt

# 模型列表
MODELS=("TimesNet" "AdaptiveVoltageTimesNet" "VoltageTimesNet" "TPATimesNet")

for MODEL in "${MODELS[@]}"; do
    echo "=== 训练模型: $MODEL ===" | tee -a $RESULT_DIR/experiment_log.txt
    echo "开始: $(date)" | tee -a $RESULT_DIR/experiment_log.txt

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_ROOT \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --train_epochs $TRAIN_EPOCHS \
        --num_workers $NUM_WORKERS \
        2>&1 | tee $RESULT_DIR/${MODEL}_log.txt

    echo "完成: $(date)" | tee -a $RESULT_DIR/experiment_log.txt
    echo "" | tee -a $RESULT_DIR/experiment_log.txt
done

echo "=== 所有实验完成 ===" | tee -a $RESULT_DIR/experiment_log.txt
echo "结束时间: $(date)" | tee -a $RESULT_DIR/experiment_log.txt
echo "结果保存在: $RESULT_DIR" | tee -a $RESULT_DIR/experiment_log.txt
