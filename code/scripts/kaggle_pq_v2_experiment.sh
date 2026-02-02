#!/bin/bash
# KagglePQ V2 数据集对比实验
# 新数据集: 15% 异常比例 (更合理的异常检测场景)

# 配置
DATA_PATH="./dataset/Kaggle_PowerQuality_2/"
EPOCHS=10
BATCH_SIZE=32
SEQ_LEN=64
D_MODEL=64
D_FF=128
ENC_IN=128
C_OUT=128
TOP_K=5
E_LAYERS=2

# 创建结果目录
RESULT_DIR="./results/kaggle_pq_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

echo "=================================================="
echo "KagglePQ V2 数据集对比实验"
echo "异常比例: 15%"
echo "结果目录: $RESULT_DIR"
echo "=================================================="

# 模型列表
MODELS=("TimesNet" "VoltageTimesNet" "VoltageTimesNet_v2" "DLinear" "PatchTST")

# 运行实验
for model in "${MODELS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Training: $model"
    echo "=================================================="

    python run.py \
        --is_training 1 \
        --model $model \
        --data KagglePQ \
        --root_path $DATA_PATH \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --seq_len $SEQ_LEN \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --train_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_workers 0 \
        2>&1 | tee "$RESULT_DIR/${model}.log"

    # 提取结果
    echo "$model: $(grep -E 'Accuracy|F1-score' $RESULT_DIR/${model}.log | tail -1)" >> "$RESULT_DIR/summary.txt"
done

echo ""
echo "=================================================="
echo "实验完成!"
echo "=================================================="
echo "结果摘要:"
cat "$RESULT_DIR/summary.txt"
