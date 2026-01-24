#!/bin/bash
# TimesNet 变体模型在 PSM 数据集上的对比实验
# 对比模型: TimesNet, VoltageTimesNet, TPATimesNet, MTSTimesNet, HybridTimesNet

set -e

# 配置
export CUDA_VISIBLE_DEVICES=0
RESULT_DIR="./results/psm_comparison"
mkdir -p $RESULT_DIR

# 公共参数
COMMON_ARGS="
  --is_training 1
  --data PSM
  --root_path ./dataset/PSM/
  --features M
  --seq_len 100
  --pred_len 0
  --d_model 64
  --d_ff 64
  --e_layers 2
  --enc_in 25
  --c_out 25
  --top_k 3
  --batch_size 128
  --train_epochs 10
  --patience 3
  --learning_rate 0.0001
  --anomaly_ratio 1
  --des PSM_Comparison
"

# 模型列表
MODELS=("TimesNet" "VoltageTimesNet" "TPATimesNet" "MTSTimesNet" "HybridTimesNet")

echo "=========================================="
echo "TimesNet 变体模型 PSM 数据集对比实验"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 训练每个模型
for model in "${MODELS[@]}"; do
    echo "------------------------------------------"
    echo "正在训练模型: $model"
    echo "开始时间: $(date)"
    echo "------------------------------------------"

    python -u run.py \
        --model_id PSM_${model} \
        --model $model \
        $COMMON_ARGS \
        2>&1 | tee $RESULT_DIR/${model}_log.txt

    echo "模型 $model 训练完成"
    echo ""
done

echo "=========================================="
echo "所有模型训练完成"
echo "结束时间: $(date)"
echo "=========================================="
