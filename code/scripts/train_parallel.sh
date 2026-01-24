#!/bin/bash
# 并行训练脚本 - 充分利用 A800 80GB 显存
#
# 策略:
# 1. 同时运行多个训练任务 (每个约 2-3GB 显存)
# 2. 使用较少的 DataLoader workers (每任务 2 个)
# 3. 自动跳过已完成的实验

echo "=============================================="
echo "并行训练 - 针对性数据集实验"
echo "=============================================="

# 并行配置
MAX_PARALLEL=8          # 最大并行任务数
NUM_WORKERS=2           # 每个任务的 DataLoader workers

# 模型参数
D_MODEL=64
D_FF=128
E_LAYERS=2
TOP_K=5
EPOCHS=10
LR=0.0001
ENC_IN=16
C_OUT=16

RESULT_DIR="./results/targeted_quick"
mkdir -p $RESULT_DIR

# 函数: 检查是否已完成
is_completed() {
    local LOG_FILE=$1
    if [ -f "$LOG_FILE" ] && grep -q "F1-score" "$LOG_FILE" 2>/dev/null; then
        return 0
    fi
    return 1
}

# 函数: 获取当前运行的训练进程数
get_running_count() {
    ps aux | grep -E "python.*run\.py" | grep -v grep | wc -l
}

# 函数: 等待有空闲槽位
wait_for_slot() {
    while true; do
        local running=$(get_running_count)
        # 每个任务有 1 主进程 + NUM_WORKERS 个 worker
        local task_count=$((running / (NUM_WORKERS + 1)))
        if [ $task_count -lt $MAX_PARALLEL ]; then
            return
        fi
        sleep 3
    done
}

# 函数: 启动训练任务
start_training() {
    local MODEL=$1
    local DATASET=$2
    local RATIO=$3
    local BS=${4:-256}
    local SL=${5:-100}

    local TASK_NAME="${MODEL}_${DATASET}"
    local LOG_FILE="$RESULT_DIR/${TASK_NAME}.log"

    # 检查是否已完成
    if is_completed "$LOG_FILE"; then
        echo "  ⏭ 跳过: $TASK_NAME (已完成)"
        return 0
    fi

    # 等待有空闲槽位
    wait_for_slot

    echo "  🚀 启动: $TASK_NAME"

    # 后台启动训练
    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path "./dataset/RuralVoltage/$DATASET" \
        --seq_len $SL \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BS \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --num_workers $NUM_WORKERS \
        --anomaly_ratio $RATIO \
        > "$LOG_FILE" 2>&1 &

    # 短暂等待确保进程启动
    sleep 1
    return 0
}

# ============================================
# 定义所有实验任务
# ============================================
echo ""
echo "配置: 最大并行=$MAX_PARALLEL, workers=$NUM_WORKERS, epochs=$EPOCHS"
echo ""

# 所有实验列表: MODEL DATASET RATIO BATCH_SIZE SEQ_LEN
EXPERIMENTS=(
    # 周期性负荷数据集
    "TimesNet periodic_load 15 256 100"
    "VoltageTimesNet periodic_load 15 256 100"
    "TPATimesNet periodic_load 15 256 100"
    "MTSTimesNet periodic_load 15 256 100"
    # 三相不平衡数据集
    "TimesNet three_phase 23 256 100"
    "VoltageTimesNet three_phase 23 256 100"
    "TPATimesNet three_phase 23 256 100"
    "MTSTimesNet three_phase 23 256 100"
    # 多尺度复合数据集
    "TimesNet multi_scale 47 128 200"
    "VoltageTimesNet multi_scale 47 128 200"
    "TPATimesNet multi_scale 47 128 200"
    "MTSTimesNet multi_scale 47 128 200"
    # 综合评估数据集
    "TimesNet comprehensive 49 256 100"
    "VoltageTimesNet comprehensive 49 256 100"
    "TPATimesNet comprehensive 49 256 100"
    "MTSTimesNet comprehensive 49 256 100"
)

# ============================================
# 启动所有实验
# ============================================
echo "启动 ${#EXPERIMENTS[@]} 个实验任务..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r MODEL DATASET RATIO BS SL <<< "$exp"
    start_training "$MODEL" "$DATASET" "$RATIO" "$BS" "$SL"
done

# ============================================
# 等待所有任务完成
# ============================================
echo ""
echo "所有任务已提交，等待完成..."
echo "监控命令: watch -n 5 'ps aux | grep python.*run | grep -v grep | wc -l'"
echo ""

while true; do
    running=$(get_running_count)
    if [ $running -eq 0 ]; then
        break
    fi
    completed=$(grep -l "F1-score" $RESULT_DIR/*.log 2>/dev/null | wc -l)
    echo "  运行中: $((running / 3)) 任务, 已完成: $completed/16"
    sleep 30
done

# ============================================
# 生成分析报告
# ============================================
echo ""
echo "=============================================="
echo "所有训练完成! 生成分析报告..."
echo "=============================================="

python scripts/analyze_targeted_results.py --result_dir $RESULT_DIR --no_timestamp

# 显示结果汇总
echo ""
echo "=== F1 分数汇总 ==="
for f in $RESULT_DIR/*.log; do
    if [ -f "$f" ] && grep -q "F1-score" "$f" 2>/dev/null; then
        name=$(basename "$f" .log)
        f1=$(grep "F1-score" "$f" | tail -1 | sed 's/.*F1-score: //' | cut -d',' -f1)
        printf "  %-35s %s\n" "$name" "$f1"
    fi
done

echo ""
echo "=============================================="
echo "完成! 结果: $RESULT_DIR"
echo "=============================================="
