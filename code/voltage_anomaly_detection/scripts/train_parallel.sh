#!/bin/bash
# 并行训练脚本 - 充分利用 A800 80GB 显存
#
# 策略:
# 1. 同时运行多个训练任务 (每个约 2-3GB 显存)
# 2. 使用较少的 DataLoader workers (每任务 2 个)
# 3. 最多同时运行 MAX_PARALLEL 个任务
# 4. 自动跳过已完成的实验

set -e

echo "=============================================="
echo "并行训练 - 针对性数据集实验"
echo "=============================================="

# 并行配置
MAX_PARALLEL=8          # 最大并行任务数 (保守估计: 8*3GB=24GB < 80GB)
NUM_WORKERS=2           # 每个任务的 DataLoader workers
WAIT_INTERVAL=5         # 检查间隔 (秒)

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

# 记录正在运行的任务
declare -A RUNNING_PIDS
declare -A TASK_NAMES

# 函数: 检查是否已完成
is_completed() {
    local LOG_FILE=$1
    if [ -f "$LOG_FILE" ] && grep -q "F1-score" "$LOG_FILE" 2>/dev/null; then
        return 0
    fi
    return 1
}

# 函数: 等待有空闲槽位
wait_for_slot() {
    while true; do
        local running=0
        local finished_pids=()

        for pid in "${!RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            else
                finished_pids+=("$pid")
            fi
        done

        # 清理已完成的任务
        for pid in "${finished_pids[@]}"; do
            local name="${TASK_NAMES[$pid]}"
            wait "$pid" 2>/dev/null
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "  ✓ 完成: $name"
            else
                echo "  ✗ 失败: $name (exit code: $exit_code)"
            fi
            unset RUNNING_PIDS[$pid]
            unset TASK_NAMES[$pid]
        done

        if [ $running -lt $MAX_PARALLEL ]; then
            return
        fi

        sleep $WAIT_INTERVAL
    done
}

# 函数: 等待所有任务完成
wait_all() {
    echo ""
    echo "等待所有任务完成..."
    for pid in "${!RUNNING_PIDS[@]}"; do
        local name="${TASK_NAMES[$pid]}"
        wait "$pid" 2>/dev/null
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ 完成: $name"
        else
            echo "  ✗ 失败: $name (exit code: $exit_code)"
        fi
    done
    RUNNING_PIDS=()
    TASK_NAMES=()
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
        echo "  ⏭ 跳过已完成: $TASK_NAME"
        return
    fi

    # 等待有空闲槽位
    wait_for_slot

    echo "  🚀 启动: $TASK_NAME (batch=$BS, seq=$SL)"

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

    local pid=$!
    RUNNING_PIDS[$pid]=1
    TASK_NAMES[$pid]=$TASK_NAME
}

# ============================================
# 开始并行训练
# ============================================
echo ""
echo "配置:"
echo "  - 最大并行任务: $MAX_PARALLEL"
echo "  - 每任务 workers: $NUM_WORKERS"
echo "  - 训练轮次: $EPOCHS"
echo ""
echo "启动训练任务..."

# 周期性负荷数据集 (anomaly_ratio=15)
echo ""
echo "[1/4] 周期性负荷数据集 (15% 异常)"
start_training "TimesNet" "periodic_load" 15 256 100
start_training "VoltageTimesNet" "periodic_load" 15 256 100
start_training "TPATimesNet" "periodic_load" 15 256 100
start_training "MTSTimesNet" "periodic_load" 15 256 100

# 三相不平衡数据集 (anomaly_ratio=23)
echo ""
echo "[2/4] 三相不平衡数据集 (23% 异常)"
start_training "TimesNet" "three_phase" 23 256 100
start_training "VoltageTimesNet" "three_phase" 23 256 100
start_training "TPATimesNet" "three_phase" 23 256 100
start_training "MTSTimesNet" "three_phase" 23 256 100

# 多尺度复合数据集 (anomaly_ratio=47)
echo ""
echo "[3/4] 多尺度复合数据集 (47% 异常)"
start_training "TimesNet" "multi_scale" 47 128 200
start_training "VoltageTimesNet" "multi_scale" 47 128 200
start_training "TPATimesNet" "multi_scale" 47 128 200
start_training "MTSTimesNet" "multi_scale" 47 128 200

# 综合评估数据集 (anomaly_ratio=49)
echo ""
echo "[4/4] 综合评估数据集 (49% 异常)"
start_training "TimesNet" "comprehensive" 49 256 100
start_training "VoltageTimesNet" "comprehensive" 49 256 100
start_training "TPATimesNet" "comprehensive" 49 256 100
start_training "MTSTimesNet" "comprehensive" 49 256 100

# 等待所有任务完成
wait_all

# ============================================
# 生成分析报告
# ============================================
echo ""
echo "=============================================="
echo "所有训练完成! 生成分析报告..."
echo "=============================================="

python scripts/analyze_targeted_results.py --result_dir $RESULT_DIR --no_timestamp

echo ""
echo "=============================================="
echo "实验完成!"
echo "结果目录: $RESULT_DIR"
echo "=============================================="

# 显示结果汇总
echo ""
echo "=== F1 分数汇总 ==="
for f in $RESULT_DIR/*.log; do
    if grep -q "F1-score" "$f" 2>/dev/null; then
        name=$(basename "$f" .log)
        f1=$(grep "F1-score" "$f" | tail -1 | grep -oP "F1-score: \K[\d.]+")
        printf "%-35s F1: %s\n" "$name" "$f1"
    fi
done
