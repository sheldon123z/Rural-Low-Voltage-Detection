#!/bin/bash
# 优化的训练脚本
# 特点:
# 1. 减少 num_workers (4个足够，避免 CPU 过载)
# 2. 增大 batch_size (充分利用 80GB 显存)
# 3. 串行执行，避免资源竞争
# 4. 完成检测，跳过已完成的训练

set -e

echo "=============================================="
echo "优化训练 - 针对性数据集实验"
echo "=============================================="
echo "优化配置:"
echo "  - num_workers: 4 (减少 CPU 负载)"
echo "  - batch_size: 根据数据集调整"
echo "  - 串行执行，一次只运行一个训练"
echo "=============================================="

# 优化后的参数
D_MODEL=64
D_FF=128
E_LAYERS=2
TOP_K=5
EPOCHS=10
LR=0.0001
ENC_IN=16
C_OUT=16
NUM_WORKERS=4  # 从 10 减少到 4

RESULT_DIR="./results/targeted_quick"
mkdir -p $RESULT_DIR

# 函数: 检查是否已完成
is_completed() {
    local LOG_FILE=$1
    if [ -f "$LOG_FILE" ] && grep -q "F1-score" "$LOG_FILE" 2>/dev/null; then
        return 0  # 已完成
    fi
    return 1  # 未完成
}

# 函数: 运行单个实验
run_experiment() {
    local MODEL=$1
    local DATASET=$2
    local RATIO=$3
    local BS=${4:-256}
    local SL=${5:-100}

    local LOG_FILE="$RESULT_DIR/${MODEL}_${DATASET}.log"

    # 检查是否已完成
    if is_completed "$LOG_FILE"; then
        echo "✓ 跳过已完成: $MODEL on $DATASET"
        grep "F1-score" "$LOG_FILE" | tail -1
        return
    fi

    echo ""
    echo ">>> 开始训练: $MODEL on $DATASET (batch=$BS, seq_len=$SL)"
    echo "    异常比例: ${RATIO}%"

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
        2>&1 | tee "$LOG_FILE"

    echo ">>> 完成: $MODEL on $DATASET"
}

# ============================================
# 实验 1: periodic_load 数据集 (anomaly_ratio=15)
# 目标: 验证 VoltageTimesNet 的预设周期机制
# ============================================
echo ""
echo "========== 周期性负荷数据集 (15% 异常) =========="
run_experiment "TimesNet" "periodic_load" 15 256 100
run_experiment "VoltageTimesNet" "periodic_load" 15 256 100

# ============================================
# 实验 2: three_phase 数据集 (anomaly_ratio=23)
# 目标: 验证 TPATimesNet 的三相注意力机制
# ============================================
echo ""
echo "========== 三相不平衡数据集 (23% 异常) =========="
run_experiment "TimesNet" "three_phase" 23 256 100
run_experiment "TPATimesNet" "three_phase" 23 256 100

# ============================================
# 实验 3: multi_scale 数据集 (anomaly_ratio=47)
# 目标: 验证 MTSTimesNet 的多尺度时序建模
# 注: 使用更长序列和较小 batch
# ============================================
echo ""
echo "========== 多尺度复合数据集 (47% 异常) =========="
run_experiment "TimesNet" "multi_scale" 47 128 200
run_experiment "MTSTimesNet" "multi_scale" 47 128 200

# ============================================
# 实验 4: comprehensive 数据集 (anomaly_ratio=49)
# 目标: 公平对比所有模型
# ============================================
echo ""
echo "========== 综合评估数据集 (49% 异常) =========="
run_experiment "TimesNet" "comprehensive" 49 256 100
run_experiment "VoltageTimesNet" "comprehensive" 49 256 100
run_experiment "TPATimesNet" "comprehensive" 49 256 100
run_experiment "MTSTimesNet" "comprehensive" 49 256 100

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
