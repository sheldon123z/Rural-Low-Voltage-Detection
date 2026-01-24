#!/bin/bash
# 训练监控脚本
# 持续监控训练进度，完成后自动运行分析

RESULT_DIR="./results/targeted_quick"
LOG_FILE="$RESULT_DIR/monitor.log"
CHECK_INTERVAL=30  # 检查间隔（秒）

mkdir -p $RESULT_DIR

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "开始监控训练进程"
log "=========================================="

while true; do
    # 统计运行中的进程
    running=$(ps aux | grep -E "python.*run\.py" | grep -v grep | wc -l)

    # 统计已完成的实验 (有 F1-score 输出的)
    completed=0
    for f in $RESULT_DIR/*.log; do
        if [ -f "$f" ] && grep -q "F1-score" "$f" 2>/dev/null; then
            ((completed++))
        fi
    done

    # GPU 状态
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "N/A")

    log "进程: $running | 已完成: $completed/16 | GPU: $gpu_info"

    # 如果没有运行中的进程，训练完成
    if [ $running -eq 0 ]; then
        log ""
        log "=========================================="
        log "所有训练完成！"
        log "=========================================="

        # 运行分析
        log "运行结果分析..."
        python scripts/analyze_targeted_results.py --result_dir $RESULT_DIR --no_timestamp 2>&1 | tee -a "$LOG_FILE"

        # 显示结果汇总
        log ""
        log "=== F1 分数汇总 ==="
        for f in $RESULT_DIR/*.log; do
            if [ -f "$f" ] && grep -q "F1-score" "$f" 2>/dev/null; then
                name=$(basename "$f" .log)
                f1=$(grep "F1-score" "$f" | tail -1)
                log "  $name: $f1"
            fi
        done

        log ""
        log "监控完成！结果保存在: $RESULT_DIR"
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
