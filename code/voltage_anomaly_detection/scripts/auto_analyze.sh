#!/bin/bash
# 自动监控训练并触发分析
# 定期检查训练进度，完成后自动运行分析脚本

RESULT_DIR="./results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$RESULT_DIR/analysis_$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/monitor.log"
CHECK_INTERVAL=60  # 检查间隔（秒）

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "自动分析监控启动"
log "输出目录: $OUTPUT_DIR"
log "=========================================="

# 等待训练完成
while true; do
    # 统计运行中的训练进程
    running=$(ps aux | grep -E "python.*run\.py" | grep -v grep | wc -l)

    # GPU 状态
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

    # 获取最新训练进度
    for pid in $(pgrep -f "python.*run.py"); do
        epoch_info=$(cat /proc/$pid/fd/1 2>/dev/null | grep "Epoch:" | tail -1)
        if [ -n "$epoch_info" ]; then
            log "进程 $pid: $epoch_info | GPU: $gpu_info"
        fi
    done

    # 如果没有运行中的进程，训练完成
    if [ $running -eq 0 ]; then
        log ""
        log "=========================================="
        log "所有训练完成！开始分析..."
        log "=========================================="
        break
    fi

    sleep $CHECK_INTERVAL
done

# 收集所有结果
log "收集实验结果..."
cp result_anomaly_detection.txt "$OUTPUT_DIR/"

# 创建结果汇总 JSON
log "生成结果汇总..."
python3 << 'PYTHON_SCRIPT'
import json
import re
from pathlib import Path
from datetime import datetime

output_dir = Path("$OUTPUT_DIR".replace("$OUTPUT_DIR", "$OUTPUT_DIR"))
results_file = Path("result_anomaly_detection.txt")

experiments = {}
current_exp = None

with open(results_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if 'Accuracy:' in line:
            # 解析指标行
            metrics = {}
            for item in line.split(', '):
                if ':' in item:
                    key, val = item.split(':')
                    metrics[key.strip()] = float(val.strip())
            if current_exp:
                experiments[current_exp] = metrics
        else:
            # 实验名称行
            current_exp = line

# 按数据集和模型分组
grouped = {}
for exp_name, metrics in experiments.items():
    # 解析实验名
    parts = exp_name.split('_')
    if 'RuralVoltage' in exp_name:
        dataset = 'RuralVoltage'
    elif 'PSM' in exp_name:
        dataset = 'PSM'
    else:
        dataset = 'Unknown'

    # 提取模型名
    for model in ['VoltageTimesNet', 'TimesNet', 'TPATimesNet', 'MTSTimesNet', 'DLinear']:
        if model in exp_name:
            model_name = model
            break
    else:
        model_name = 'Unknown'

    key = f"{dataset}_{model_name}"
    if key not in grouped:
        grouped[key] = []
    grouped[key].append({
        'experiment': exp_name,
        'metrics': metrics
    })

# 保存结果
output = {
    'timestamp': datetime.now().isoformat(),
    'total_experiments': len(experiments),
    'grouped_results': grouped,
    'all_results': experiments
}

output_path = output_dir / 'experiment_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"结果已保存到: {output_path}")
print(f"共 {len(experiments)} 个实验")
PYTHON_SCRIPT

log "分析完成！结果保存在: $OUTPUT_DIR"
log "=========================================="

# 显示结果汇总
echo ""
echo "=== 实验结果汇总 ==="
cat "$OUTPUT_DIR/experiment_results.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"总实验数: {data['total_experiments']}\")
print()
for group, exps in sorted(data['grouped_results'].items()):
    best = max(exps, key=lambda x: x['metrics'].get('F1-score', 0))
    print(f\"{group}: 最佳 F1={best['metrics'].get('F1-score', 0):.4f}\")
"
