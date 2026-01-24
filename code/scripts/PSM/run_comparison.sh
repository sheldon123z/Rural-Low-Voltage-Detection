#!/bin/bash
# PSM 数据集多模型故障检测对比实验
# 10 个模型：6 基线 + 4 创新

set -e

# 切换到代码目录
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"

# 加载公共配置
source scripts/PSM/config.sh

# 创建结果目录
mkdir -p "${RESULT_DIR}"
echo "结果目录: ${RESULT_DIR}"

# 模型列表（10 个模型）
MODELS=(
    # 基线模型 (6)
    "TimesNet"
    "Transformer"
    "DLinear"
    "PatchTST"
    "iTransformer"
    "Autoformer"
    # 创新模型 (4)
    "VoltageTimesNet"
    "TPATimesNet"
    "MTSTimesNet"
    "HybridTimesNet"
)

echo "=========================================="
echo "PSM 数据集多模型故障检测对比实验"
echo "=========================================="
echo "时间戳: ${TIMESTAMP}"
echo "模型数量: ${#MODELS[@]}"
echo "数据集: PSM (25 维特征)"
echo "序列长度: ${SEQ_LEN}"
echo "训练轮数: ${TRAIN_EPOCHS}"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 训练每个模型
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    idx=$((i + 1))

    echo ""
    echo "------------------------------------------"
    echo "[${idx}/${#MODELS[@]}] 模型: ${model}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    # 模型特定参数
    MODEL_ARGS=""
    case $model in
        TimesNet|VoltageTimesNet|MTSTimesNet|HybridTimesNet)
            MODEL_ARGS="--top_k ${TOP_K} --num_kernels ${NUM_KERNELS}"
            ;;
        TPATimesNet)
            MODEL_ARGS="--top_k ${TOP_K}"
            ;;
        Autoformer)
            MODEL_ARGS="--moving_avg 25 --factor 1"
            ;;
        PatchTST)
            MODEL_ARGS="--patch_len 16 --stride 8"
            ;;
    esac

    # 执行训练
    python -u run.py \
        --model_id "PSM_${model}" \
        --model "${model}" \
        ${COMMON_ARGS} \
        ${MODEL_ARGS} \
        2>&1 | tee "${RESULT_DIR}/${model}.log"

    echo "[${idx}/${#MODELS[@]}] 模型 ${model} 完成: $(date '+%Y-%m-%d %H:%M:%S')"
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "所有模型训练完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "结果目录: ${RESULT_DIR}"
echo "=========================================="

# 运行结果分析
echo ""
echo "正在分析实验结果..."
python scripts/analyze_comparison_results.py \
    --result_dir "${RESULT_DIR}" \
    --dataset PSM \
    2>&1 || echo "结果分析脚本未找到或执行失败，请手动分析"

echo ""
echo "实验完成！"
