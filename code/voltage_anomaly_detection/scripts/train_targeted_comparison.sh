#!/bin/bash
# 针对性数据集训练对比脚本
# 在各个针对性数据集上训练对应的优势模型进行对比

set -e

echo "=============================================="
echo "针对性农村低电压数据集训练对比实验"
echo "=============================================="

# 通用参数
SEQ_LEN=100
D_MODEL=64
D_FF=128
E_LAYERS=2
TOP_K=5
BATCH_SIZE=128
EPOCHS=10
LR=0.0001
ENC_IN=17
C_OUT=17

# 结果目录
RESULT_DIR="./results/targeted_comparison"
mkdir -p $RESULT_DIR

# ============================================
# 实验1: 周期性负荷数据集 (periodic_load)
# 目标: 验证 VoltageTimesNet 的预设周期优势
# ============================================
echo ""
echo "=============================================="
echo "实验1: 周期性负荷数据集 (periodic_load)"
echo "预期优势模型: VoltageTimesNet"
echo "=============================================="

DATA_PATH="./dataset/RuralVoltage/periodic_load"
SETTING="periodic_load"

for MODEL in TimesNet VoltageTimesNet TPATimesNet MTSTimesNet; do
    echo ""
    echo ">>> 训练 $MODEL 在 $SETTING 数据集上..."

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_PATH \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --anomaly_ratio 15 \
        2>&1 | tee "$RESULT_DIR/${MODEL}_${SETTING}.log"
done

# ============================================
# 实验2: 三相不平衡数据集 (three_phase)
# 目标: 验证 TPATimesNet 的三相注意力优势
# ============================================
echo ""
echo "=============================================="
echo "实验2: 三相不平衡数据集 (three_phase)"
echo "预期优势模型: TPATimesNet"
echo "=============================================="

DATA_PATH="./dataset/RuralVoltage/three_phase"
SETTING="three_phase"

for MODEL in TimesNet VoltageTimesNet TPATimesNet MTSTimesNet; do
    echo ""
    echo ">>> 训练 $MODEL 在 $SETTING 数据集上..."

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_PATH \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --anomaly_ratio 23 \
        2>&1 | tee "$RESULT_DIR/${MODEL}_${SETTING}.log"
done

# ============================================
# 实验3: 多尺度复合异常数据集 (multi_scale)
# 目标: 验证 MTSTimesNet 的多尺度建模优势
# ============================================
echo ""
echo "=============================================="
echo "实验3: 多尺度复合异常数据集 (multi_scale)"
echo "预期优势模型: MTSTimesNet"
echo "=============================================="

DATA_PATH="./dataset/RuralVoltage/multi_scale"
SETTING="multi_scale"

# 多尺度数据集使用更长的序列
SEQ_LEN_MS=200

for MODEL in TimesNet VoltageTimesNet TPATimesNet MTSTimesNet; do
    echo ""
    echo ">>> 训练 $MODEL 在 $SETTING 数据集上..."

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_PATH \
        --seq_len $SEQ_LEN_MS \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size 64 \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --anomaly_ratio 47 \
        2>&1 | tee "$RESULT_DIR/${MODEL}_${SETTING}.log"
done

# ============================================
# 实验4: 混合周期数据集 (hybrid_period)
# 目标: 验证 HybridTimesNet 的混合周期发现优势
# ============================================
echo ""
echo "=============================================="
echo "实验4: 混合周期数据集 (hybrid_period)"
echo "预期优势模型: HybridTimesNet"
echo "=============================================="

DATA_PATH="./dataset/RuralVoltage/hybrid_period"
SETTING="hybrid_period"

for MODEL in TimesNet VoltageTimesNet TPATimesNet MTSTimesNet; do
    echo ""
    echo ">>> 训练 $MODEL 在 $SETTING 数据集上..."

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_PATH \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --anomaly_ratio 60 \
        2>&1 | tee "$RESULT_DIR/${MODEL}_${SETTING}.log"
done

# ============================================
# 实验5: 综合评估数据集 (comprehensive)
# 目标: 公平对比所有模型的综合能力
# ============================================
echo ""
echo "=============================================="
echo "实验5: 综合评估数据集 (comprehensive)"
echo "目标: 公平对比所有模型"
echo "=============================================="

DATA_PATH="./dataset/RuralVoltage/comprehensive"
SETTING="comprehensive"

for MODEL in TimesNet VoltageTimesNet TPATimesNet MTSTimesNet; do
    echo ""
    echo ">>> 训练 $MODEL 在 $SETTING 数据集上..."

    python run.py \
        --is_training 1 \
        --model $MODEL \
        --data RuralVoltage \
        --root_path $DATA_PATH \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --e_layers $E_LAYERS \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --anomaly_ratio 49 \
        2>&1 | tee "$RESULT_DIR/${MODEL}_${SETTING}.log"
done

echo ""
echo "=============================================="
echo "所有训练完成!"
echo "结果保存在: $RESULT_DIR"
echo "=============================================="

# 运行结果分析
echo ""
echo "运行结果分析..."
python scripts/analyze_targeted_results.py --result_dir $RESULT_DIR
