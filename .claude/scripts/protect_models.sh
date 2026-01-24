#!/bin/bash
# 模型代码保护脚本
# 只允许修改 TimesNet 系列模型，保护其他对比模型

FILE_PATH="$CLAUDE_FILE_PATH"

# 检查是否是 models 目录下的 Python 文件
if ! echo "$FILE_PATH" | grep -qE 'models/.*\.py$'; then
    exit 0  # 不是模型文件，放行
fi

# 允许修改的文件（TimesNet 系列 + __init__.py）
ALLOWED_PATTERNS=(
    "TimesNet\.py$"
    "VoltageTimesNet\.py$"
    "HybridTimesNet\.py$"
    "MTSTimesNet\.py$"
    "TPATimesNet\.py$"
    "__init__\.py$"
)

for pattern in "${ALLOWED_PATTERNS[@]}"; do
    if echo "$FILE_PATH" | grep -qE "$pattern"; then
        exit 0  # 允许修改
    fi
done

# 如果文件不存在，说明是新建模型，允许
if [ ! -f "$FILE_PATH" ]; then
    exit 0  # 允许新建
fi

# 其他情况：修改已存在的对比模型，禁止
FILENAME=$(basename "$FILE_PATH")
echo "⚠️ 禁止修改对比模型: $FILENAME"
echo "   允许: TimesNet系列, 新建模型"
exit 1
