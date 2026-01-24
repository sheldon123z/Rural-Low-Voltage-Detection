#!/bin/bash
# Periodic Auto Commit Script
# 定期检查并自动提交变更（每N次工具调用或时间间隔）
# 用于 PostToolUse hook

# 配置
COMMIT_INTERVAL=300  # 最小提交间隔（秒）
MIN_CHANGES=3        # 最少变更文件数才提交

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$PROJECT_ROOT" || exit 0

TRACK_FILE="${PROJECT_ROOT}/.claude/.auto_commit_track"
CURRENT_TIME=$(date +%s)

# 读取上次提交时间
LAST_COMMIT_TIME=0
if [ -f "$TRACK_FILE" ]; then
    LAST_COMMIT_TIME=$(cat "$TRACK_FILE" 2>/dev/null || echo 0)
fi

# 检查时间间隔
TIME_DIFF=$((CURRENT_TIME - LAST_COMMIT_TIME))
if [ "$TIME_DIFF" -lt "$COMMIT_INTERVAL" ]; then
    exit 0
fi

# 检查变更数量
CHANGED_FILES=$(git status --porcelain 2>/dev/null | wc -l)
if [ "$CHANGED_FILES" -lt "$MIN_CHANGES" ]; then
    exit 0
fi

# 执行提交
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
COMMIT_MSG="chore: 自动保存进度 (${TIMESTAMP} - ${CHANGED_FILES} 文件)"

git add -A 2>/dev/null || exit 0
git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || exit 0

# 更新追踪文件
echo "$CURRENT_TIME" > "$TRACK_FILE"

echo "🔄 定期提交: $CHANGED_FILES 个文件"
exit 0
