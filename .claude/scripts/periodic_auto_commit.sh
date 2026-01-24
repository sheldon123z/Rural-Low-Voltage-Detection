#!/bin/bash
# Periodic Auto Commit Script
# 定期检查并自动提交变更（每N次工具调用或时间间隔）
# 用于 PostToolUse hook
# 优化版：使用智能 commit 信息生成

# 配置
COMMIT_INTERVAL=1200  # 最小提交间隔（秒）= 20分钟
MIN_CHANGES=8         # 最少变更文件数才提交
MAX_UNPUSHED=5        # 超过此数量未push的commit则自动push

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$PROJECT_ROOT" || exit 0

TRACK_FILE="${PROJECT_ROOT}/.claude/.auto_commit_track"
LOG_FILE="${PROJECT_ROOT}/.claude/.auto_commit.log"
CURRENT_TIME=$(date +%s)

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}

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

# 先暂存所有文件以便分析
git add -A 2>/dev/null || exit 0

# 使用智能消息生成器
SCRIPT_DIR="${PROJECT_ROOT}/.claude/scripts"
if [ -x "${SCRIPT_DIR}/smart_commit_msg.sh" ]; then
    COMMIT_MSG=$(bash "${SCRIPT_DIR}/smart_commit_msg.sh" 2>/dev/null)
    if [ -z "$COMMIT_MSG" ] || [ "$COMMIT_MSG" = "chore: 无变更" ]; then
        COMMIT_MSG="chore: 自动保存进度 (${CHANGED_FILES}个文件)"
    fi
else
    COMMIT_MSG="chore: 自动保存进度 (${CHANGED_FILES}个文件)"
fi

# 执行提交
git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || {
    log "提交失败或无变更"
    exit 0
}

# 更新追踪文件
echo "$CURRENT_TIME" > "$TRACK_FILE"

log "✅ 定期提交: $COMMIT_MSG"
echo "🔄 定期提交: $CHANGED_FILES 个文件"

# 检查未push的commit数量并自动push
REMOTE_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null) || exit 0
UNPUSHED_COUNT=$(git rev-list --count @{u}..HEAD 2>/dev/null) || exit 0

if [ "$UNPUSHED_COUNT" -gt "$MAX_UNPUSHED" ]; then
    # 先推送LFS对象（如果有）
    git lfs push --all origin 2>/dev/null || true
    git push --no-verify 2>/dev/null && {
        log "📤 自动推送: $UNPUSHED_COUNT 个提交"
        echo "📤 自动推送: $UNPUSHED_COUNT 个提交"
    } || true
fi

exit 0
