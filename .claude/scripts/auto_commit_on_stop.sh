#!/bin/bash
# Auto Commit on Stop Script
# 会话结束时自动提交所有变更
# 优化版：使用智能 commit 信息生成，遵循 Conventional Commits 规范

set -e

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Not in a git repository"
    exit 0
}

cd "$PROJECT_ROOT" || exit 0

LOG_FILE="${PROJECT_ROOT}/.claude/.auto_commit.log"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}

log "开始自动提交检查..."

# 检查是否有未提交的变更
CHANGED_FILES_COUNT=$(git status --porcelain 2>/dev/null | wc -l)

if [ "$CHANGED_FILES_COUNT" -eq 0 ]; then
    log "无变更需要提交"
    exit 0
fi

# 先暂存所有文件以便分析
git add -A 2>/dev/null || {
    log "git add 失败"
    exit 1
}

# 使用智能消息生成器
SCRIPT_DIR="${PROJECT_ROOT}/.claude/scripts"
if [ -x "${SCRIPT_DIR}/smart_commit_msg.sh" ]; then
    COMMIT_MSG=$(bash "${SCRIPT_DIR}/smart_commit_msg.sh" 2>/dev/null)
    if [ -z "$COMMIT_MSG" ] || [ "$COMMIT_MSG" = "chore: 无变更" ]; then
        # 降级到基本描述
        COMMIT_MSG="chore: 自动保存进度 (${CHANGED_FILES_COUNT}个文件)"
    fi
else
    # 降级到基本描述
    COMMIT_MSG="chore: 自动保存进度 (${CHANGED_FILES_COUNT}个文件)"
fi

# 分析变更统计（用于日志）
MODIFIED=$(git status --porcelain | grep -cE '^ M|^M ' || true)
ADDED=$(git status --porcelain | grep -cE '^\?\?|^A ' || true)
DELETED=$(git status --porcelain | grep -cE '^ D|^D ' || true)

DETAILS=""
[ "$MODIFIED" -gt 0 ] && DETAILS="${DETAILS}修改:$MODIFIED "
[ "$ADDED" -gt 0 ] && DETAILS="${DETAILS}新增:$ADDED "
[ "$DELETED" -gt 0 ] && DETAILS="${DETAILS}删除:$DELETED"

# 执行提交
git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || {
    log "git commit 失败或无变更"
    exit 0
}

log "✅ 已提交: $COMMIT_MSG ($DETAILS)"
echo "✅ 已自动提交: $COMMIT_MSG"
echo "   $CHANGED_FILES_COUNT 个文件 ($DETAILS)"

# 检查未push的commit数量并自动push（超过5个则push）
MAX_UNPUSHED=5
REMOTE_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null) || exit 0
UNPUSHED_COUNT=$(git rev-list --count @{u}..HEAD 2>/dev/null) || exit 0

if [ "$UNPUSHED_COUNT" -gt "$MAX_UNPUSHED" ]; then
    # 先推送LFS对象（如果有）
    git lfs push --all origin 2>/dev/null || true
    git push --no-verify 2>/dev/null && {
        log "📤 自动推送: $UNPUSHED_COUNT 个提交"
        echo "📤 自动推送 $UNPUSHED_COUNT 个提交到远程"
    } || true
fi

exit 0
