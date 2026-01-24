#!/bin/bash
# Auto Commit on Stop Script
# 会话结束时自动提交所有变更
# 优化版本：更可靠的错误处理和日志记录

set -e

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "${PROJECT_ROOT}/.claude/.auto_commit.log" 2>/dev/null || true
}

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Not in a git repository"
    exit 0
}

cd "$PROJECT_ROOT" || exit 0

log "开始自动提交检查..."

# 检查是否有未提交的变更
CHANGED_FILES=$(git status --porcelain 2>/dev/null | wc -l)

if [ "$CHANGED_FILES" -eq 0 ]; then
    log "无变更需要提交"
    exit 0
fi

# 生成时间戳
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

# 分析变更类型
MODIFIED=$(git status --porcelain | grep -cE '^ M|^M ' || true)
ADDED=$(git status --porcelain | grep -cE '^\?\?|^A ' || true)
DELETED=$(git status --porcelain | grep -cE '^ D|^D ' || true)

# 构建简洁的提交信息
COMMIT_MSG="chore: 自动保存进度 (${TIMESTAMP})"

# 构建详细信息
DETAILS=""
[ "$MODIFIED" -gt 0 ] && DETAILS="${DETAILS}修改:$MODIFIED "
[ "$ADDED" -gt 0 ] && DETAILS="${DETAILS}新增:$ADDED "
[ "$DELETED" -gt 0 ] && DETAILS="${DETAILS}删除:$DELETED"

# 执行提交
git add -A 2>/dev/null || {
    log "git add 失败"
    exit 1
}

git commit -m "$COMMIT_MSG" -m "$DETAILS" --no-verify 2>/dev/null || {
    log "git commit 失败或无变更"
    exit 0
}

log "✅ 已提交 $CHANGED_FILES 个文件 ($DETAILS)"
echo "✅ 已自动提交 $CHANGED_FILES 个文件变更"

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
