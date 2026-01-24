#!/bin/bash
# Auto Commit on Stop Script
# 会话结束时自动提交所有变更

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$PROJECT_ROOT" ]; then
    exit 0
fi

cd "$PROJECT_ROOT"

# 检查是否有未提交的变更
CHANGED_FILES=$(git status --porcelain 2>/dev/null | wc -l)

if [ "$CHANGED_FILES" -eq 0 ]; then
    exit 0
fi

# 生成提交信息
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

# 分析变更类型
MODIFIED=$(git status --porcelain | grep -E '^ M|^M ' | wc -l)
ADDED=$(git status --porcelain | grep -E '^\?\?|^A ' | wc -l)
DELETED=$(git status --porcelain | grep -E '^ D|^D ' | wc -l)

# 构建提交信息
COMMIT_MSG="chore: 自动保存 Claude Code 会话变更"
COMMIT_BODY="时间: $TIMESTAMP\n变更: $CHANGED_FILES 文件"

if [ "$MODIFIED" -gt 0 ]; then
    COMMIT_BODY="$COMMIT_BODY (修改: $MODIFIED)"
fi
if [ "$ADDED" -gt 0 ]; then
    COMMIT_BODY="$COMMIT_BODY (新增: $ADDED)"
fi
if [ "$DELETED" -gt 0 ]; then
    COMMIT_BODY="$COMMIT_BODY (删除: $DELETED)"
fi

# 执行提交
git add -A
git commit -m "$COMMIT_MSG" -m "$(echo -e $COMMIT_BODY)"

echo "✅ 已自动提交 $CHANGED_FILES 个文件变更"
exit 0
