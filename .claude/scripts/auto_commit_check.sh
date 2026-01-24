#!/bin/bash
# Auto Commit Check Script
# 当修改文件超过5个时自动提交

# 获取项目根目录
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$PROJECT_ROOT" ]; then
    exit 0
fi

cd "$PROJECT_ROOT"

# 统计已修改的文件数量（包括新增、修改、删除）
MODIFIED_COUNT=$(git status --porcelain 2>/dev/null | grep -E '^[MADRC]' | wc -l)
UNSTAGED_COUNT=$(git status --porcelain 2>/dev/null | grep -E '^ [MADRC]|^\?\?' | wc -l)
TOTAL_CHANGED=$((MODIFIED_COUNT + UNSTAGED_COUNT))

# 记录文件用于跟踪变更
TRACK_FILE="$PROJECT_ROOT/.claude/.auto_commit_track"

# 如果跟踪文件不存在，创建它
if [ ! -f "$TRACK_FILE" ]; then
    echo "0" > "$TRACK_FILE"
fi

# 读取上次的计数
LAST_COUNT=$(cat "$TRACK_FILE" 2>/dev/null || echo "0")

# 更新计数
echo "$TOTAL_CHANGED" > "$TRACK_FILE"

# 如果修改文件数超过5个，且有新增修改，提示自动提交
if [ "$TOTAL_CHANGED" -ge 5 ] && [ "$TOTAL_CHANGED" -gt "$LAST_COUNT" ]; then
    echo ""
    echo "📦 检测到 $TOTAL_CHANGED 个文件变更，建议执行自动提交"
    echo "   运行: git add -A && git commit -m 'chore: 自动保存进度 ($TOTAL_CHANGED 文件)'"
    echo ""

    # 可选：取消下面的注释以启用真正的自动提交
    git add -A
    git commit -m "chore: 自动保存进度 ($(date '+%Y-%m-%d %H:%M') - $TOTAL_CHANGED 文件)"
fi

exit 0
