#!/bin/bash
# Smart Commit Message Generator
# 智能分析变更内容并生成规范的 commit 信息
# 遵循 Conventional Commits 规范

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 1
cd "$PROJECT_ROOT"

# 获取变更文件列表
CHANGED_FILES=$(git diff --cached --name-only 2>/dev/null)
if [ -z "$CHANGED_FILES" ]; then
    CHANGED_FILES=$(git status --porcelain 2>/dev/null | awk '{print $2}')
fi

# 如果没有变更，退出
if [ -z "$CHANGED_FILES" ]; then
    echo "chore: 无变更"
    exit 0
fi

# 初始化计数器
count_feat=0
count_fix=0
count_docs=0
count_refactor=0
count_test=0
count_chore=0
count_model=0
count_data=0
count_exp=0

# 收集各类变更的详细信息
model_details=""
data_details=""
exp_details=""
feat_details=""
fix_details=""
docs_details=""
test_details=""
refactor_details=""
chore_details=""

# 分析每个变更文件
while IFS= read -r file; do
    [ -z "$file" ] && continue

    basename_file=$(basename "$file")

    # 根据文件路径和扩展名分类
    case "$file" in
        # 模型相关
        */models/*.py|*/layers/*.py)
            ((count_model++)) || true
            model_details="${model_details}${basename_file},"
            ;;
        # 数据处理
        */data_provider/*.py|*/dataset/*.py)
            ((count_data++)) || true
            data_details="${data_details}${basename_file},"
            ;;
        # 实验脚本 (非.claude目录)
        code/*/scripts/*.sh|*/exp/*.py|run.py)
            ((count_exp++)) || true
            exp_details="${exp_details}${basename_file},"
            ;;
        # 测试文件
        *test*.py|*_test.py|tests/*)
            ((count_test++)) || true
            test_details="${test_details}${basename_file},"
            ;;
        # 文档文件
        *.md|*.rst|*.txt|docs/*|thesis/*.tex)
            ((count_docs++)) || true
            docs_details="${docs_details}${basename_file},"
            ;;
        # Hook 和 Claude 配置 (需要在通用配置之前匹配)
        .claude/*)
            ((count_chore++)) || true
            chore_details="${chore_details}${basename_file},"
            ;;
        # 配置文件
        *.json|*.yaml|*.yml|*.toml|*.ini|*.cfg|requirements*.txt|setup.py|pyproject.toml)
            ((count_chore++)) || true
            chore_details="${chore_details}${basename_file},"
            ;;
        # Python 源文件 - 进一步分析
        *.py)
            # 检查 git diff 内容来判断类型
            diff_content=$(git diff --cached -- "$file" 2>/dev/null || git diff -- "$file" 2>/dev/null || echo "")

            if echo "$diff_content" | grep -qE '^\+.*def\s+\w+|^\+.*class\s+\w+'; then
                # 新增函数或类 -> feat
                ((count_feat++)) || true
                feat_details="${feat_details}${basename_file},"
            elif echo "$diff_content" | grep -qiE '^\+.*fix|^\+.*bug|^\+.*error|^\+.*issue'; then
                # 修复相关
                ((count_fix++)) || true
                fix_details="${fix_details}${basename_file},"
            else
                # 默认为重构
                ((count_refactor++)) || true
                refactor_details="${refactor_details}${basename_file},"
            fi
            ;;
        # 图片和结果文件
        *.png|*.pdf|*.jpg|*.jpeg|results/*)
            ((count_exp++)) || true
            exp_details="${exp_details}${basename_file},"
            ;;
        # 其他
        *)
            ((count_chore++)) || true
            chore_details="${chore_details}${basename_file},"
            ;;
    esac
done <<< "$CHANGED_FILES"

# 确定主要变更类型
main_type="chore"
max_count=0

# 检查各类型计数
if [ "$count_model" -gt "$max_count" ]; then
    max_count=$count_model
    main_type="model"
fi
if [ "$count_feat" -gt "$max_count" ]; then
    max_count=$count_feat
    main_type="feat"
fi
if [ "$count_exp" -gt "$max_count" ]; then
    max_count=$count_exp
    main_type="exp"
fi
if [ "$count_data" -gt "$max_count" ]; then
    max_count=$count_data
    main_type="data"
fi
if [ "$count_fix" -gt "$max_count" ]; then
    max_count=$count_fix
    main_type="fix"
fi
if [ "$count_refactor" -gt "$max_count" ]; then
    max_count=$count_refactor
    main_type="refactor"
fi
if [ "$count_docs" -gt "$max_count" ]; then
    max_count=$count_docs
    main_type="docs"
fi
if [ "$count_test" -gt "$max_count" ]; then
    max_count=$count_test
    main_type="test"
fi
# chore 保持默认，不需要额外检查

# 映射内部类型到 Conventional Commits 类型
scope=""
case "$main_type" in
    model)
        commit_type="feat"
        scope="model"
        details="$model_details"
        ;;
    data)
        commit_type="feat"
        scope="data"
        details="$data_details"
        ;;
    exp)
        commit_type="feat"
        scope="exp"
        details="$exp_details"
        ;;
    feat)
        commit_type="feat"
        details="$feat_details"
        ;;
    fix)
        commit_type="fix"
        details="$fix_details"
        ;;
    docs)
        commit_type="docs"
        details="$docs_details"
        ;;
    test)
        commit_type="test"
        details="$test_details"
        ;;
    refactor)
        commit_type="refactor"
        details="$refactor_details"
        ;;
    *)
        commit_type="chore"
        details="$chore_details"
        ;;
esac

# 生成简洁描述
generate_description() {
    local details="$1"
    local main_type="$2"

    # 移除末尾逗号并分割
    details="${details%,}"

    # 统计文件数
    local count=$(echo "$details" | tr ',' '\n' | grep -c . || echo 0)

    if [ "$count" -eq 0 ]; then
        echo "更新代码"
    elif [ "$count" -eq 1 ]; then
        echo "$details"
    elif [ "$count" -le 3 ]; then
        echo "$details" | tr ',' ' '
    else
        local first=$(echo "$details" | cut -d',' -f1)
        echo "${first}等${count}个文件"
    fi
}

# 生成描述
file_desc=$(generate_description "$details" "$main_type")

# 构建描述信息
case "$main_type" in
    model)
        description="更新模型 ${file_desc}"
        ;;
    data)
        description="更新数据处理 ${file_desc}"
        ;;
    exp)
        description="更新实验脚本 ${file_desc}"
        ;;
    feat)
        description="添加新功能 ${file_desc}"
        ;;
    fix)
        description="修复 ${file_desc}"
        ;;
    docs)
        description="更新文档 ${file_desc}"
        ;;
    test)
        description="更新测试 ${file_desc}"
        ;;
    refactor)
        description="重构 ${file_desc}"
        ;;
    *)
        total_files=$(echo "$CHANGED_FILES" | wc -l)
        description="更新配置 ${file_desc}"
        ;;
esac

# 构建最终 commit 信息
if [ -n "$scope" ]; then
    commit_msg="${commit_type}(${scope}): ${description}"
else
    commit_msg="${commit_type}: ${description}"
fi

# 限制长度（GitHub 推荐 72 字符以内）
if [ ${#commit_msg} -gt 72 ]; then
    commit_msg="${commit_msg:0:69}..."
fi

echo "$commit_msg"
