#!/bin/bash
# CI/CD Local Testing Script
# 本脚本用于在提交前本地运行 CI 检查

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="$PROJECT_ROOT/code/voltage_anomaly_detection"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CI/CD 本地测试脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}检查 Python 环境...${NC}"
python --version
echo ""

# 1. 代码格式化检查
echo -e "${YELLOW}[1/5] 运行 Black 格式化检查...${NC}"
cd "$CODE_DIR"
if black --check --diff .; then
    echo -e "${GREEN}✓ Black 检查通过${NC}"
else
    echo -e "${RED}✗ Black 检查失败${NC}"
    echo -e "${YELLOW}运行 'black .' 自动修复${NC}"
    exit 1
fi
echo ""

# 2. 导入排序检查
echo -e "${YELLOW}[2/5] 运行 isort 导入排序检查...${NC}"
if isort --check-only --diff .; then
    echo -e "${GREEN}✓ isort 检查通过${NC}"
else
    echo -e "${RED}✗ isort 检查失败${NC}"
    echo -e "${YELLOW}运行 'isort .' 自动修复${NC}"
    exit 1
fi
echo ""

# 3. 代码规范检查
echo -e "${YELLOW}[3/5] 运行 flake8 代码规范检查...${NC}"
# 严格检查：语法错误和未定义名称
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
    echo -e "${GREEN}✓ flake8 严格检查通过${NC}"
else
    echo -e "${RED}✗ flake8 发现严重问题${NC}"
    exit 1
fi

# 警告检查：其他代码质量问题
echo -e "${YELLOW}检查代码质量警告...${NC}"
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics --exclude=checkpoints,dataset,test_results
echo ""

# 4. 模型测试
echo -e "${YELLOW}[4/5] 运行模型导入和实例化测试...${NC}"
export PYTHONPATH="${PYTHONPATH}:$CODE_DIR"
if python test_models.py; then
    echo -e "${GREEN}✓ 模型测试通过${NC}"
else
    echo -e "${RED}✗ 模型测试失败${NC}"
    exit 1
fi
echo ""

# 5. 单元测试（如果存在）
echo -e "${YELLOW}[5/5] 运行单元测试...${NC}"
if [ -d "tests" ]; then
    if pytest tests/ -v --tb=short; then
        echo -e "${GREEN}✓ 单元测试通过${NC}"
    else
        echo -e "${RED}✗ 单元测试失败${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}未找到 tests 目录，跳过 pytest${NC}"
fi
echo ""

# 总结
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有本地 CI 检查通过！ ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}可以安全提交代码${NC}"
echo ""
