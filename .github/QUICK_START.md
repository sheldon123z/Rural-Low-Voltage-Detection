# CI/CD 快速开始指南

## 1. 安装开发依赖

```bash
# 安装代码质量工具
pip install black isort flake8 pytest pytest-cov

# 或者安装项目的完整开发依赖
pip install -e ".[dev]"
```

## 2. 格式化代码（首次设置）

在首次启用 CI/CD 前，建议格式化整个代码库：

```bash
cd code/voltage_anomaly_detection

# 自动格式化所有 Python 文件
black .

# 自动排序所有导入语句
isort .

# 提交格式化后的代码
git add .
git commit -m "style: 应用 Black 和 isort 格式化"
```

## 3. 使用 Pre-commit Hooks（推荐）

Pre-commit hooks 可以在每次提交前自动运行检查：

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
cd /path/to/Rural-Low-Voltage-Detection
pre-commit install

# 现在每次 git commit 都会自动运行检查
```

手动运行 pre-commit 检查所有文件：
```bash
pre-commit run --all-files
```

## 4. 本地运行 CI 检查

使用提供的脚本在提交前运行完整的 CI 检查：

```bash
# 从项目根目录运行
./.github/scripts/run-local-ci.sh
```

或手动运行各个检查：

```bash
cd code/voltage_anomaly_detection

# 代码格式化检查
black --check --diff .

# 导入排序检查
isort --check-only --diff .

# 代码规范检查
flake8 .

# 模型测试
python test_models.py

# 单元测试（如果有）
pytest tests/ -v
```

## 5. 修复常见问题

### Black 格式化问题
```bash
# 查看需要修改的地方
black --check --diff code/voltage_anomaly_detection/

# 自动修复
black code/voltage_anomaly_detection/
```

### isort 导入顺序问题
```bash
# 查看需要修改的地方
isort --check-only --diff code/voltage_anomaly_detection/

# 自动修复
isort code/voltage_anomaly_detection/
```

### flake8 代码规范问题
```bash
# 查看所有问题
flake8 code/voltage_anomaly_detection/

# 大多数问题需要手动修复
# 常见问题：
# - E501: 行太长（超过 88 字符）
# - F401: 导入但未使用
# - E302: 函数定义前需要两个空行
```

## 6. IDE 集成

### VS Code

安装扩展：
- Python (Microsoft)
- Black Formatter
- isort

在 `.vscode/settings.json` 中配置：
```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. Settings → Tools → Black
   - 勾选 "On save"
2. Settings → Tools → isort
   - 勾选 "On save"
3. Settings → Editor → Inspections
   - 启用 Flake8

## 7. 工作流触发条件

### 自动触发
- 推送到 `main`, `develop`, `copilot/**` 分支
- 针对 `main`, `develop` 的 Pull Request
- 每周一自动运行安全扫描

### 手动触发
访问 GitHub Actions 页面，选择 Security 工作流，点击 "Run workflow"

## 8. 查看 CI 结果

1. 访问 GitHub 仓库页面
2. 点击 "Actions" 标签
3. 查看最近的工作流运行
4. 点击具体的运行查看详细日志

## 9. CI 徽章

在 README.md 中已添加 CI 状态徽章，显示各工作流的运行状态。

## 10. 常见问题

### Q: CI 检查失败但本地通过？
A: 确保本地安装了相同版本的工具，并且在正确的目录运行命令。

### Q: 如何跳过 CI 检查？
A: 不建议跳过。如果确实需要，在提交信息中添加 `[skip ci]`。

### Q: Black 和 isort 冲突？
A: 已配置 isort 使用 "black" profile，应该不会冲突。

### Q: 工作流运行太慢？
A: 工作流使用了 pip cache 来加速。首次运行会慢一些。

### Q: 如何只运行特定的工作流？
A: 可以修改 workflow 文件的 `paths` 过滤器来限制触发条件。

## 11. 下一步

- 添加更多单元测试到 `tests/` 目录
- 配置代码覆盖率报告
- 添加性能基准测试
- 集成文档自动生成

## 12. 获取帮助

- 查看 [CI/CD 工作流文档](.github/CI_CD_GUIDE.md)
- 查看 GitHub Actions 日志
- 在项目 Issue 中提问
