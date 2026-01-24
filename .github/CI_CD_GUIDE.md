# CI/CD 工作流文档

本项目实施了完整的 CI/CD 流程，确保代码质量、测试覆盖和安全性。

## 工作流概览

### 1. CI (持续集成) - `ci.yml`
**触发条件：** 
- 推送到 `main`, `develop`, `copilot/**` 分支
- 针对 `main`, `develop` 的 Pull Request

**检查内容：**
- ✅ 代码格式检查 (Black)
- ✅ 导入排序检查 (isort)
- ✅ 代码规范检查 (flake8)
- ✅ 模型导入和实例化测试
- ✅ 单元测试（如果存在）

**使用说明：**
```bash
# 本地运行相同检查
cd code/voltage_anomaly_detection

# 格式化代码
black .
isort .

# 检查代码规范
flake8 .

# 运行测试
python test_models.py
```

### 2. Code Quality (代码质量) - `code-quality.yml`
**触发条件：** Python 文件变更时

**检查内容：**
- Black 代码格式化标准
- isort 导入排序标准
- flake8 代码质量和复杂度

**修复方法：**
```bash
# 自动修复格式问题
black code/voltage_anomaly_detection/
isort code/voltage_anomaly_detection/

# 查看 flake8 问题
flake8 code/voltage_anomaly_detection/ --statistics
```

### 3. Testing (测试) - `testing.yml`
**触发条件：** 代码或依赖变更时

**测试环境：** Python 3.9, 3.10, 3.11

**测试内容：**
- 所有 15 个模型的导入和实例化
- 前向传播测试
- 数据生成脚本验证
- pytest 单元测试（如果存在）

**本地测试：**
```bash
cd code/voltage_anomaly_detection

# 运行模型测试
python test_models.py

# 运行 pytest（如果有 tests/ 目录）
pytest tests/ -v

# 测试数据生成
cd dataset/RuralVoltage
python generate_sample_data.py --train_samples 100 --test_samples 20
```

### 4. Security (安全扫描) - `security.yml`
**触发条件：**
- 依赖文件变更
- 每周一自动运行
- 手动触发

**扫描内容：**
- 依赖包漏洞扫描 (safety)
- 依赖审计 (pip-audit)

**修复漏洞：**
```bash
# 安装安全工具
pip install safety pip-audit

# 检查已安装包
safety check
pip-audit

# 更新有漏洞的包
pip install --upgrade <package-name>
```

### 5. Documentation (文档检查) - `documentation.yml`
**触发条件：** Markdown 或论文文件变更时

**检查内容：**
- Markdown 格式规范
- 文档链接有效性
- LaTeX 论文结构

### 6. PR Validation (PR 验证) - `pr-validation.yml`
**触发条件：** Pull Request 打开、同步或重新打开时

**验证内容：**
- PR 信息展示
- 提交信息格式
- 文件变更统计
- PR 大小警告（>500 行）

## 工作流状态徽章

在 README.md 中添加徽章显示 CI 状态：

```markdown
[![CI](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/workflows/CI/badge.svg)](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/actions/workflows/ci.yml)
[![Code Quality](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/workflows/Code%20Quality/badge.svg)](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/actions/workflows/code-quality.yml)
[![Testing](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/workflows/Testing/badge.svg)](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/actions/workflows/testing.yml)
[![Security](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/workflows/Security/badge.svg)](https://github.com/sheldon123z/Rural-Low-Voltage-Detection/actions/workflows/security.yml)
```

## 开发工作流

### 提交代码前
```bash
# 1. 格式化代码
black code/voltage_anomaly_detection/
isort code/voltage_anomaly_detection/

# 2. 检查代码质量
flake8 code/voltage_anomaly_detection/

# 3. 运行测试
cd code/voltage_anomaly_detection
python test_models.py

# 4. 提交代码
git add .
git commit -m "feat: 添加新功能"
git push
```

### Pull Request 流程
1. 创建功能分支：`git checkout -b feature/your-feature`
2. 进行开发和测试
3. 确保所有 CI 检查通过
4. 创建 Pull Request
5. 等待代码审查
6. 合并到主分支

## CI/CD 最佳实践

### 1. 保持 CI 快速
- 缓存依赖包 (pip cache)
- 并行运行独立的工作流
- 仅在相关文件变更时触发

### 2. 及时修复失败
- CI 失败应该立即修复
- 不要在 CI 失败的基础上继续开发
- 使用 `continue-on-error` 处理非关键检查

### 3. 本地优先
- 在提交前本地运行所有检查
- 使用 pre-commit hooks（可选）
- 配置 IDE 集成工具

### 4. 安全意识
- 定期更新依赖
- 关注安全扫描结果
- 不要在代码中硬编码密钥

## 配置文件

### pyproject.toml
包含所有 Python 工具的配置：
- Black: 代码格式化
- isort: 导入排序
- pytest: 测试配置
- flake8: 代码检查

### .gitignore
确保不提交：
- `__pycache__/`, `*.pyc`
- `checkpoints/`, `test_results/`
- `*.pth`, `*.pt`, `*.ckpt`
- `venv/`, `.env`

## 故障排查

### CI 失败常见原因

1. **Black 格式化失败**
   ```bash
   black --check --diff code/voltage_anomaly_detection/
   # 修复：
   black code/voltage_anomaly_detection/
   ```

2. **isort 导入顺序错误**
   ```bash
   isort --check-only --diff code/voltage_anomaly_detection/
   # 修复：
   isort code/voltage_anomaly_detection/
   ```

3. **flake8 代码规范问题**
   ```bash
   flake8 code/voltage_anomaly_detection/
   # 逐个修复报告的问题
   ```

4. **模型测试失败**
   ```bash
   cd code/voltage_anomaly_detection
   python test_models.py
   # 检查模型定义和依赖
   ```

5. **依赖安装失败**
   ```bash
   pip install -r code/voltage_anomaly_detection/requirements.txt
   # 检查 requirements.txt 是否正确
   ```

## 工作流文件位置

```
.github/
├── workflows/
│   ├── ci.yml                    # 主 CI 流程
│   ├── code-quality.yml          # 代码质量检查
│   ├── testing.yml               # 测试流程
│   ├── security.yml              # 安全扫描
│   ├── documentation.yml         # 文档检查
│   └── pr-validation.yml         # PR 验证
└── markdown-link-check-config.json  # Markdown 链接检查配置
```

## 联系和支持

如有 CI/CD 相关问题，请：
1. 查看 GitHub Actions 日志
2. 参考本文档的故障排查部分
3. 在项目 Issue 中提问

## 更新日志

- 2026-01: 初始 CI/CD 工作流实施
  - 添加代码质量检查
  - 添加自动化测试
  - 添加安全扫描
  - 添加文档验证
  - 添加 PR 验证
