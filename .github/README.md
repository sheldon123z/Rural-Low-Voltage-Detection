# CI/CD 文档索引

欢迎使用农村低压配电网异常检测项目的 CI/CD 系统。本文档提供完整的导航指南。

## 📚 文档结构

### 1. 快速开始
**文件**: [QUICK_START.md](QUICK_START.md)

适合：首次使用 CI/CD 的开发者

内容：
- ✅ 环境安装指南
- ✅ 代码格式化步骤
- ✅ Pre-commit hooks 设置
- ✅ 本地 CI 测试
- ✅ IDE 集成配置
- ✅ 常见问题解答

**阅读时间**: 10-15 分钟

---

### 2. 使用指南
**文件**: [CI_CD_GUIDE.md](CI_CD_GUIDE.md)

适合：日常开发使用

内容：
- 📋 工作流概览（6个工作流详解）
- 🔧 本地运行指南
- 🏷️ 状态徽章说明
- 📝 开发工作流最佳实践
- 🐛 故障排查指南
- 📁 配置文件说明

**阅读时间**: 20-30 分钟

---

### 3. 架构文档
**文件**: [CI_CD_ARCHITECTURE.md](CI_CD_ARCHITECTURE.md)

适合：了解系统设计、维护 CI/CD

内容：
- 🏗️ 架构图和数据流
- 🔄 工作流层次结构
- 📊 检查矩阵
- ⚡ 性能优化策略
- 🔐 安全考虑
- 💰 成本分析

**阅读时间**: 20-25 分钟

---

### 4. 实施总结
**文件**: [CI_CD_IMPLEMENTATION_SUMMARY.md](CI_CD_IMPLEMENTATION_SUMMARY.md)

适合：项目管理、技术总结

内容：
- ✅ 已完成工作清单
- 🎯 设计特点
- 📈 使用统计
- 🚀 后续建议
- 🔧 技术栈
- 📚 参考资源

**阅读时间**: 10 分钟

---

## 🎯 按角色查看

### 新手开发者
1. 阅读 [QUICK_START.md](QUICK_START.md)
2. 运行 `.github/scripts/run-local-ci.sh`
3. 参考 [CI_CD_GUIDE.md](CI_CD_GUIDE.md) 的"开发工作流"章节

### 日常开发者
1. 使用 [CI_CD_GUIDE.md](CI_CD_GUIDE.md) 作为参考
2. 查看"故障排查"章节解决问题
3. 关注 GitHub Actions 页面的 CI 状态

### 维护者/管理员
1. 理解 [CI_CD_ARCHITECTURE.md](CI_CD_ARCHITECTURE.md)
2. 查看 [CI_CD_IMPLEMENTATION_SUMMARY.md](CI_CD_IMPLEMENTATION_SUMMARY.md)
3. 定期审查安全扫描结果

---

## 🔧 工作流文件

所有工作流配置文件位于 `.github/workflows/` 目录：

| 文件 | 说明 | 触发时机 |
|------|------|----------|
| `ci.yml` | 主 CI 流程 | 每次推送/PR |
| `code-quality.yml` | 代码质量检查 | Python 文件变更 |
| `testing.yml` | 测试执行（多版本） | 代码/依赖变更 |
| `security.yml` | 安全扫描 | 依赖变更/每周一 |
| `documentation.yml` | 文档检查 | Markdown 变更 |
| `pr-validation.yml` | PR 信息验证 | PR 打开/更新 |

---

## 🛠️ 配置文件

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | Python 工具配置（Black, isort, pytest, flake8） |
| `.pre-commit-config.yaml` | Pre-commit hooks 配置 |
| `markdown-link-check-config.json` | Markdown 链接检查配置 |

---

## 📜 脚本工具

| 脚本 | 说明 | 使用方法 |
|------|------|----------|
| `.github/scripts/run-local-ci.sh` | 本地 CI 测试 | `./github/scripts/run-local-ci.sh` |

---

## 📖 使用场景

### 场景 1：我想在提交前检查代码
```bash
# 方法 1：使用脚本
./.github/scripts/run-local-ci.sh

# 方法 2：手动检查
cd code/voltage_anomaly_detection
black --check .
isort --check-only .
flake8 .
python test_models.py
```

参考：[QUICK_START.md](QUICK_START.md) 第 4 节

---

### 场景 2：CI 失败了，如何修复？
1. 查看 GitHub Actions 的错误日志
2. 参考 [CI_CD_GUIDE.md](CI_CD_GUIDE.md) 的"故障排查"章节
3. 根据错误类型执行相应的修复命令

常见修复：
- Black: `black code/voltage_anomaly_detection/`
- isort: `isort code/voltage_anomaly_detection/`
- flake8: 根据具体错误手动修复

---

### 场景 3：如何添加新的检查？
1. 阅读 [CI_CD_ARCHITECTURE.md](CI_CD_ARCHITECTURE.md) 的"扩展性设计"
2. 在相应的工作流文件中添加步骤
3. 更新相关文档
4. 测试新的工作流

---

### 场景 4：如何理解 CI/CD 架构？
阅读顺序：
1. [CI_CD_IMPLEMENTATION_SUMMARY.md](CI_CD_IMPLEMENTATION_SUMMARY.md) - 了解整体
2. [CI_CD_ARCHITECTURE.md](CI_CD_ARCHITECTURE.md) - 理解设计
3. 查看 `.github/workflows/*.yml` - 查看实现

---

## 🎓 学习路径

### 初级（1-2 小时）
- [ ] 阅读 QUICK_START.md
- [ ] 安装开发工具
- [ ] 运行本地 CI 检查
- [ ] 理解基本工作流

### 中级（3-4 小时）
- [ ] 深入阅读 CI_CD_GUIDE.md
- [ ] 学习故障排查
- [ ] 配置 IDE 集成
- [ ] 设置 pre-commit hooks

### 高级（5+ 小时）
- [ ] 研究 CI_CD_ARCHITECTURE.md
- [ ] 理解工作流设计
- [ ] 学习性能优化
- [ ] 贡献工作流改进

---

## 📞 获取帮助

### 遇到问题？
1. **查看文档**: 先检查相关文档是否有答案
2. **搜索 Issues**: GitHub Issues 中可能有类似问题
3. **提出问题**: 在 Issues 中详细描述问题
4. **查看日志**: GitHub Actions 日志包含详细信息

### 改进建议？
- 提交 Pull Request
- 在 Issues 中讨论
- 联系维护团队

---

## 📊 快速链接

- **GitHub Actions**: [查看运行历史](../../actions)
- **主仓库**: [Rural-Low-Voltage-Detection](../..)
- **代码目录**: [voltage_anomaly_detection](../../code/voltage_anomaly_detection)
- **README**: [项目主页](../../README.md)

---

## 🔄 更新记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-01-24 | 1.0.0 | 初始 CI/CD 系统实施 |

---

**维护者**: 开发团队
**最后更新**: 2026-01-24
**反馈**: 通过 GitHub Issues
