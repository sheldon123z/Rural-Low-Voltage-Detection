# Scientific Writer 技能参考

## 如何使用本文件

本文件提供所有可用技能的快速索引。详细说明请查看 `.claude/skills/<skill-name>/SKILL.md`。

## 核心技能（写作必备）

### 1. research-lookup
**用途**：实时文献检索，查找真实学术论文
- 自动搜索 Google Scholar、arXiv、PubMed
- 验证引用真实性
- 生成 BibTeX 条目
- **关键技能：所有写作必须先使用此技能**

### 2. peer-review
**用途**：系统性论文评审
- 按学术标准评估论文质量
- 提供改进建议
- 自动化同行评议流程

### 3. citation-management
**用途**：BibTeX 和参考文献管理
- 管理引用数据库
- 自动格式化引用
- 检查引用完整性

## 专业领域技能

### 4. clinical-reports
**用途**：临床医学文档标准
- 病例报告
- 试验文档
- 临床研究论文

### 5. research-grants
**用途**：研究资助提案
- NSF、NIH、DOE 提案支持
- 资助申请书写作
- 预算编制

### 6. literature-review
**用途**：系统文献综述
- 系统性综述
- 元分析支持
- 文献搜索策略

## 可视化技能

### 7. scientific-slides
**用途**：学术研究演示文稿
- Nano Banana Pro AI 生成的 PDF 幻灯片
- 自动化幻灯片创建
- 学术演讲支持

### 8. latex-posters
**用途**：会议海报生成（LaTeX）
- LaTeX 格式海报
- AI 生成的可视化
- 学术会议海报

### 9. pptx-posters
**用途**：会议海报生成（PPTX）
- PowerPoint 格式海报
- 仅在明确要求 PPTX 时使用

### 10. scientific-schematics
**用途**：科学原理图生成
- 学术图表
- 原理图
- 流程图

### 11. generate-image
**用途**：信息图表和概念插图
- 信息可视化
- 概念艺术
- 示意图

## 写作辅助技能

### 12. hypothesis-generation
**用途**：假设生成文档
- 竞争性假设生成
- 彩色框 LaTeX 模板
- 可测试预测
- **特殊文档类型检测**

### 13. market-research-reports
**用途**：市场研究报告
- 行业分析
- 竞争格局
- 市场规模分析
- **特殊文档类型检测（50+ 页，25-30 视觉图）**

### 14. treatment-plans
**用途**：治疗计划
- 专业医疗格式
- 个性化治疗方案
- 临床决策支持

### 15. grant-writing
**用途**：通用资助写作
- 研究提案
- 项目申请
- 资金申请

### 16. data-visualization
**用途**：数据可视化
- 科研图表
- 数据分析图
- 统计图形

### 17. academic-writing
**用途**：学术论文写作
- 期刊论文
- 会议论文
- 学位论文章节

### 18. technical-writing
**用途**：技术文档写作
- 用户手册
- 技术规范
- API 文档

### 19. scientific-proposal
**用途**：科学提案
- 研究计划书
- 项目提案
- 合作协议

## 如何使用技能

### 在提示词中引用技能
```
"使用 @research-lookup 技能查找关于 'transformer attention mechanisms' 的论文"
"使用 @latex-posters 技能生成会议海报"
"使用 @peer-review 技能评估这篇论文"
```

### 自动检测
某些技能会自动触发：
- "hypothesis generation" → 自动使用 `hypothesis-generation`
- "market research" → 自动使用 `market-research-reports`

### 详细文档
每个技能的完整文档位于：
```
.claude/skills/<skill-name>/SKILL.md
```

## 快速参考

| 技能 | 主要用途 | 输出格式 |
|------|---------|---------|
| research-lookup | 文献检索 | BibTeX + PDF 链接 |
| peer-review | 论文评审 | HTML/Markdown 报告 |
| citation-management | 引用管理 | BibTeX 文件 |
| literature-review | 文献综述 | LaTeX 论文 |
| scientific-slides | 演示文稿 | PDF 幻灯片 |
| latex-posters | 会议海报 | LaTeX PDF |
| hypothesis-generation | 假设生成 | LaTeX 彩色框文档 |
| market-research-reports | 市场分析 | LaTeX 报告（50+ 页） |

## 更多信息

- 技能目录：`.claude/skills/`
- 完整文档：https://github.com/K-Dense-AI/claude-scientific-writer
- 写作指南：`WRITING_GUIDE.md`
