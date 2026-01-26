# 科学论文写作指南

## 核心原则

### 1. 实证研究优先

- **所有引用必须是真实可验证的论文**
- 禁止使用占位符引用（如 "Smith et al. 2023" 除非已验证）
- 禁止使用示例性引用（用于演示的假引用）
- 禁止杜撰引用（不存在的论文）
- 使用 `research-lookup` 技能广泛检索真实文献

### 2. 写作流程

1. **规划阶段**
   - 分析文档类型（研究论文、综述、提案等）
   - 确定科学领域和主题
   - 默认使用 LaTeX 格式

2. **研究阶段（先于写作）**
   - 使用 `research-lookup` 技能检索 5-10 篇相关真实论文
   - 验证每篇论文的相关性和真实性
   - 记录所有发现

3. **写作阶段**
   - 基于真实论文进行写作
   - 仅使用已验证的引用
   - 如果需要更多引用，先执行更多研究

4. **验证阶段**
   - 自动 PDF 质量检查
   - 迭代改进直到视觉专业

### 3. 默认格式

- **LaTeX + BibTeX**（学术/科学出版标准）
- 除非用户明确要求其他格式
- 中文论文使用 XeLaTeX 编译

## 特殊文档类型

### 假设生成文档
- 关键词："hypothesis generation"、"generate hypotheses"
- 使用专用彩色框 LaTeX 模板
- 结构：4 页正文 + 详细附录
- 编译器：XeLaTeX

### 市场研究报告
- 关键词："market research"、"market analysis"
- 模板位置：`skills/market-research-reports/assets/market_report_template.tex`
- 最小 50 页，25-30 张可视化图表
- 编译器：XeLaTeX

### 其他类型
- **治疗计划**：`treatment-plans` 技能
- **临床报告**：`clinical-reports` 技能
- **科学海报**：`latex-posters` 技能（默认）或 `pptx-posters`（PPTX 明确要求时）
- **演示文稿**：`scientific-slides` 技能
- **文献综述**：`literature-review` 技能
- **研究资助**：`research-grants` 技能

## 工作流程

### 步骤 1：项目设置
```bash
# 创建唯一项目文件夹
writing_outputs/<timestamp>_<brief_description>/
```

### 步骤 2：文献检索
```python
# 使用 research-lookup 技能
# 检索关键词："topic" + "field" + specific terms
# 目标：每章 5-10 篇真实论文
```

### 步骤 3：编写文档
- 根据文档类型使用相应模板
- 整合所有真实引用
- 保持学术严谨性

### 步骤 4：质量检查
- 编译 LaTeX 文档
- 检查引用完整性
- 验证格式规范

## 可用技能参考

所有可用技能的详细说明请参见：
- `SKILLS_REFERENCE.md` - 19+ 个专业技能完整描述
- 按需查询技能名称即可获取使用方法

## 中文论文写作规范

### 图表规范
- 格式：仅 PNG（300 DPI）
- 标题：无标题（文件名即为标题）
- 字体：中文宋体/黑体
- 尺寸：宽度 8-10 英寸

### 引用格式
- 中文论文：GB/T 7714-2015 标准
- 英文论文：IEEE 或 APA（根据期刊要求）
- BibTeX 条目必须完整

### 中文翻译
- 避免过度翻译（保留英文术语）
- 技术名词首次出现时标注英文
- 保持学术语言的专业性

## 项目输出目录

所有写作成果保存在：
```
writing_outputs/
├── <timestamp>_<description>/
│   ├── main.tex              # 主文档
│   ├── references.bib        # 参考文献
│   ├── figures/              # 图表
│   ├── progress.md           # 进度日志
│   └── final/                # 最终 PDF
```
