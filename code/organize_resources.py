#!/usr/bin/env python3
"""
资料整理工具 - 将收集的资料按主题分类
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class ResourceOrganizer:
    def __init__(self, source_dir="collected_resources", target_dir="organized_resources"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)

        # 创建分类目录
        self.categories = {
            '01_政策文件与标准': self.target_dir / '01_政策文件与标准',
            '02_学术论文_深度学习': self.target_dir / '02_学术论文_深度学习',
            '03_技术报告': self.target_dir / '03_技术报告',
            '04_行业资讯': self.target_dir / '04_行业资讯',
        }

        for cat_path in self.categories.values():
            cat_path.mkdir(exist_ok=True)

    def organize(self):
        """整理资料"""
        print("开始整理资料...\n")

        # 定义文件分类映射
        file_mapping = {
            # 政策文件与标准
            '国网龙泉市供电公司乡村振兴五年行动方案.pdf': '01_政策文件与标准',
            '2024年能源领域行业标准制定计划.pdf': '01_政策文件与标准',
            '关于实施农村电网巩固提升工程的指导意见.md': '01_政策文件与标准',

            # 学术论文
            'Short-Term_Electricity-Load_Forecasting_Deep_Learning_Survey.md': '02_学术论文_深度学习',
            'Enhanced_LSTM_robotic_agent_load_forecasting.md': '02_学术论文_深度学习',
            'Enhanced_Load_Forecasting_GAT-LSTM.md': '02_学术论文_深度学习',
            'RNN-BiLSTM-CRF_electricity_theft_detection.md': '02_学术论文_深度学习',
            '基于深度学习技术的电表大数据检测系统.md': '02_学术论文_深度学习',
            '人工智能与机器人研究2022.pdf': '02_学术论文_深度学习',

            # 技术报告
            '中国分布式光伏韧性发展路径2026-2027展望报告.pdf': '03_技术报告',
            '电网专题研究报告2025.pdf': '03_技术报告',

            # 行业资讯
            '数千亿元电网投资勾勒能源变革新版图.md': '04_行业资讯',
            '山西长治政企联手开展分布式光伏电压越限治理.md': '04_行业资讯',
        }

        # 复制文件到对应分类
        pdf_dir = self.source_dir / 'pdfs'
        markdown_dir = self.source_dir / 'markdown'

        for filename, category in file_mapping.items():
            # 查找源文件
            source_file = None
            if (pdf_dir / filename).exists():
                source_file = pdf_dir / filename
            elif (markdown_dir / filename).exists():
                source_file = markdown_dir / filename

            if source_file and source_file.exists():
                target_file = self.categories[category] / filename
                shutil.copy2(source_file, target_file)
                print(f"✓ {filename} → {category}")
            else:
                print(f"✗ 未找到文件: {filename}")

        print("\n资料整理完成!")

    def generate_summary(self):
        """生成资料汇总文档"""
        summary_content = f"""# 农网低电压问题研究资料汇总

**整理日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、研究背景

农村电网低电压问题是影响农村用电质量的重要因素。随着农村用电需求增长和分布式能源接入，低电压问题愈发突出。本资料库汇集了最新的政策文件、学术研究、技术报告和行业资讯，为深入研究提供全面支撑。

---

## 二、资料分类

### 📋 1. 政策文件与标准

该分类收录国家和地方关于农村电网改造、低电压治理的政策文件和行业标准。

**核心文件**:
- **国网龙泉市供电公司乡村振兴五年行动方案** (PDF)
  - 内容: 地方供电公司在乡村振兴战略下的电网建设规划
  - 重点: 低电压治理目标、技术路线、投资计划

- **2024年能源领域行业标准制定计划** (PDF)
  - 内容: 国家能源局发布的2024年度标准制定计划
  - 重点: 配电网相关标准、智能电网技术规范

- **关于实施农村电网巩固提升工程的指导意见** (Markdown)
  - 来源: 国家发展和改革委员会
  - 内容: 农村电网建设的顶层设计文件
  - 重点: 低电压治理工程要求、供电质量指标

**应用价值**:
- 了解政策导向和治理目标
- 把握技术标准和规范要求
- 明确项目实施的政策依据

---

### 🔬 2. 学术论文 - 深度学习方向

该分类汇集了应用深度学习技术进行电力系统异常检测、负荷预测的最新研究成果。

**核心论文**:

1. **Short-Term Electricity-Load Forecasting: Deep Learning Survey** (Markdown)
   - 类型: 综述论文 (arXiv)
   - 内容: 短期电力负荷预测的深度学习方法全面综述
   - 技术: LSTM, GRU, Transformer, CNN, Hybrid Models
   - 价值: ⭐⭐⭐ 必读综述，了解领域全貌

2. **Enhanced LSTM-based Robotic Agent for Load Forecasting** (Markdown)
   - 来源: Frontiers in Neurorobotics (2024)
   - 应用场景: 低压分布式光伏配电网
   - 技术创新: 增强LSTM + 机器人代理
   - 价值: ⭐⭐⭐ 与农网场景高度相关

3. **Enhanced Load Forecasting with GAT-LSTM** (Markdown)
   - 来源: arXiv (2025)
   - 核心技术: 图注意力网络 (GAT) + LSTM
   - 创新点: 结合电网拓扑空间特征和时序特征
   - 价值: ⭐⭐⭐ 方法创新性强

4. **RNN-BiLSTM-CRF Electricity Theft Detection** (Markdown)
   - 来源: PMC (2024)
   - 应用: 智能电网窃电检测
   - 技术: 双向LSTM + 条件随机场
   - 价值: ⭐⭐ 异常检测相关

5. **基于深度学习技术的电表大数据检测系统** (Markdown)
   - 来源: 汉斯出版社 (中文)
   - 内容: 智能电表异常检测系统设计
   - 技术: 大数据分析 + 深度学习
   - 价值: ⭐⭐ 系统设计参考

6. **人工智能与机器人研究2022** (PDF)
   - 类型: 期刊论文集
   - 内容: AI技术在多个领域的应用研究
   - 价值: ⭐ 扩展阅读

**研究热点**:
- LSTM/GRU用于时序预测
- GNN处理电网拓扑结构
- Transformer处理长序列依赖
- 混合模型结合多种技术优势

**技术路线参考**:
```
数据采集 → 特征工程 → 模型训练 → 异常检测/负荷预测 → 实时监控
   ↓           ↓           ↓              ↓                ↓
智能电表    时序特征    LSTM/GNN      阈值判定        可视化系统
          拓扑特征    Transformer    统计分析        预警机制
```

---

### 📊 3. 技术报告

该分类包含权威机构发布的行业技术研究报告和发展趋势分析。

**核心报告**:

1. **中国分布式光伏韧性发展路径: 2026与2027年展望报告** (PDF)
   - 发布机构: RMI (落基山研究所)
   - 发布时间: 2025年12月
   - 核心内容:
     - 分布式光伏接入对配电网的影响
     - 电压越限问题分析与治理方案
     - 2026-2027年发展趋势预测
   - 价值: ⭐⭐⭐ 行业权威报告，前瞻性强

2. **电网专题研究报告2025** (PDF)
   - 来源: 证券研究报告
   - 核心内容:
     - 电网投资规模与方向
     - 配电网智能化升级
     - 新能源并网技术挑战
   - 价值: ⭐⭐ 投资视角，了解行业动态

**关键发现**:
- 分布式光伏渗透率提高导致电压波动加剧
- 需要智能调控技术应对电压越限
- 边缘计算 + AI是未来发展方向

---

### 📰 4. 行业资讯

该分类收录最新的行业新闻和实践案例。

**核心资讯**:

1. **数千亿元电网投资勾勒能源变革新版图** (Markdown)
   - 来源: 证券时报网
   - 内容: 国家电网"十五五"规划分析
   - 关键数据:
     - 电网投资规模持续增长
     - 配电网智能化成为重点
     - 农村电网改造升级加速

2. **山西长治: 政企联手开展分布式光伏电压越限治理** (Markdown)
   - 来源: 山西省国资委
   - 实践案例: 政企合作治理电压越限
   - 技术方案:
     - 电能质量治理APP系统
     - 数据采集 - 智能决策 - 指令下发 - 效果反馈
     - 自动调压技术应用

**实践启示**:
- 低电压治理需要政企协同
- 数字化监控系统是关键
- 实时调控能力显著提升治理效果

---

## 三、研究方向建议

### 3.1 核心研究问题

1. **农村低压电网特殊性**
   - 线路长、损耗大、负荷波动剧烈
   - 拓扑结构复杂、分布式电源接入
   - 监测数据稀疏、通信条件受限

2. **低电压异常检测**
   - 实时监测与预警
   - 多源数据融合
   - 边缘计算部署

3. **电压质量预测**
   - 短期负荷预测
   - 电压趋势预测
   - 异常模式识别

### 3.2 技术方案设计

**推荐技术栈**:
- **数据采集**: 智能电表 + 边缘网关
- **特征工程**: 时序特征 + 拓扑特征 + 气象特征
- **模型架构**: GAT-LSTM混合模型
  - GAT处理电网拓扑空间关系
  - LSTM捕获时序动态特性
- **部署方式**: 边缘计算 + 云端协同
- **预警机制**: 多级阈值 + 趋势分析

### 3.3 创新点

1. **方法创新**
   - 针对农网特点改进现有深度学习模型
   - 引入注意力机制增强关键特征识别
   - 设计适应稀疏数据的学习策略

2. **系统创新**
   - 边缘智能 + 云端大脑的分布式架构
   - 轻量化模型满足边缘设备算力限制
   - 在线学习能力适应负荷变化

3. **应用创新**
   - 多场景适应性（农业、居民、工业混合负荷）
   - 与配网自动化系统集成
   - 提供治理决策支持

---

## 四、后续工作建议

### 阶段一: 文献精读与方法选型 (2周)
- [ ] 精读⭐⭐⭐级别论文，整理技术要点
- [ ] 对比不同方法的优缺点
- [ ] 确定论文采用的技术路线

### 阶段二: 数据准备与预处理 (2周)
- [ ] 收集或生成农网低电压数据
- [ ] 数据清洗和特征提取
- [ ] 划分训练集、验证集、测试集

### 阶段三: 模型开发与实验 (4-6周)
- [ ] 实现基线模型 (LSTM)
- [ ] 开发改进模型 (GAT-LSTM)
- [ ] 对比实验与参数调优
- [ ] 消融实验验证各模块有效性

### 阶段四: 系统验证与论文撰写 (4周)
- [ ] 在真实/仿真数据上验证
- [ ] 分析不同场景下的性能
- [ ] 撰写论文各章节
- [ ] 准备实验图表和结果分析

---

## 五、关键术语索引

| 中文 | 英文 | 缩写 |
|-----|------|-----|
| 农村电网 | Rural Power Grid | - |
| 低电压 | Low Voltage | LV |
| 异常检测 | Anomaly Detection | AD |
| 长短期记忆网络 | Long Short-Term Memory | LSTM |
| 图注意力网络 | Graph Attention Network | GAT |
| 图神经网络 | Graph Neural Network | GNN |
| 智能电表 | Smart Meter | SM |
| 配电网 | Distribution Network | DN |
| 分布式光伏 | Distributed Photovoltaic | DPV |
| 电能质量 | Power Quality | PQ |
| 边缘计算 | Edge Computing | - |

---

## 六、参考资源

### 在线数据集
- **FiN Dataset**: 智能电网和电力线通信数据
- **Open Energy Data**: 公开电力数据平台

### 开发工具
- **深度学习框架**: PyTorch, TensorFlow
- **图处理库**: PyTorch Geometric, DGL
- **电力系统仿真**: pandapower, GridLAB-D
- **数据处理**: pandas, numpy, scipy

### 论文数据库
- Google Scholar
- IEEE Xplore
- arXiv
- 中国知网 (CNKI)

---

## 七、联系与更新

本资料库将持续更新，如有新的研究进展或相关资料，请及时补充。

**最后更新**: {datetime.now().strftime('%Y年%m月%d日')}

---

*本文档由自动整理工具生成*
"""

        summary_file = self.target_dir / "资料汇总与研究建议.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        print(f"\n✓ 资料汇总文档已生成: {summary_file}")

    def create_readme(self):
        """创建README文件"""
        readme_content = """# 农网低电压问题研究资料库

本目录包含整理后的研究资料，按主题分类。

## 目录结构

```
organized_resources/
├── 01_政策文件与标准/        # 国家政策、行业标准、指导意见
├── 02_学术论文_深度学习/     # 深度学习相关学术论文
├── 03_技术报告/              # 行业研究报告、发展趋势
├── 04_行业资讯/              # 最新行业动态、实践案例
└── 资料汇总与研究建议.md     # 全面的资料索引和研究指南
```

## 使用指南

1. **快速开始**: 先阅读 `资料汇总与研究建议.md`，了解全局
2. **政策导向**: 查看 `01_政策文件与标准/` 了解政策要求
3. **技术学习**: 阅读 `02_学术论文_深度学习/` 中的论文
4. **行业趋势**: 参考 `03_技术报告/` 了解发展方向
5. **实践案例**: 查看 `04_行业资讯/` 学习实际应用

## 推荐阅读顺序

### 初学者
1. 资料汇总与研究建议.md (整体认识)
2. 关于实施农村电网巩固提升工程的指导意见 (背景知识)
3. Short-Term Electricity-Load Forecasting Survey (技术综述)

### 研究者
1. GAT-LSTM论文 (最新方法)
2. LSTM Robotic Agent论文 (相关应用)
3. 技术报告 (行业趋势)

## 更新日志

- 2026-01-20: 初始资料整理完成
"""

        readme_file = self.target_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✓ README文件已创建: {readme_file}")


def main():
    organizer = ResourceOrganizer()

    # 整理资料
    organizer.organize()

    # 生成汇总文档
    organizer.generate_summary()

    # 创建README
    organizer.create_readme()

    print("\n" + "="*60)
    print("所有整理任务完成!")
    print(f"整理后的资料位置: {organizer.target_dir}")
    print(f"请查看 '资料汇总与研究建议.md' 获取详细信息")
    print("="*60)


if __name__ == "__main__":
    main()
