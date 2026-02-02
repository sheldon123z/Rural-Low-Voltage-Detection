# 农村低压配电网电压异常检测研究

基于深度学习的农村低压配电网电压异常检测研究。本项目采用 TimesNet 及其变体模型，结合清华 Time-Series-Library 框架，实现对电压异常的精准检测。

## 🔗 资源链接

| 资源 | 链接 | 说明 |
|------|------|------|
| **🤗 模型检查点** | [Sheldon123z/rural-voltage-detection-models](https://huggingface.co/Sheldon123z/rural-voltage-detection-models) | 35 个预训练模型 (687MB) |
| **🤗 数据集** | [Sheldon123z/rural-voltage-datasets](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets) | RuralVoltage + PSM + KagglePQ (153MB) |
| **📄 论文仓库** | [BJFU-thesis](https://github.com/sheldon123z/BJFU-thesis) | 北京林业大学毕业论文 |

## 项目亮点

- **5 种 TimesNet 变体模型**：针对电力系统电压信号特点设计的创新模型
- **15 种深度学习模型**：全面的模型对比基准，包括 Transformer、AutoEncoder、线性模型等
- **完整实验框架**：基于清华 Time-Series-Library 构建，支持 5 大类时序分析任务
- **农村电压数据集**：16 维特征，9 种异常类型（欠压、过压、电压暂降、谐波、三相不平衡、闪变、暂态、复合异常等）
- **学术写作支持**：集成 Scientific Writer，支持论文撰写、文献检索、引用管理
- **Hugging Face 集成**：模型和数据集已上传至 Hugging Face，方便下载和复现

## 项目结构

```
Rural-Low-Voltage-Detection/
├── code/                           # 核心代码（基于 Time-Series-Library）
│   ├── run.py                      # 主入口脚本
│   ├── models/                      # 19+ 模型实现
│   │   ├── TimesNet.py             # 基础 TimesNet（ICLR 2023）
│   │   ├── VoltageTimesNet.py        # 预设电网周期 + FFT 混合
│   │   ├── TPATimesNet.py            # 三相交叉注意力机制
│   │   ├── MTSTimesNet.py            # 多尺度时序并行
│   │   ├── HybridTimesNet.py           # 混合周期发现
│   │   ├── DLinear.py               # 趋势-季节性分解线性模型
│   │   ├── PatchTST.py              # 补丁级 Transformer
│   │   ├── iTransformer.py            # 倒置 Transformer
│   │   ├── Autoformer.py            # 自相关分解 Transformer
│   │   ├── Informer.py              # ProbSparse 自注意力
│   │   ├── Reformer.py              # 局部敏感哈希
│   │   └── FEDformer.py             # 频率增强分解
│   ├── exp/                         # 实验类
│   ├── data_provider/               # 数据加载器（PSM, MSL, SMAP, SMD, SWAT, RuralVoltage）
│   ├── layers/                      # 网络层组件
│   ├── utils/                       # 工具函数
│   ├── scripts/                     # 训练脚本（按数据集分类）
│   ├── dataset/                     # 数据集
│   ├── checkpoints/                  # 模型检查点
│   ├── results/                      # 实验结果
│   └── test_results/                # 测试结果
├── thesis/                          # 毕业论文（北林 BJFUThesis 模板）
│   ├── bjfuthesis-main.tex      # 论文主文件
│   ├── chap00.tex                 # 第一章 绪论
│   ├── chap01.tex                 # 第二章 数据采集与预处理
│   ├── chap02.tex                 # 第三章 基于 TimesNet 的电压异常检测算法
│   ├── chap03.tex                 # 第四章 实验设计与结果分析
│   ├── chap04.tex                 # 第五章 农村电网低电压监管平台设计
│   └── chap05.tex                 # 第六章 结论与展望
├── resources/                       # 研究资料
│   ├── papers/                    # 时序异常检测论文（9 篇核心论文）
│   ├── collected_resources/        # 已收集的资料（电力政策、研究报告）
│   └── pdfs/                   # PDF 文档
├── CLAUDE.md                       # Claude Code 指南（含 Scientific Writer 配置）
└── README.md                       # 项目说明
```

## TimesNet 模型家族

| 模型 | 核心创新 | 适用场景 | 对应论文 |
|------|---------|---------|----------|
| **TimesNet** | FFT 周期发现 + 2D 卷积 | 通用时序异常检测 | Wu et al., ICLR 2023 |
| **VoltageTimesNet** | 预设电网周期 + FFT 混合 | 电力系统电压监测 | 创新模型 |
| **VoltageTimesNet_v2** | 可学习权重 + 异常放大器 | 高召回率场景 | 创新模型 (新增) |
| **TPATimesNet** | 三相交叉注意力机制 | 三相不平衡异常检测 | 创新模型 |
| **MTSTimesNet** | 多尺度并行 + 自适应融合 | 复杂多尺度异常 | 创新模型 |
| **HybridTimesNet** | 置信度融合周期发现 | 鲁棒周期检测 | 创新模型 |

### 其他支持的深度学习模型

**Transformer 系列**：
- **Autoformer** - 分解 Transformer + Auto-Correlation（NeurIPS 2021）
- **Informer** - ProbSparse 自注意力（AAAI 2021）
- **Reformer** - 可逆残差层（ICLR 2020）
- **FEDformer** - 频率增强分解 Transformer（ICML 2022）
- **PatchTST** - 补丁级 Transformer（ICLR 2023）
- **iTransformer** - 倒置 Transformer（ICLR 2024）

**轻量级与线性模型**：
- **DLinear** - 趋势-季节性分解线性模型（arXiv 2022）
- **LightGTS** - 周期性 Tokenization 轻量级模型（ICML 2025）

## 数据集

### 标准数据集

| 数据集 | 特征数 | 训练集 | 测试集 | 应用领域 |
|--------|--------|--------|--------|----------|
| **PSM** | 25 | 132,481 | 87,841 | 服务器机器 |
| **MSL** | 55 | 58,317 | 73,729 | NASA 航天器 |
| **SMAP** | 25 | 135,183 | 427,617 | NASA 航天器 |
| **SMD** | 38 | 708,405 | 708,420 | 服务器机器 |
| **SWAT** | 51 | 495,000 | 449,919 | 安全水处理 |
| **KagglePQ** | 128 | 2,400 | 9,598 | 电力质量波形 (新增) |

### 农村电压数据集 (RuralVoltage)

| 特征类别 | 特征名称 | 说明 |
|---------|---------|------|
| **三相电压** | Va, Vb, Vc | 200-240V 范围 |
| **三相电流** | Ia, Ib, Ic | 10-20A 范围 |
| **功率指标** | P, Q, S, PF | 有功/无功/视在功率 |
| **电能质量** | THD_Va, THD_Vb, THD_Vc | 谐波失真率 (GB/T 12325-2008) |
| **不平衡因子** | V_unbalance, I_unbalance | 三相不平衡度 |
| **频率** | Freq | 50Hz 标准频率 |

**5 种异常类型**：
1. **Undervoltage (欠压)**: 电压低于 198V
2. **Overvoltage (过压)**: 电压高于 235V
3. **Voltage_Sag (电压骤降)**: 电压突然下降 10%+
4. **Harmonic (谐波畸变)**: THD 超过 5%
5. **Unbalance (三相不平衡)**: 不平衡度超过 2%

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 模型隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet top-k 周期数 |
| `--enc_in` | 25 / 17 | 输入特征数（PSM=25, RuralVoltage=17） |
| `--anomaly_ratio` | 1.0 | 预期异常比例 (%) |

## 快速开始

### 环境配置

```bash
# 创建并激活 conda 环境
conda create -n tslib python=3.11
conda activate tslib

# 安装依赖
pip install -r code/requirements.txt
```

### 训练模型

```bash
cd code

# 使用 TimesNet 在 PSM 数据集训练
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64

# 使用 VoltageTimesNet_v2 在农村电压数据集训练（推荐）
python run.py --is_training 1 --model VoltageTimesNet_v2 --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 --anomaly_ratio 3.0

# 使用 Kaggle 电力质量数据集
python run.py --is_training 1 --model TimesNet --data KagglePQ \
  --root_path ./dataset/Kaggle_PowerQuality_2/ --enc_in 128 --c_out 128

# 运行模型对比实验
bash scripts/PSM/run_comparison.sh
```

### 分析实验结果

```bash
cd code
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_XXXXXX
```

### 论文编译

```bash
cd thesis
xelatex bjfuthesis-main.tex
biber bjfuthesis-main
xelatex bjfuthesis-main.tex
xelatex bjfuthesis-main.tex
```

## 时序异常检测核心论文

本项目已下载并整理了 9 篇时序异常检测核心论文（共 24 MB），保存在 `resources/papers/` 目录：

### 已下载论文

| 论文 | 作者 | 会议 | 年份 | 文件 |
|------|------|--------|------|------|
| **TimesNet** | Haixu Wu 等 | ICLR 2023 | TimesNet_ICLR_2023_2023.pdf |
| **DLinear** | Mingze Zeng 等 | arXiv 2022 | DLinear_arXiv_2022.pdf |
| **PatchTST** | Yuqi Nie 等 | ICLR 2023 | PatchTST_ICLR_2023_2023.pdf |
| **iTransformer** | Yong Liu 等 | ICLR 2024 | iTransformer_ICLR_2024_2024.pdf |
| **Autoformer** | Haixu Wu 等 | NeurIPS 2021 | Autoformer_NeurIPS_2021_2021.pdf |
| **Informer** | Haoyi Zhou 等 | AAAI 2021 | Informer_AAAI_2021_2021.pdf |
| **Reformer** | Nikita Kitaev 等 | ICLR 2020 | Reformer_ICLR_2020_2020.pdf |
| **FEDformer** | Tian Zhou 等 | ICML 2022 | FEDformer_ICML_2022_2022.pdf |
| **LightGTS** | Yihang Wang 等 | ICML 2025 | LightGTS_ICML_2025_2025.pdf |

### 论文资源

- **索引文件**: [INDEX.md](resources/papers/INDEX.md) - 完整的中文介绍
- **BibTeX 文件**: [references.bib](resources/papers/references.bib) - 可直接引用
- **下载脚本**: [download_papers.py](resources/papers/download_papers.py) - 可重复下载

### BibTeX 引用示例

```bibtex
@inproceedings{timesnet2023,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2023},
  pdf={https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf},
  arxiv={2210.02186}
}
```

## 技术栈

- **深度学习框架**: PyTorch 2.x + CUDA
- **时序库**: Time-Series-Library (清华)
- **可视化**: Matplotlib, Seaborn, Plotly
- **论文模板**: BJFUThesis (北京林业大学)
- **论文写作**: Scientific Writer (集成在 CLAUDE.md)

## 论文结构

| 章节 | 文件 | 对应代码模块 |
|------|------|-------------|
| 第一章 绪论 | `chap00.tex` | - |
| 第二章 数据采集与预处理 | `chap01.tex` | `data_provider/`, `dataset/` |
| 第三章 基于 TimesNet 的电压异常检测算法 | `chap02.tex` | `models/`, `layers/` |
| 第四章 实验设计与结果分析 | `chap03.tex` | `run.py`, `scripts/`, `results/` |
| 第五章 农村电网低电压监管平台设计 | `chap04.tex` | 系统设计 |
| 第六章 结论与展望 | `chap05.tex` | - |

## 异常检测原理

采用**重构误差**方法进行异常检测：

1. **训练阶段**：模型仅在正常数据上训练，学习正常时序模式
2. **检测阶段**：计算测试数据的重构误差
3. **阈值判定**：基于训练集误差分布确定阈值
   ```python
   # 使用百分位数阈值
   threshold = np.percentile(train_energy, 100 - anomaly_ratio)
   pred = (test_energy > threshold).astype(int)
   ```
4. **点调整策略**：异常段内任何点被正确检测，则整段标记为正确

## 参考资源

### 已收集的资料

`resources/collected_resources/` 目录包含：
- 电力行业政策文件
- 电网研究报告
- 学术论文（已转换为 Markdown）
- 时序预测相关研究

### Time-Series-Library

- **GitHub**: https://github.com/thuml/time-series-library
- **说明**: 统一的深度时序模型开发框架，支持 40+ 种 SOTA 模型
- **任务**: 长期预测、短期预测、插补、分类、异常检测

## 详细文档

- **项目指南**: [CLAUDE.md](CLAUDE.md) - Claude Code 使用指南（含 Scientific Writer 配置）
- **代码文档**: [code/README.md](code/README.md) - 详细代码使用说明

## CI/CD 工作流

本项目配置了完整的 CI/CD 流程，包括：
- ✅ 代码质量检查（Black、isort、flake8）
- ✅ 自动化测试（模型测试、单元测试）
- ✅ 安全扫描（依赖漏洞检测）
- ✅ 文档验证（Markdown、链接检查）

详细说明请查看 [CI/CD 工作流文档](.github/CI_CD_GUIDE.md)。

## Star History

如果本项目对您有帮助，请点击 ⭐ Star 支持一下！
