# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

农村低压配电网电压异常检测研究生毕业论文项目。核心代码位于 `code/`，基于清华大学 Time-Series-Library 框架构建，支持 19 种深度学习模型进行时序异常检测。

## 常用命令

### 环境配置
```bash
conda create -n tslib python=3.11
conda activate tslib
pip install -r code/requirements.txt
```

### 训练模型
```bash
cd code

# 使用 TimesNet 在 PSM 数据集训练
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 \
  --train_epochs 10 --batch_size 32

# 使用定制 VoltageTimesNet 在农村电压数据集训练
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ --enc_in 16 --c_out 16 \
  --seq_len 100 --d_model 64 --d_ff 128 --top_k 5

# 仅测试（加载已有检查点）
python run.py --is_training 0 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/
```

### 生成样本数据
```bash
cd code/dataset/RuralVoltage
python generate_sample_data.py --train_samples 10000 --test_samples 2000 --anomaly_ratio 0.1
```

### 使用训练脚本
```bash
cd code

# PSM 数据集多模型对比实验
bash scripts/PSM/run_comparison.sh

# 农村电压数据集
bash scripts/RuralVoltage/VoltageTimesNet.sh
```

## 代码架构

### 核心入口
- `run.py` - 主入口脚本，解析命令行参数并执行训练/测试流程

### 模块结构
```
code/
├── run.py                      # 主入口脚本
├── exp/exp_anomaly_detection.py    # 实验类：train() 和 test() 方法
├── models/                         # 19 个模型实现
│   ├── TimesNet.py                 # 核心模型：FFT + 2D卷积周期建模
│   ├── VoltageTimesNet.py          # 定制模型：预设电网周期 + TimesNet
│   ├── TPATimesNet.py              # 三相注意力 TimesNet
│   ├── MTSTimesNet.py              # 多尺度时序 TimesNet
│   ├── HybridTimesNet.py           # 混合周期发现 TimesNet
│   └── DLinear.py                  # 轻量级：序列分解 + 线性层
├── data_provider/
│   ├── data_factory.py             # 数据工厂：data_provider(args, flag)
│   └── data_loader.py              # 6 个数据集加载器
├── layers/                         # 网络层组件
│   ├── Embed.py                    # TokenEmbedding, PositionalEmbedding
│   ├── SelfAttention_Family.py     # FullAttention, ProbAttention
│   └── Conv_Blocks.py              # Inception 2D 卷积块
├── utils/
│   ├── tools.py                    # EarlyStopping, StandardScaler
│   └── voltage_metrics.py          # 电压异常检测指标
├── scripts/                        # 训练脚本（按数据集分类）
├── dataset/                        # 数据集
├── checkpoints/                    # 模型检查点
├── results/                        # 实验结果
└── test_results/                   # 测试结果
```

### 异常检测原理
采用**重构误差**方法：模型学习正常数据的时序模式，通过重构误差超过阈值来判定异常。
```python
threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
pred = (test_energy > threshold).astype(int)
```
评估时使用**点调整**(Point Adjustment)策略：异常段内任何点被正确检测，则整段标记为正确。

## 支持的模型 (19 个)

### 基线模型 (15 个)
- TimesNet, Transformer, DLinear, PatchTST, iTransformer
- Autoformer, Informer, FiLM, LightTS, SegRNN
- KANAD, Nonstationary_Transformer, MICN, TimeMixer, Reformer

### 创新模型 (4 个)
- VoltageTimesNet - 预设电网周期 + FFT 混合
- TPATimesNet - 三相注意力 TimesNet
- MTSTimesNet - 多尺度时序 TimesNet
- HybridTimesNet - 混合周期发现 TimesNet

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 模型隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet 的 top-k 周期数 |
| `--enc_in` | 25 | 输入特征数（PSM=25, RuralVoltage=16） |
| `--anomaly_ratio` | 1.0 | 预期异常比例(%) |

## 数据集

| 数据集 | 特征数 | 训练集 | 测试集 |
|--------|--------|--------|--------|
| PSM | 25 | 132,481 | 87,841 |
| MSL | 55 | 58,317 | 73,729 |
| SMAP | 25 | 135,183 | 427,617 |
| SMD | 38 | 708,405 | 708,420 |
| SWAT | 51 | 495,000 | 449,919 |
| RuralVoltage | 16 | 10,000 | 2,000 |

## TimesNet 模型详解

### 核心创新：1D→2D 时序建模
TimesNet 将一维时间序列转换为二维张量，利用 2D 卷积捕获周期内和周期间的复杂模式。

### 模型架构
```
输入 [B, T, C] → TokenEmbedding → TimesBlock×N → LayerNorm → 投影层 → 输出 [B, T, C]
```

### TimesBlock 工作流程
```python
# 1. FFT 周期发现
xf = torch.fft.rfft(x, dim=1)
frequency_list = abs(xf).mean(0).mean(-1)
period = (T / frequency_list.topk(k).indices).int()

# 2. 1D → 2D 重塑
x_2d = x.reshape(B, period, num_periods, C)

# 3. Inception 2D 卷积 (1×1, 3×3, 5×5, MaxPool)
x_2d = Inception_Block_V1(x_2d)

# 4. 2D → 1D 还原 + 多周期加权融合
output = Σ(softmax(weights) × x_1d_i)
```

### VoltageTimesNet 改进
1. **预设周期与 FFT 混合策略**：70% FFT + 30% 预设周期（60/300/900/3600 点）
2. **时域平滑层**：深度可分离卷积抑制高频噪声

## 论文章节结构

论文位于 `thesis/` 目录，采用北京林业大学官方模板 (BJFUThesis)。

| 章节 | 文件 | 对应代码模块 |
|------|------|-------------|
| 第一章 绪论 | `chap00.tex` | - |
| 第二章 数据采集与预处理 | `chap01.tex` | `data_provider/`, `dataset/RuralVoltage/` |
| 第三章 基于TimesNet的电压异常检测算法 | `chap02.tex` | `models/TimesNet.py`, `models/VoltageTimesNet.py` |
| 第四章 实验设计与结果分析 | `chap03.tex` | `run.py`, `scripts/`, `exp/` |
| 第五章 农村电网低电压监管平台设计与实现 | `chap04.tex` | 系统设计文档 |
| 第六章 结论与展望 | `chap05.tex` | - |

### 论文编译
```bash
cd thesis
xelatex bjfuthesis-main.tex
biber bjfuthesis-main
xelatex bjfuthesis-main.tex
xelatex bjfuthesis-main.tex
```

### RuralVoltage 数据集 16 维特征
| 特征类别 | 特征名称 | 说明 |
|---------|---------|------|
| 三相电压 | Va, Vb, Vc | 200-240V 范围 |
| 三相电流 | Ia, Ib, Ic | 10-20A 范围 |
| 功率指标 | P, Q, S, PF | 有功/无功/视在功率及功率因数 |
| 电能质量 | THD_Va, THD_Vb, THD_Vc | 谐波失真率 (GB/T 12325-2008) |
| 不平衡因子 | V_unbalance, I_unbalance | 三相不平衡程度 |
| 频率 | Freq | 50Hz 标准频率 |

### 5 种异常类型
1. **Undervoltage (欠压)**: 电压低于 198V
2. **Overvoltage (过压)**: 电压高于 235V
3. **Voltage_Sag (电压骤降)**: 电压突然下降 10%+
4. **Harmonic (谐波畸变)**: THD 超过 5%
5. **Unbalance (三相不平衡)**: 不平衡度超过 2%

## 添加新模型

1. 在 `models/` 创建模型文件，实现 `__init__` 和 `forward` 方法
2. 在 `models/__init__.py` 的 `model_dict` 中注册
3. 确保 `forward` 返回与输入维度相同的重构输出

## 实验结果规范

### 时间戳分组
- 所有实验结果**必须**按时间戳分组保存
- 目录格式：`results/PSM_comparison_YYYYMMDD_HHMMSS/`

### 中文规范
- 图表标题、坐标轴标签、图例**必须**使用中文
- 报告和 JSON 文件中的字段名**必须**使用中文

### 分析脚本使用
```bash
cd code
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_XXXXXX
```

## Commit 规范

使用中文，格式：`类型: 描述`
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `refactor:` 重构代码

示例：`feat: 添加三相电压注意力层`

## 图表规范

### IEEE 风格要求
- 格式：仅 PNG（300 DPI）
- 标题：无标题（文件名即为标题）
- 字体：中文宋体/黑体
- 尺寸：宽度 8-10 英寸

---

## 科学写作支持

本项目已集成 Scientific Writer 功能，用于学术论文、文献综述、研究报告等科学文档的撰写。

### 核心写作指南

- **写作指南**：`.claude/WRITING_GUIDE.md` - 科学论文写作的核心原则和流程
- **技能参考**：`.claude/SKILLS_REFERENCE.md` - 19+ 个专业技能的快速索引

### 关键原则

1. **所有引用必须是真实可验证的论文** - 使用 `research-lookup` 技能检索
2. **先研究后写作** - 每章 5-10 篇相关真实论文
3. **默认格式** - LaTeX + BibTeX（学术出版标准）
4. **自动质量检查** - PDF 格式验证和迭代改进

### 常用提示词

```
"使用 @research-lookup 技能查找关于 '深度学习异常检测' 的论文"
"创建一篇关于 TimesNet 的文献综述"
"使用 @peer-review 技能评估论文质量"
"生成关于电压异常检测的研究提案"
```

### 支持的文档类型

- 学术论文（Nature, Science, NeurIPS, IEEE 等）
- 文献综述和系统综述
- 研究资助提案（NSF, NIH, DOE）
- 会议海报和演示文稿
- 临床报告和治疗计划

### 项目输出

所有写作成果保存在：`writing_outputs/<timestamp>_<description>/`

详细文档：https://github.com/K-Dense-AI/claude-scientific-writer
