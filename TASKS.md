# 论文项目任务清单

> 最后更新: 2026-02-12
> 详细规划请参考: `THESIS_PLAN.md`
> 项目状态请参考: `PROJECT_STATUS.md`

---

## 状态说明

- ✅ 已完成
- 🔄 进行中
- ⏳ 待做
- ❌ 已取消

---

## 一、实验任务

### 1.1 数据集对比实验

| ID | 任务 | 状态 | 优先级 | 备注 |
|----|------|------|--------|------|
| EXP-01 | PSM 数据集多模型对比 | ✅ | 高 | 8模型完成，F1=97.35% |
| EXP-02 | MSL 数据集多模型对比 | ✅ | 中 | TimesNet F1=0.7636 |
| EXP-03 | SMAP 数据集多模型对比 | ✅ | 中 | TimesNet F1=0.6865 |
| EXP-04 | SMD 数据集多模型对比 | ✅ | 中 | TimesNet F1=0.8246 |
| EXP-05 | SWAT 数据集多模型对比 | ⏳ | 低 | 51维特征，非核心实验 |
| EXP-06 | RuralVoltage 基线模型对比 | ✅ | **高** | 6模型完成，DLinear F1=0.8785 |
| EXP-07 | RuralVoltage 创新模型验证 | ✅ | **高** | Optuna 30-trial 优化，VoltageTimesNet F1=0.8149 |

### 1.2 创新模型专项验证

| ID | 任务 | 状态 | 优先级 | 数据集 | 备注 |
|----|------|------|--------|--------|------|
| INN-01 | VoltageTimesNet vs TimesNet | ✅ | 高 | RuralVoltage/realistic_v2 | Optuna优化后 F1 提升 34.5% |
| INN-02 | TPATimesNet vs TimesNet | ✅ | 高 | RuralVoltage/three_phase | 补充实验已完成 |
| INN-03 | MTSTimesNet vs TimesNet | ✅ | 高 | RuralVoltage/multi_scale | 补充实验已完成 |
| INN-04 | HybridTimesNet vs TimesNet | ✅ | 高 | RuralVoltage/hybrid_period | 补充实验已完成 |

### 1.3 消融实验

| ID | 任务 | 状态 | 优先级 | 变量 | 取值范围 |
|----|------|------|--------|------|----------|
| ABL-01 | 预设周期比例消融 (alpha) | ✅ | **高** | alpha | 0.5, 0.6, 0.7, 0.8, 0.9 |
| ABL-02 | 序列长度消融 | ✅ | 中 | seq_len | 50, 100, 200, 500 (TimesNet) + 100, 360, 720 (VoltageTimesNet) |
| ABL-03 | 模型深度消融 | ⏳ | 低 | e_layers | 1, 2, 3, 4 |
| ABL-04 | 周期数消融 | ⏳ | 低 | top_k | 3, 5, 7, 10 |

### 1.4 可视化任务

| ID | 任务 | 状态 | 优先级 | 备注 |
|----|------|------|--------|------|
| VIS-01 | PSM 实验分析图表 | ✅ | 高 | 8张图已生成 |
| VIS-02 | RuralVoltage 实验分析图表 | ✅ | 高 | chap4/ 下 27 张图已生成 |
| VIS-03 | 异常检测时序可视化 | ✅ | 中 | fig_4_8 系列已生成 |
| VIS-04 | 相位注意力热力图 | ✅ | 中 | fig_4_14 已生成 |
| VIS-05 | 消融实验曲线图 | ✅ | 中 | fig_4_10 (seq_len) + fig_4_11 (alpha) 已生成 |
| VIS-06 | 16维特征相关性热力图 | ⏳ | **高** | 新增：第二章数据描述 |
| VIS-07 | 重构误差分布对比图 | ⏳ | **高** | 新增：正常 vs 异常 |
| VIS-08 | 跨数据集F1分组柱状图 | ⏳ | **高** | 新增：模型泛化能力 |
| VIS-09 | Optuna搜索过程图 | ⏳ | **高** | 新增：超参数优化 |
| VIS-10 | 五种异常类型波形对比 | ⏳ | 中 | 新增：第二章补充 |
| VIS-11 | FFT频谱分析对比图 | ⏳ | 中 | 新增：正常 vs 异常频谱 |
| VIS-12 | 各异常类型检测率分解 | ⏳ | 中 | 新增：第四章深化分析 |
| VIS-13 | 模型参数量-性能散点图 | ⏳ | 低 | 新增：效率分析 |

---

## 二、论文任务

### 2.1 章节编写

| ID | 任务 | 状态 | 优先级 | 完成度 | 文件 | 行数 |
|----|------|------|--------|--------|------|------|
| DOC-00 | 绪论 | ✅ | 高 | 100% | `chap00.tex` | 91 |
| DOC-01 | 第一章: 数据采集与预处理 | ✅ | **高** | 100% | `chap01.tex` | 282 |
| DOC-02 | 第二章: TimesNet异常检测算法 | ✅ | **高** | 100% | `chap02.tex` | 761 |
| DOC-03 | 第三章: 实验设计与结果分析 | 🔄 | **高** | 85% | `chap03.tex` | 575 |
| DOC-04 | 第四章: 监管平台设计 | ✅ | 中 | 100% | `chap04.tex` | 385 |
| DOC-05 | 第五章: 结论与展望 | ✅ | 中 | 100% | `chap05.tex` | 74 |

### 2.2 第一章子任务

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| DOC-01a | 1.1 数据采集体系架构 | ✅ | 智能电表、通信方式 |
| DOC-01b | 1.2 16维特征详细说明 | ✅ | 物理意义、范围、相关性 |
| DOC-01c | 1.3 数据预处理方法 | ✅ | 缺失值、标准化、窗口采样 |
| DOC-01d | 1.4 五种异常类型定义 | ✅ | 国标依据、判定条件 |
| DOC-01e | 1.4 异常注入与数据集划分 | ✅ | 合成方法、划分策略 |

### 2.3 第二章子任务 (核心)

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| DOC-02a | 2.1 时序异常检测概述 | ✅ | 问题形式化、重构原理 |
| DOC-02b | 2.2 TimesNet原理深度解析 | ✅ | FFT、1D→2D、Inception、聚合 |
| DOC-02c | 2.3.1 VoltageTimesNet改进 | ✅ | 预设周期融合策略 |
| DOC-02d | 2.3.2 TPATimesNet改进 | ✅ | 三相注意力机制 |
| DOC-02e | 2.3.3 MTSTimesNet改进 | ✅ | 多尺度时序建模 |
| DOC-02f | 2.3.4 HybridTimesNet改进 | ✅ | 混合周期发现 |
| DOC-02g | 2.4 异常检测框架 | ✅ | 重构误差、阈值、评估 |
| DOC-02h | 2.5 训练优化策略 | ✅ | 损失函数、优化器、早停 |

### 2.4 第三章子任务

| ID | 任务 | 状态 | 依赖 | 备注 |
|----|------|------|------|------|
| DOC-03a | 3.1 实验环境配置 | ✅ | - | 硬件、软件、超参数 |
| DOC-03b | 3.2 数据集介绍 | ✅ | - | 6个数据集特性对比 |
| DOC-03c | 3.3 评估指标体系 | ✅ | - | Acc/P/R/F1/AUC |
| DOC-03d | 3.4 基线对比结果 | 🔄 | EXP-01~06 | 需更新最新 RuralVoltage 6模型结果 |
| DOC-03e | 3.5 创新模型分析 | 🔄 | EXP-07, INN-* | 需更新 Optuna 优化结果 |
| DOC-03f | 3.6 消融实验分析 | ✅ | ABL-* | alpha + seq_len 消融已写入 |
| DOC-03g | 3.7 效率分析 | ⏳ | - | 时间、参数、内存 |
| DOC-03h | 3.8 案例分析 | ✅ | VIS-03 | 典型异常可视化已完成 |

### 2.5 第四章子任务

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| DOC-04a | 4.1 需求分析 | ✅ | 功能、性能、用户 |
| DOC-04b | 4.2 架构设计 | ✅ | 四层架构、技术选型 |
| DOC-04c | 4.3-4.6 模块设计 | ✅ | 采集、算法、告警、展示 |
| DOC-04d | 4.7 系统测试 | ✅ | 功能、性能、压力 |

---

## 三、代码任务

### 3.1 训练脚本

| ID | 任务 | 状态 | 优先级 | 备注 |
|----|------|------|--------|------|
| CODE-01 | PSM 对比脚本 | ✅ | - | `scripts/PSM/run_comparison.sh` |
| CODE-02 | RuralVoltage 基线脚本 | ✅ | - | `scripts/RuralVoltage/run_baselines.sh` |
| CODE-03 | RuralVoltage 消融脚本 | ✅ | - | `scripts/RuralVoltage/run_ablation.sh` |
| CODE-04 | MSL 对比脚本 | ✅ | - | 实验已运行完成 |
| CODE-05 | SMAP 对比脚本 | ✅ | - | 实验已运行完成 |
| CODE-06 | SMD 对比脚本 | ✅ | - | 实验已运行完成 |
| CODE-07 | SWAT 对比脚本 | ⏳ | 低 | 非核心实验 |

### 3.2 分析脚本

| ID | 任务 | 状态 | 优先级 | 备注 |
|----|------|------|--------|------|
| CODE-08 | 结果分析脚本 | ✅ | - | `analyze_comparison_results.py` |
| CODE-09 | 消融实验分析 | ✅ | - | fig_4_10/fig_4_11 脚本已生成 |
| CODE-10 | 时序可视化脚本 | ✅ | - | fig_4_8 系列脚本已生成 |
| CODE-11 | Optuna 优化脚本 | ✅ | - | `scripts/optuna_optimize_voltage.py` |

### 3.3 图表脚本

| ID | 任务 | 状态 | 优先级 | 备注 |
|----|------|------|--------|------|
| CODE-12 | 特征相关性热力图脚本 | ⏳ | **高** | 新增 |
| CODE-13 | 重构误差分布图脚本 | ⏳ | **高** | 新增 |
| CODE-14 | 跨数据集F1对比图脚本 | ⏳ | **高** | 新增 |
| CODE-15 | Optuna搜索过程图脚本 | ⏳ | **高** | 新增 |

---

## 四、剩余工作

### 高优先级

| 序号 | 任务 | 说明 |
|:----:|------|------|
| 1 | 更新第三章实验数据 | 补充 RuralVoltage 6模型对比 + Optuna 优化结果 |
| 2 | 生成新增高优先级图表 | VIS-06~09: 特征热力图、重构误差分布、跨数据集对比、Optuna过程 |
| 3 | 论文质量优化 | 参考 `Thesis/TODO.md` 和 `Thesis/REMAINING_WORK_PLAN.md` |

### 中优先级

| 序号 | 任务 | 说明 |
|:----:|------|------|
| 4 | 补充效率分析 (DOC-03g) | 训练时间、参数量、推理速度对比 |
| 5 | 统计显著性测试 | 5 random seeds 重复实验 |
| 6 | 生成中优先级图表 | VIS-10~12 |

### 低优先级

| 序号 | 任务 | 说明 |
|:----:|------|------|
| 7 | SWAT 数据集实验 (EXP-05) | 非核心，可选 |
| 8 | 模型深度/周期数消融 (ABL-03/04) | 补充性实验 |
| 9 | 论文格式最终检查 | 编译、格式、图表美化 |

---

## 五、快速参考

### 5.1 关键文件路径

| 类型 | 路径 |
|------|------|
| 项目状态 | `PROJECT_STATUS.md` |
| 任务清单 | `TASKS.md` (本文件) |
| 论文规划 | `THESIS_PLAN.md` |
| 论文质量优化 | `Thesis/TODO.md` |
| 剩余工作计划 | `Thesis/REMAINING_WORK_PLAN.md` |
| PSM 实验结果 | `code/results/PSM_comparison_20260125_013217/` |
| RuralVoltage 实验结果 | `code/results/full_experiment_20260201_235401/` |
| Optuna 优化结果 | `code/results/optuna/full_search_20260203_202801.json` |
| 消融实验结果 | `code/results/alpha_ablation_20260128_003233/` + `seq_len_ablation_20260128_071326/` |
| 最优模型配置 | `code/newest_models/best_model_config.json` |
| 训练脚本 | `code/scripts/` |
| 论文源文件 | `Thesis/contents/` |
| 模型代码 | `code/models/` |
| 图表脚本 | `code/figures/scripts/` |
| 图表输出 | `code/figures/output/` |
| 写作材料 | `code/docs/` |

### 5.2 核心实验结果速查

**RuralVoltage（论文主实验）**：

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|------|:------:|:------:|:------:|:------:|
| **VoltageTimesNet (Optuna)** | **0.9393** | **0.7371** | **0.9110** | **0.8149** |
| TimesNet | 0.8584 | 0.5143 | 0.7115 | 0.5970 |
| LSTMAutoEncoder | 0.7905 | 0.3654 | 0.5712 | 0.4457 |
| Isolation Forest | 0.3474 | 0.3474 | 1.0000 | 0.5157 |
| One-Class SVM | 0.3474 | 0.3474 | 1.0000 | 0.5157 |

### 5.3 快速命令

```bash
# 查看项目状态
cat PROJECT_STATUS.md

# 编译论文
cd Thesis && latexmk -xelatex --shell-escape bjfuthesis-main.tex

# 生成所有图表
cd code/figures/scripts && python generate_all_figures.py

# 同步图表到论文
cd code/figures/scripts && python sync_to_thesis.py
```

### 5.4 恢复上下文

重启 Claude Code 后，告诉 Claude:

> "请读取以下文件恢复上下文并继续执行:
> 1. PROJECT_STATUS.md
> 2. TASKS.md
> 3. THESIS_PLAN.md"

---

## 六、进度统计

### 实验进度
- 总任务: 20
- 已完成: 16
- 进行中: 0
- 待做: 4 (EXP-05, ABL-03, ABL-04, 非核心)

### 论文进度
- 总章节: 6
- 已完成: 5 (绪论 + 第一~二章 + 第四~五章)
- 进行中: 1 (第三章 85%)

### 图表进度
- 已有图表: 37 张 (chap2: 2, chap3: 8, chap4: 27)
- 待生成: 8 张 (VIS-06~13)

### 代码进度
- 总任务: 15
- 已完成: 11
- 待做: 4 (CODE-07, CODE-12~15)

### 总体完成率: ~90-95%

---

*更新时间: 2026-02-12*
*下次更新时请同步修改 PROJECT_STATUS.md*
