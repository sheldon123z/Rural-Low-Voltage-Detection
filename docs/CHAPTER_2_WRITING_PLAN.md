# 第三章写作规划：基于TimesNet的电压异常检测算法

> 科学批判性思维分析与详细写作指导
> 创建时间: 2026-01-26
> 对应文件: thesis/contents/chap02.tex

## 一、科学批判性分析

### 1.1 研究方法论评估

**研究设计类型**：基于重构误差的深度学习时序异常检测

**方法学优势**：
- ✅ 无监督/自监督学习范式，不需要异常标签进行训练
- ✅ 利用FFT自动发现数据周期特征，具有可解释性
- ✅ 2D卷积能够同时捕获周期内和周期间的时序变化
- ✅ 多个创新模型针对农村电网特性进行了定制化改进

**潜在弱点与缓解措施**：
| 潜在问题 | 科学评估 | 缓解措施 |
|---------|---------|---------|
| FFT假设平稳周期 | 农村电网具有日、周、季节等多层周期，但存在突变 | MTSTimesNet多尺度设计应对 |
| 重构误差对罕见模式敏感 | 可能将罕见正常模式误判为异常 | 使用Point Adjustment评估策略 |
| 超参数敏感性 | top_k、seq_len等参数选择影响性能 | 消融实验验证鲁棒性 |
| 阈值设定主观性 | percentile方法依赖先验异常比例 | 结合国标GB/T 12325-2008客观标准 |

### 1.2 证据质量评估框架（GRADE）

**证据来源与质量**：
1. **TimesNet原论文**（ICLR 2023）- 顶级会议，高质量证据
2. **PSM/MSL/SMAP/SMD/SWAT数据集** - 标准公开基准，可复现
3. **RuralVoltage数据集** - 自建合成数据集，需说明生成方法和局限性
4. **创新模型** - 原创贡献，需充分消融验证

**对比基线的选择依据**：
- 经典模型：Transformer、Informer、Autoformer（引用充分）
- 轻量级模型：DLinear（2023）、LightTS（2022）
- 新型模型：PatchTST（2023）、iTransformer（2024）

### 1.3 逻辑谬误检查清单

写作时需避免：
- ❌ 相关性推断因果性（周期发现只是关联，非因果）
- ❌ 过度泛化（PSM结果不能直接推广到所有场景）
- ❌ 选择性报告（必须报告所有模型结果，包括表现不佳的）
- ❌ P-hacking（预设评估指标，不能事后选择有利指标）

---

## 二、章节写作详细规划

### 2.1 时序异常检测方法概述 (约2000字)

**内容要点**：

#### 2.1.1 异常检测问题形式化
- 数学定义：给定时序 $X = \{x_1, x_2, ..., x_T\}$，目标是识别异常子序列
- 三种异常类型：点异常、上下文异常、集体异常
- 形式化损失函数：$L_{reconstruction} = \frac{1}{T}\sum_{t=1}^{T}||x_t - \hat{x}_t||^2$

#### 2.1.2 基于重构的异常检测原理
- 核心假设：正常模式可重构，异常模式难重构
- 数学推导：重构误差 $e_t = ||x_t - f_\theta(x_{t-w:t-1})||$
- 阈值判定：$y_t = \mathbb{1}[e_t > \tau]$

#### 2.1.3 深度学习时序模型演进
| 阶段 | 代表模型 | 核心创新 | 局限性 |
|------|---------|---------|--------|
| RNN时代 | LSTM/GRU | 门控机制解决梯度消失 | 顺序计算，训练慢 |
| 注意力时代 | Transformer | 自注意力捕获长程依赖 | 时序归纳偏置不足 |
| 时序特化 | TimesNet | FFT周期+2D卷积 | 假设周期平稳 |

**引用规范**：
- Hochreiter & Schmidhuber 1997 (LSTM)
- Vaswani et al. 2017 (Transformer)
- Wu et al. 2023 (TimesNet - ICLR 2023)

---

### 2.2 TimesNet模型原理深度解析 (约4000字)

**核心内容与公式**：

#### 2.2.1 模型整体架构
```
输入 X ∈ R^{B×T×C}
    → TokenEmbedding
    → TimesBlock × N_layers
    → LayerNorm
    → OutputProjection
    → 输出 Ŷ ∈ R^{B×T×C}
```

**架构图描述**（需生成IEEE风格图表）：
- 图号：图3-1 TimesNet模型整体架构图
- 内容：显示数据流、各模块连接关系

#### 2.2.2 FFT周期发现机制

**数学公式**：
$$X_f = \text{FFT}(X), \quad X_f \in \mathbb{C}^{T \times C}$$

$$A_k = \frac{1}{C}\sum_{c=1}^{C}|X_f[k, c]|, \quad k = 1, 2, ..., \lfloor T/2 \rfloor$$

$$\{p_1, p_2, ..., p_K\} = \arg\text{Top-K}(\{A_k\}_{k=1}^{\lfloor T/2 \rfloor})$$

$$\text{period}_i = \lfloor T / p_i \rfloor$$

**代码对应**：
```python
# 对应 models/TimesNet.py: FFT_for_Period()
xf = torch.fft.rfft(x, dim=1)
frequency_list = abs(xf).mean(0).mean(-1)
_, top_list = torch.topk(frequency_list, k)
period_list = T // (top_list + 1)
```

#### 2.2.3 1D-to-2D时序转换

**核心创新**：将1D时序重塑为2D张量，利用成熟的2D卷积技术

**数学表示**：
$$X_{1D} \in \mathbb{R}^{B \times T \times D} \rightarrow X_{2D} \in \mathbb{R}^{B \times D \times (T/p) \times p}$$

**图示**（需生成）：
- 图号：图3-2 1D到2D时序转换示意图
- 内容：展示padding、reshape过程

#### 2.2.4 Inception 2D卷积特征提取

**多尺度卷积核**：
- 1×1卷积：点级特征
- 3×3卷积：局部模式
- 5×5卷积：中等范围模式
- 7×7卷积：大范围模式
- MaxPool：显著特征

**代码对应**：layers/Conv_Blocks.py 的 Inception_Block_V1

#### 2.2.5 自适应多周期聚合

**加权融合**：
$$X_{out} = \sum_{i=1}^{K} \text{softmax}(A_{p_i}) \cdot X_{2D \rightarrow 1D}^{(i)}$$

---

### 2.3 VoltageTimesNet改进设计 (约2500字)

**核心创新**：领域知识引导的周期发现

#### 2.3.1 电网固有周期分析

**物理依据**：
| 周期 | 时长 | 物理意义 |
|------|------|---------|
| 1min (60点) | 短期波动 | 分钟级负荷变化 |
| 5min (300点) | 中短期 | 设备启停周期 |
| 15min (900点) | 中期 | 电力调度周期 |
| 1h (3600点) | 长期 | 日负荷曲线基础 |

#### 2.3.2 FFT与预设周期混合策略

**数学公式**：
$$P_{hybrid} = \alpha \cdot P_{FFT} + (1-\alpha) \cdot P_{preset}$$

其中 $\alpha = 0.7$（70% FFT + 30% 预设）

**代码对应**：models/VoltageTimesNet.py: FFT_for_Period_Voltage()

#### 2.3.3 时域平滑层设计

**深度可分离卷积**：
$$X_{smooth} = \text{DepthwiseConv1D}(X) + X$$

目的：抑制高频噪声，保留主要趋势

---

### 2.4 其他创新模型 (约3000字)

#### 2.4.1 TPATimesNet：三相注意力机制

**物理约束**：三相电压具有120°相位差
$$V_a + V_b + V_c \approx 0 \quad (\text{在平衡状态下})$$

**注意力偏置矩阵**：
$$\text{bias} = \begin{pmatrix} 1 & 0.5 & 0.5 \\ 0.5 & 1 & 0.5 \\ 0.5 & 0.5 & 1 \end{pmatrix}$$

**代码对应**：models/TPATimesNet.py

#### 2.4.2 MTSTimesNet：多尺度时序建模

**三尺度并行分支**：
- 短期分支：min_period=2, max_period=20
- 中期分支：min_period=20, max_period=60
- 长期分支：min_period=60, max_period=200

**跨尺度残差连接**和**自适应融合门**

#### 2.4.3 HybridTimesNet：混合周期发现

**置信度加权融合**：
$$W_{final} = \alpha \cdot W_{learned} + (1-\alpha) \cdot [0.5, 0.5]$$

---

### 2.5 异常检测框架 (约1500字)

#### 2.5.1 重构误差计算
$$e_t = \frac{1}{C}\sum_{c=1}^{C}(x_{t,c} - \hat{x}_{t,c})^2$$

#### 2.5.2 动态阈值设定
$$\tau = \text{percentile}(E_{combined}, 100 - r)$$
其中 $r$ 为预期异常比例

#### 2.5.3 点调整评估策略
- 原因：时序异常具有连续性，单点检出即可视为成功
- 算法：若异常段内任一点被检出，整段标记为TP

---

### 2.6 模型训练与优化 (约1000字)

#### 2.6.1 损失函数
$$L = \text{MSE}(X, \hat{X}) = \frac{1}{BTC}\sum_{b,t,c}(x_{b,t,c} - \hat{x}_{b,t,c})^2$$

#### 2.6.2 优化器配置
- Adam优化器：$\beta_1=0.9, \beta_2=0.999$
- 学习率：$lr=10^{-4}$
- 学习率衰减：每epoch按比例衰减

#### 2.6.3 早停机制
- patience=3
- 监控指标：验证集Loss

---

## 三、图表规划

| 图号 | 标题 | 类型 | 优先级 |
|------|------|------|--------|
| 图3-1 | TimesNet模型整体架构图 | 架构图 | 高 |
| 图3-2 | 1D到2D时序转换示意图 | 流程图 | 高 |
| 图3-3 | FFT周期发现机制示意图 | 流程图 | 中 |
| 图3-4 | Inception 2D卷积结构图 | 网络结构图 | 中 |
| 图3-5 | VoltageTimesNet架构图 | 架构图 | 高 |
| 图3-6 | 预设周期与FFT混合策略 | 示意图 | 中 |
| 图3-7 | 三相注意力机制示意图 | 示意图 | 中 |
| 图3-8 | 多尺度时序建模框架图 | 架构图 | 中 |
| 图3-9 | 异常检测框架流程图 | 流程图 | 高 |

---

## 四、参考文献规划

### 核心引用（必须）
1. Wu et al. 2023. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
2. GB/T 12325-2008. 电能质量 供电电压偏差.
3. Vaswani et al. 2017. "Attention Is All You Need." NeurIPS 2017.

### 补充引用（推荐）
- Informer (AAAI 2021)
- Autoformer (NeurIPS 2021)
- DLinear (AAAI 2023)
- PatchTST (ICLR 2023)
- iTransformer (ICLR 2024)

---

## 五、写作检查清单

- [ ] 所有数学公式编号正确
- [ ] 代码与公式对应关系清晰
- [ ] 图表引用完整
- [ ] 术语首次出现时有英文全称
- [ ] 创新点与引言一致
- [ ] 无逻辑跳跃
- [ ] 参考文献格式统一
