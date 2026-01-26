# 农村低压配电网电压异常检测研究 - 完整论文规划

> 创建时间: 2026-01-26
> 本文档记录论文规划的所有细节，供后续执行参考

---

## 一、项目背景与目标

### 1.1 研究背景

农村低压配电网电压质量问题日益突出，传统监测方法难以满足智能化需求。本研究基于深度学习方法，开发适用于农村电网特点的电压异常检测算法。

### 1.2 研究目标

1. 基于 TimesNet 模型开发适用于农村电压数据的改进算法
2. 验证领域知识融合（电网周期、三相特性）的有效性
3. 构建完整的实验评估体系
4. 设计农村电网低电压监管平台原型

### 1.3 核心创新点

| 创新点 | 说明 | 对应模型 |
|--------|------|----------|
| 自适应周期发现 | FFT + 预设周期混合策略 | VoltageTimesNet |
| 三相时空解耦 | 120°相位约束建模 | TPATimesNet |
| 多尺度时序建模 | 短/中/长期并行分支 | MTSTimesNet |
| 混合周期发现 | 置信度加权融合 | HybridTimesNet |

---

## 二、TimesNet 论文核心原理

### 2.1 论文信息

- **标题**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
- **会议**: ICLR 2023
- **核心思想**: 将一维时间序列转换为二维张量，利用2D卷积捕获周期内和周期间的复杂模式

### 2.2 TimesNet 架构

```
输入 [B, T, C]
    ↓
TokenEmbedding (线性投影)
    ↓
TimesBlock × N 层
    ├── FFT 周期发现 → 得到 top-k 周期
    ├── 1D → 2D 重塑 → [B, C, T/p, p]
    ├── Inception 2D 卷积 (1×1, 3×3, 5×5, MaxPool)
    ├── 2D → 1D 还原
    └── 多周期自适应聚合 (softmax加权)
    ↓
LayerNorm
    ↓
投影层
    ↓
输出 [B, T, C]
```

### 2.3 关键代码逻辑

```python
# 1. FFT 周期发现
def FFT_for_Period(x, k=5):
    xf = torch.fft.rfft(x, dim=1)           # 频域变换
    frequency_list = abs(xf).mean(0).mean(-1)  # 取幅值
    top_k_freq = frequency_list.topk(k)      # 选 top-k 频率
    period = (T / top_k_freq.indices).int()  # 计算周期长度
    return period, top_k_freq.values

# 2. 1D → 2D 重塑
x_2d = x.reshape(B, period, num_periods, C)  # [B, T, C] → [B, p, T/p, C]

# 3. Inception 2D 卷积
x_2d = Inception_Block_V1(x_2d)  # 多尺度 2D 卷积核

# 4. 多周期自适应聚合
output = Σ(softmax(weights) × x_1d_i)  # 加权融合各周期结果
```

### 2.4 异常检测原理

采用**重构误差**方法：
1. 模型学习正常数据的时序模式
2. 计算重构误差 (MSE)
3. 使用百分位数设定阈值
4. 重构误差超过阈值判定为异常

```python
# 阈值计算
threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
# 异常判定
pred = (test_energy > threshold).astype(int)
```

评估时使用**点调整**(Point Adjustment)策略：异常段内任何点被正确检测，则整段标记为正确。

---

## 三、创新模型设计详解

### 3.1 VoltageTimesNet - 预设周期融合

**问题**: 纯 FFT 可能遗漏已知的电网固有周期

**解决方案**: 70% FFT + 30% 预设周期混合

**预设周期列表**:
| 周期名称 | 采样点数 | 物理意义 |
|----------|----------|----------|
| 1分钟 | 60 | 短期负荷波动 |
| 5分钟 | 300 | 中期负荷变化 |
| 15分钟 | 900 | 电力调度周期 |
| 1小时 | 3600 | 日常用电模式 |

**核心代码结构** (`models/VoltageTimesNet.py`):
```python
class Model(nn.Module):
    def __init__(self, configs):
        # 预设电网周期
        self.preset_periods = [60, 300, 900, 3600]
        self.preset_weight = 0.3  # 预设周期权重
        self.fft_weight = 0.7     # FFT发现权重

        # 时域平滑层 (深度可分离卷积)
        self.smoothing = nn.Conv1d(...)

    def forward(self, x):
        # 1. 时域平滑
        x = self.smoothing(x)
        # 2. FFT 周期发现
        fft_periods, fft_weights = FFT_for_Period(x, self.top_k)
        # 3. 混合预设周期
        periods = self.merge_periods(fft_periods, self.preset_periods)
        weights = self.fft_weight * fft_weights + self.preset_weight * preset_weights
        # 4. TimesBlock 处理
        ...
```

### 3.2 TPATimesNet - 三相注意力

**问题**: 三相电压 Va, Vb, Vc 具有 120° 相位差物理约束，现有方法未显式建模

**解决方案**: 三相交叉注意力 + 相位感知编码

**三相关系建模**:
```
Va ─┐
    ├── Three-Phase Attention ─→ 融合特征
Vb ─┤     (120° 相位约束)
    │
Vc ─┘
```

**核心代码结构** (`models/TPATimesNet.py`):
```python
class ThreePhaseAttention(nn.Module):
    def __init__(self, d_model):
        self.phase_offset = nn.Parameter(torch.tensor([0, 2*pi/3, 4*pi/3]))
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4)

    def forward(self, va, vb, vc):
        # 相位感知位置编码
        va = va + self.phase_encoding(0)
        vb = vb + self.phase_encoding(2*pi/3)
        vc = vc + self.phase_encoding(4*pi/3)
        # 三相交叉注意力
        fused = self.cross_attention(va, vb, vc)
        return fused
```

**三相不平衡感知损失**:
```python
def unbalance_loss(va, vb, vc):
    # 正常系统: Va + Vb + Vc ≈ 0 (120°相位差)
    sum_voltage = va + vb + vc
    return torch.mean(sum_voltage ** 2)
```

### 3.3 MTSTimesNet - 多尺度时序

**问题**: 单一尺度难以同时捕获瞬时异常和长期趋势

**解决方案**: 并行多尺度分支 + 自适应融合

**尺度定义**:
| 尺度 | 时间范围 | 检测目标 |
|------|----------|----------|
| 短期 | 秒~分钟 | 电压骤降、瞬变 |
| 中期 | 分钟~小时 | 负荷变化、日常模式 |
| 长期 | 小时~天 | 季节模式、系统性问题 |

**核心代码结构** (`models/MTSTimesNet.py`):
```python
class Model(nn.Module):
    def __init__(self, configs):
        # 多尺度分支
        self.short_term = ScaleSpecificTimesBlock(scale='short')
        self.mid_term = ScaleSpecificTimesBlock(scale='mid')
        self.long_term = ScaleSpecificTimesBlock(scale='long')
        # 自适应融合门
        self.fusion_gate = nn.Linear(3 * d_model, 3)

    def forward(self, x):
        # 并行多尺度处理
        short_out = self.short_term(x)
        mid_out = self.mid_term(x)
        long_out = self.long_term(x)
        # 自适应融合
        gate = F.softmax(self.fusion_gate(concat), dim=-1)
        output = gate[0]*short_out + gate[1]*mid_out + gate[2]*long_out
        return output
```

### 3.4 HybridTimesNet - 混合周期发现

**问题**: 预设周期可能不完整，FFT 可能不稳定

**解决方案**: 双路径处理 + 置信度加权融合

**架构设计**:
```
输入
  ├── PresetPeriodBlock ─→ 预设周期特征 (权重 w1)
  │     └── 20ms, 100ms, 1s, 60s, 900s, 3600s
  │
  └── FFTDiscoveryBlock ─→ FFT发现特征 (权重 w2)
        └── 动态发现的 top-k 周期

融合: output = w1 * preset_out + w2 * fft_out
其中 w1, w2 由置信度网络学习
```

**核心代码结构** (`models/HybridTimesNet.py`):
```python
class Model(nn.Module):
    def __init__(self, configs):
        # 预设电气周期 (基于采样率)
        self.electrical_periods = [
            1,      # 20ms (1个电气周期 @ 50Hz)
            5,      # 100ms
            50,     # 1s
            3000,   # 60s
            45000,  # 900s (15分钟)
            180000  # 3600s (1小时)
        ]
        self.preset_block = PresetPeriodBlock(self.electrical_periods)
        self.fft_block = FFTDiscoveryBlock(top_k=5)
        self.confidence_net = nn.Sequential(...)

    def forward(self, x):
        preset_out = self.preset_block(x)
        fft_out = self.fft_block(x)
        # 置信度学习
        confidence = self.confidence_net(x)
        w1, w2 = confidence[:, 0], confidence[:, 1]
        output = w1 * preset_out + w2 * fft_out
        return output
```

---

## 四、论文章节结构详细设计

### 4.1 第一章: 数据采集与预处理 (12-15页)

```
1.1 农村电网数据采集体系
    1.1.1 采集设备与通信架构
        - 智能电表技术选型
        - 通信方式: RS485/LoRa/NB-IoT
        - 采集频率: 1秒/1分钟/15分钟
    1.1.2 采集参数与频率设计
        - 三相电压电流
        - 功率因数
        - 电能质量指标
    1.1.3 数据质量保障措施
        - 时钟同步
        - 数据校验
        - 断点续传

1.2 电压数据特征体系
    1.2.1 基础电气量特征 (Va,Vb,Vc,Ia,Ib,Ic)
        - 物理意义
        - 正常范围
        - 相互关系
    1.2.2 功率指标特征 (P,Q,S,PF)
        - 有功功率计算
        - 无功功率计算
        - 功率因数意义
    1.2.3 电能质量特征 (THD, 不平衡因子, 频率)
        - THD 计算方法
        - 三相不平衡度定义
        - 频率偏差
    1.2.4 16维特征相关性分析
        - 特征相关矩阵
        - 主成分分析
        - 特征重要性排序

1.3 数据预处理方法
    1.3.1 缺失值与异常值处理
        - 线性插值
        - 异常值检测与剔除
    1.3.2 Z-Score 标准化
        - 公式: z = (x - μ) / σ
        - 训练集统计量
    1.3.3 滑动窗口采样
        - 窗口长度选择
        - 步长设置
        - 数据增强

1.4 异常类型定义与标注
    1.4.1 国标 GB/T 12325-2008 解读
        - 电压偏差限值
        - 谐波限值
        - 不平衡度限值
    1.4.2 五类电压异常定义
        - 欠压 (V < 198V, -10%)
        - 过压 (V > 242V, +10%)
        - 电压骤降 (瞬时下降 >10%)
        - 谐波畸变 (THD > 5%)
        - 三相不平衡 (不平衡度 > 4%)
    1.4.3 异常注入方法
        - 基于规则的合成
        - 参数随机化
    1.4.4 数据集划分策略
        - 训练集/验证集/测试集
        - 时序划分 vs 随机划分

1.5 本章小结
```

### 4.2 第二章: 基于TimesNet的电压异常检测算法 (25-30页)

```
2.1 时序异常检测方法概述
    2.1.1 异常检测问题形式化
        - 无监督 vs 半监督
        - 点异常 vs 序列异常
    2.1.2 基于重构的异常检测原理
        - 自编码器思想
        - 重构误差定义
    2.1.3 深度学习时序模型演进
        - RNN/LSTM 时代
        - Transformer 时代
        - TimesNet 创新

2.2 TimesNet模型原理深度解析 ⭐
    2.2.1 模型整体架构
        - 架构图
        - 各模块功能
    2.2.2 FFT周期发现机制
        - 离散傅里叶变换原理
        - 幅值谱分析
        - Top-k周期选择策略
        - 数学推导
    2.2.3 1D-to-2D时序转换
        - 张量重塑数学表达
        - 周期内变化含义
        - 周期间变化含义
        - 可视化示例
    2.2.4 Inception 2D卷积特征提取
        - Inception 模块设计
        - 多尺度卷积核
        - 参数效率
    2.2.5 自适应多周期聚合
        - Softmax 加权
        - 注意力机制解释

2.3 面向电压检测的TimesNet改进方法 ⭐⭐ (核心创新)
    2.3.1 VoltageTimesNet: 领域知识引导的周期发现
        - 动机: 电网固有周期
        - 预设周期列表设计
        - FFT与预设周期混合策略
        - 时域平滑卷积层
        - 消融实验验证
    2.3.2 TPATimesNet: 三相注意力机制
        - 动机: 三相电压物理约束
        - 120°相位差建模
        - 三相交叉注意力设计
        - 相位感知位置编码
        - 不平衡检测能力
    2.3.3 MTSTimesNet: 多尺度时序建模
        - 动机: 农村电网多时间尺度
        - 短/中/长期分支设计
        - 自适应融合门
        - 跨尺度残差连接
    2.3.4 HybridTimesNet: 混合周期发现
        - 动机: 周期发现稳定性
        - 预设周期处理模块
        - FFT动态发现模块
        - 置信度融合机制

2.4 基于重构误差的异常检测框架
    2.4.1 重构误差计算
        - MSE 损失
        - 特征维度聚合
    2.4.2 动态阈值设定
        - 百分位数方法
        - 参数选择
    2.4.3 点调整评估策略
        - Point Adjustment 原理
        - 实现细节
    2.4.4 异常类型识别
        - 多分类扩展
        - 特征归因

2.5 模型训练与优化策略
    2.5.1 损失函数设计
        - MSE 重构损失
        - 可选: 对比学习损失
    2.5.2 Adam优化器与学习率策略
        - 初始学习率
        - 学习率衰减
    2.5.3 早停机制
        - 验证集监控
        - 耐心参数
    2.5.4 模型复杂度分析
        - 参数量统计
        - FLOPs 计算
        - 推理时间

2.6 本章小结
```

### 4.3 第三章: 实验设计与结果分析 (20-25页)

```
3.1 实验环境配置
    3.1.1 硬件环境
        - GPU: NVIDIA RTX 3090
        - 内存: 32GB
    3.1.2 软件环境
        - Python 3.11
        - PyTorch 2.x
        - 依赖库版本
    3.1.3 超参数设置
        - seq_len: 100
        - d_model: 64
        - e_layers: 2
        - top_k: 5
        - batch_size: 128
        - learning_rate: 0.0001
        - train_epochs: 10

3.2 实验数据集
    3.2.1 标准基准数据集
        - PSM (25维, 服务器监控)
        - MSL (55维, 航天器遥测)
        - SMAP (25维, 航天器传感器)
        - SMD (38维, 服务器机器)
        - SWAT (51维, 水处理系统)
    3.2.2 农村电压数据集 RuralVoltage
        - 数据来源
        - 特征定义
        - 异常标注
    3.2.3 数据集特性对比表

3.3 评估指标体系
    3.3.1 基础指标
        - Accuracy
        - Precision
        - Recall
        - F1-Score
    3.3.2 点调整后指标
        - Adjusted F1
    3.3.3 ROC-AUC, PR-AUC
        - 曲线绘制
        - 面积计算

3.4 基线模型对比实验
    3.4.1 对比模型列表 (15个)
        - TimesNet, Transformer, Informer, Autoformer
        - iTransformer, DLinear, PatchTST
        - MICN, LightTS, SegRNN, FiLM
        - KANAD, Nonstationary_Transformer, TimeMixer, Reformer
    3.4.2 标准数据集结果
        - PSM 结果表格
        - MSL 结果表格
        - SMAP 结果表格
        - SMD 结果表格
        - SWAT 结果表格
    3.4.3 RuralVoltage 结果
        - 完整对比表格
        - 最佳模型分析

3.5 改进模型性能验证
    3.5.1 VoltageTimesNet vs TimesNet
        - 性能对比
        - 周期发现可视化
    3.5.2 TPATimesNet 三相不平衡检测
        - 不平衡异常召回率
        - 注意力可视化
    3.5.3 MTSTimesNet 多尺度异常检测
        - 不同尺度异常检测效果
        - 融合权重分析
    3.5.4 HybridTimesNet 周期发现稳定性
        - 不同数据上的表现
        - 置信度分布

3.6 消融实验
    3.6.1 预设周期混合比例
        - 0%, 30%, 50%, 70%, 100%
        - 结果曲线
    3.6.2 序列长度影响
        - 50, 100, 200, 500
        - 性能与效率权衡
    3.6.3 模型深度影响
        - 1, 2, 3, 4 层
    3.6.4 周期数影响
        - top_k = 3, 5, 7, 10

3.7 效率分析
    3.7.1 训练时间对比
    3.7.2 推理速度对比
    3.7.3 参数量与内存占用
    3.7.4 边缘部署可行性

3.8 案例分析
    3.8.1 典型异常检测可视化
        - 欠压案例
        - 过压案例
        - 电压骤降案例
        - 谐波畸变案例
        - 三相不平衡案例
    3.8.2 重构误差分布分析
    3.8.3 误检与漏检分析

3.9 本章小结
```

### 4.4 第四章: 农村电网低电压监管平台设计与实现 (12-15页)

```
4.1 平台需求分析
    4.1.1 功能需求
        - 实时监测
        - 异常检测
        - 告警推送
        - 历史查询
        - 报表生成
    4.1.2 性能需求
        - 延迟 <1s
        - F1 >0.7
        - 可用性 99.9%
    4.1.3 用户角色分析
        - 运维人员
        - 管理人员
        - 设备厂商

4.2 平台整体架构设计
    4.2.1 四层架构
        - 采集层: 智能电表 + 集中器
        - 存储层: TimescaleDB + Redis
        - 算法层: 模型服务 + 消息队列
        - 展示层: Web 前端
    4.2.2 技术选型
        - 后端: FastAPI / Python
        - 前端: Vue3 + ECharts
        - 数据库: TimescaleDB (时序) + PostgreSQL (业务)
        - 消息队列: Kafka
        - 模型服务: TorchServe
    4.2.3 系统架构图

4.3 数据采集与存储模块
    4.3.1 数据接入适配
    4.3.2 时序数据存储
    4.3.3 数据压缩策略

4.4 算法服务模块
    4.4.1 模型部署方案
    4.4.2 实时推理流程
    4.4.3 模型更新机制

4.5 预警与告警模块
    4.5.1 告警规则引擎
    4.5.2 多级告警策略
    4.5.3 告警推送渠道

4.6 可视化展示模块
    4.6.1 实时监控大屏
    4.6.2 历史趋势分析
    4.6.3 异常事件回溯

4.7 系统测试
    4.7.1 功能测试
    4.7.2 性能测试
    4.7.3 压力测试

4.8 本章小结
```

### 4.5 第五章: 结论与展望 (3-5页)

```
5.1 研究工作总结
    5.1.1 主要贡献
    5.1.2 创新点回顾

5.2 研究不足
    5.2.1 数据局限性
    5.2.2 模型局限性

5.3 未来展望
    5.3.1 时空联合建模
    5.3.2 可解释性增强
    5.3.3 实际部署优化
```

---

## 五、实验设计详细方案

### 5.1 必做实验清单

#### 5.1.1 数据集对比实验

| 实验ID | 数据集 | 模型数 | 预计时间 | 输出 |
|--------|--------|--------|----------|------|
| EXP-01 | PSM | 8+ | 已完成 | 已有结果 |
| EXP-02 | MSL | 15+ | 4h | 对比表格 |
| EXP-03 | SMAP | 15+ | 4h | 对比表格 |
| EXP-04 | SMD | 15+ | 6h | 对比表格 |
| EXP-05 | SWAT | 15+ | 6h | 对比表格 |
| EXP-06 | RuralVoltage | 19 | 6h | 对比表格 |

#### 5.1.2 创新模型验证实验

| 实验ID | 对比项 | 数据集 | 目标 |
|--------|--------|--------|------|
| INN-01 | VoltageTimesNet vs TimesNet | RuralVoltage | 验证预设周期有效性 |
| INN-02 | TPATimesNet vs TimesNet | RuralVoltage/three_phase | 验证三相注意力有效性 |
| INN-03 | MTSTimesNet vs TimesNet | RuralVoltage/multi_scale | 验证多尺度有效性 |
| INN-04 | HybridTimesNet vs TimesNet | RuralVoltage/hybrid_period | 验证混合策略有效性 |

#### 5.1.3 消融实验

| 实验ID | 变量 | 取值范围 | 模型 | 数据集 |
|--------|------|----------|------|--------|
| ABL-01 | 预设周期权重 | 0%, 30%, 50%, 70%, 100% | VoltageTimesNet | RuralVoltage |
| ABL-02 | 序列长度 | 50, 100, 200, 500 | 所有 | PSM, RuralVoltage |
| ABL-03 | 模型层数 | 1, 2, 3, 4 | TimesNet系列 | PSM |
| ABL-04 | top_k | 3, 5, 7, 10 | TimesNet系列 | PSM |

### 5.2 可视化方案

| 图表类型 | 内容 | 工具 |
|----------|------|------|
| 训练曲线 | Loss vs Epoch | matplotlib |
| 性能柱状图 | Acc/P/R/F1 对比 | matplotlib |
| 雷达图 | 多维度性能 | matplotlib |
| ROC曲线 | 分类器性能 | sklearn + matplotlib |
| PR曲线 | 精确率-召回率 | sklearn + matplotlib |
| 热力图 | 模型×指标 | seaborn |
| 混淆矩阵 | TP/FP/TN/FN | sklearn + seaborn |
| 时序可视化 | 原始+重构+标注 | matplotlib |
| 周期重要性 | 热力图 | seaborn |

### 5.3 统计显著性检验

- 多次重复实验 (3-5次)
- 计算均值和标准差
- t检验比较显著性
- 使用 * 和 ** 标记显著性水平

---

## 六、PSM 实验结果汇总 (已完成)

### 6.1 实验配置

```yaml
数据集: PSM
特征数: 25
训练集: 132,481 样本
测试集: 87,841 样本
序列长度: 100
模型维度: 64
层数: 2
top_k: 3
批大小: 128
学习率: 0.0001
训练轮数: 10
```

### 6.2 性能排名

| 排名 | 模型 | Accuracy | Precision | Recall | F1 |
|:----:|:-----|:--------:|:---------:|:------:|:--:|
| 1 | TimesNet | 98.55% | 98.48% | 96.25% | **97.35%** |
| 2 | VoltageTimesNet | 98.53% | 98.40% | 96.25% | **97.31%** |
| 3 | TPATimesNet | 98.50% | 98.44% | 96.12% | **97.27%** |
| 4 | DLinear | 98.16% | 98.64% | 94.66% | 96.61% |
| 5 | MTSTimesNet | 97.97% | 97.56% | 95.06% | 96.29% |
| 6 | iTransformer | 97.22% | 97.98% | 91.85% | 94.82% |
| 7 | Transformer | 95.24% | 99.38% | 83.35% | 90.66% |
| 8 | Autoformer | 94.17% | 99.99% | 78.99% | 88.26% |

### 6.3 关键发现

1. **TimesNet 表现最佳**: F1=97.35%，验证了 2D 时序建模的有效性
2. **创新模型表现优异**: 3个创新模型进入前5名
3. **VoltageTimesNet 接近 TimesNet**: 仅差0.04%，说明预设周期未损害通用性
4. **TPATimesNet 第3名**: 三相注意力在通用数据集也有良好表现
5. **MTSTimesNet 第5名**: 多尺度设计具有一定优势
6. **DLinear 第4名**: 简单模型也能取得好效果

### 6.4 生成文件列表

```
code/results/PSM_comparison_20260125_013217/
├── analysis_20260125_135502/
│   ├── 训练曲线对比.png
│   ├── 性能指标对比.png
│   ├── 雷达图对比.png
│   ├── F1分数对比.png
│   ├── ROC曲线.png
│   ├── PR曲线.png
│   ├── 性能热力图.png
│   ├── 混淆矩阵_TimesNet.png
│   ├── 实验结果.json
│   ├── 实验分析报告.md
│   └── 科学实验分析报告.md
└── [各模型训练日志]
```

---

## 七、RuralVoltage 数据集详情

### 7.1 数据集结构

```
dataset/RuralVoltage/
├── train.csv (3.0MB, 10000样本)
├── test.csv (620KB, 2000样本)
├── test_label.csv (47KB)
├── comprehensive/ (综合评估)
├── periodic_load/ (周期负荷 → VoltageTimesNet)
├── three_phase/ (三相数据 → TPATimesNet)
├── multi_scale/ (多尺度 → MTSTimesNet)
└── hybrid_period/ (混合周期 → HybridTimesNet)
```

### 7.2 16维特征定义

| 序号 | 特征名 | 类别 | 范围 | 说明 |
|------|--------|------|------|------|
| 1 | Va | 三相电压 | 200-240V | A相电压 |
| 2 | Vb | 三相电压 | 200-240V | B相电压 |
| 3 | Vc | 三相电压 | 200-240V | C相电压 |
| 4 | Ia | 三相电流 | 10-20A | A相电流 |
| 5 | Ib | 三相电流 | 10-20A | B相电流 |
| 6 | Ic | 三相电流 | 10-20A | C相电流 |
| 7 | P | 功率 | kW | 有功功率 |
| 8 | Q | 功率 | kVar | 无功功率 |
| 9 | S | 功率 | kVA | 视在功率 |
| 10 | PF | 功率 | 0-1 | 功率因数 |
| 11 | THD_Va | 电能质量 | % | A相谐波失真率 |
| 12 | THD_Vb | 电能质量 | % | B相谐波失真率 |
| 13 | THD_Vc | 电能质量 | % | C相谐波失真率 |
| 14 | V_unbalance | 不平衡 | % | 电压不平衡度 |
| 15 | I_unbalance | 不平衡 | % | 电流不平衡度 |
| 16 | Freq | 频率 | Hz | 系统频率 |

### 7.3 5种异常类型

| 类型 | 英文名 | 判定条件 | 国标依据 |
|------|--------|----------|----------|
| 欠压 | Undervoltage | V < 198V (-10%) | GB/T 12325-2008 |
| 过压 | Overvoltage | V > 242V (+10%) | GB/T 12325-2008 |
| 电压骤降 | Voltage_Sag | 瞬时下降 >10% | IEEE 1159 |
| 谐波畸变 | Harmonic | THD > 5% | GB/T 14549-1993 |
| 三相不平衡 | Unbalance | 不平衡度 > 4% | GB/T 15543-2008 |

### 7.4 针对性子数据集

| 子数据集 | 目标模型 | 训练集 | 测试集 | 异常比例 | 异常类型 |
|----------|----------|--------|--------|----------|----------|
| periodic_load | VoltageTimesNet | 10,000 | 3,000 | 15% | 2 |
| three_phase | TPATimesNet | 10,000 | 3,000 | 23.3% | 8 |
| multi_scale | MTSTimesNet | 15,000 | 5,000 | 47.4% | 10 |
| hybrid_period | HybridTimesNet | 12,000 | 4,000 | 60.2% | 4 |
| comprehensive | 所有模型 | 20,000 | 6,000 | 48.8% | 6 |

---

## 八、训练命令参考

### 8.1 单模型训练

```bash
cd code

# TimesNet on PSM
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 \
  --e_layers 2 --top_k 3 --train_epochs 10 --batch_size 128

# VoltageTimesNet on RuralVoltage
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ --enc_in 16 --c_out 16 \
  --seq_len 100 --d_model 64 --d_ff 128 --top_k 5

# TPATimesNet on RuralVoltage/three_phase
python run.py --is_training 1 --model TPATimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/three_phase/ --enc_in 16 --c_out 16
```

### 8.2 批量实验脚本

```bash
# PSM 对比实验
bash scripts/PSM/run_comparison.sh

# RuralVoltage 基线实验
bash scripts/RuralVoltage/run_baselines.sh

# RuralVoltage 消融实验
bash scripts/RuralVoltage/run_ablation.sh
```

### 8.3 结果分析

```bash
# 分析 PSM 实验结果
python scripts/common/analyze_comparison_results.py \
  --result_dir ./results/PSM_comparison_*

# 生成完整分析报告
python scripts/common/generate_full_analysis.py
```

---

## 九、论文编译命令

```bash
cd thesis

# 完整编译流程
xelatex bjfuthesis-main.tex
biber bjfuthesis-main
xelatex bjfuthesis-main.tex
xelatex bjfuthesis-main.tex

# 清理临时文件
rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.run.xml
```

---

## 十、后续执行指南

### 10.1 恢复上下文

重启 Claude Code 后，执行:

```
请读取以下文件恢复上下文:
1. PROJECT_STATUS.md - 项目状态
2. TASKS.md - 任务清单
3. THESIS_PLAN.md - 论文规划 (本文件)

然后继续执行待做任务。
```

### 10.2 优先级排序

**第一阶段 (本周)**:
1. EXP-06: RuralVoltage 基线对比
2. EXP-07: RuralVoltage 创新模型验证
3. ABL-01: 预设周期比例消融

**第二阶段 (下周)**:
1. DOC-02: 第二章 TimesNet 算法
2. DOC-03: 第三章 实验结果
3. DOC-01: 第一章 数据预处理

**第三阶段 (第三周)**:
1. EXP-02~05: 其他标准数据集
2. ABL-02~04: 其他消融实验

**第四阶段 (第四周)**:
1. DOC-04: 第四章 平台设计
2. DOC-05: 第五章 结论
3. 论文编译和格式调整

---

*本文档完整记录论文规划信息，供后续执行参考*
