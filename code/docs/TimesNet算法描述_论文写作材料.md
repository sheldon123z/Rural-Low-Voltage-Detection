# TimesNet 算法原理与改进方法 - 论文写作材料

> **用途**: 本文档为论文第三章"基于TimesNet的电压异常检测算法"的写作素材
> **生成时间**: 2026-01-26

---

## 一、TimesNet 模型原理

### 1.1 核心创新思想

TimesNet 的核心创新在于将**一维时间序列转换为二维张量**进行处理，从而能够同时捕获：
- **周期内变化 (Intraperiod Variation)**: 单个周期内的局部模式
- **周期间变化 (Interperiod Variation)**: 跨周期的趋势和相关性

### 1.2 模型整体架构

```
输入 X ∈ R^(B×T×C)
    ↓
TokenEmbedding (线性投影)
    ↓
PositionalEmbedding (位置编码)
    ↓
┌─────────────────────────────────────┐
│       TimesBlock × N 层             │
│  ┌─────────────────────────────────┐│
│  │ 1. FFT 周期发现                 ││
│  │ 2. 1D → 2D 重塑                 ││
│  │ 3. Inception 2D 卷积            ││
│  │ 4. 2D → 1D 还原                 ││
│  │ 5. 多周期自适应聚合             ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    ↓
LayerNorm
    ↓
线性投影层
    ↓
输出 Y ∈ R^(B×T×C)
```

### 1.3 FFT 周期发现机制

**算法步骤**:

1. **傅里叶变换**: 对输入序列进行快速傅里叶变换
   ```
   X_f = FFT(X)  # X ∈ R^(B×T×C) → X_f ∈ C^(B×T×C)
   ```

2. **幅值谱计算**: 计算各频率分量的幅值
   ```
   A = |X_f|  # 取复数模
   A_avg = mean(A, dim=[0, 2])  # 平均得到 A_avg ∈ R^T
   ```

3. **Top-k 周期选择**: 选择幅值最大的 k 个频率对应的周期
   ```
   f_1, f_2, ..., f_k = argtop_k(A_avg[1:T/2])  # 排除直流分量
   p_i = T / f_i  # 计算周期长度
   ```

**数学表示**:

设输入序列为 $\mathbf{x} \in \mathbb{R}^T$，其离散傅里叶变换为：

$$\mathbf{A} = \text{Amp}(\text{FFT}(\mathbf{x})) = \text{Avg}\left(\sqrt{\text{Re}(\mathbf{F})^2 + \text{Im}(\mathbf{F})^2}\right)$$

选择 Top-k 频率：

$$\{f_1, f_2, \ldots, f_k\} = \text{arg Top-}k(\mathbf{A})$$

对应周期长度：

$$p_i = \lceil T / f_i \rceil, \quad i = 1, 2, \ldots, k$$

### 1.4 1D → 2D 时序转换

**转换原理**:

给定周期 $p$，将一维序列重塑为二维张量：

$$\mathbf{x}_{1D} \in \mathbb{R}^{B \times T \times C} \rightarrow \mathbf{x}_{2D} \in \mathbb{R}^{B \times (T/p) \times p \times C}$$

**物理意义**:
- 行方向 (T/p): 周期间变化 (相邻周期的对应位置)
- 列方向 (p): 周期内变化 (单个周期内的时间演进)

**代码实现要点**:

```python
# 填充序列长度使其能被周期整除
if seq_len % period != 0:
    pad_len = period - (seq_len % period)
    x = F.pad(x, (0, 0, 0, pad_len))

# 重塑为 2D 张量
x_2d = x.reshape(B, -1, period, C)  # [B, num_periods, period, C]
x_2d = x_2d.permute(0, 3, 1, 2)      # [B, C, num_periods, period]
```

### 1.5 Inception 2D 卷积模块

**多尺度卷积设计**:

| 分支 | 卷积核大小 | 作用 |
|------|:----------:|------|
| 分支1 | 1×1 | 通道间信息交互，降维 |
| 分支2 | 3×3 | 捕获局部模式 |
| 分支3 | 5×5 | 捕获中等尺度模式 |
| 分支4 | MaxPool 3×3 + 1×1 | 提取显著特征 |

**特点**:
- 多尺度并行处理，捕获不同尺度的时序模式
- 2D 卷积能够同时建模周期内和周期间的依赖关系

### 1.6 自适应多周期聚合

**聚合公式**:

$$\mathbf{x}_{out} = \sum_{i=1}^{k} \text{softmax}(\mathbf{A}_{f_i}) \cdot \mathbf{x}_{2D \rightarrow 1D}^{(i)}$$

其中：
- $\mathbf{A}_{f_i}$ 是第 $i$ 个周期对应频率的幅值
- softmax 归一化确保权重和为 1
- 幅值越大的周期，对最终输出贡献越大

---

## 二、VoltageTimesNet 改进模型

### 2.1 设计动机

**问题分析**:
1. 原始 TimesNet 完全依赖 FFT 自动发现周期
2. 电力系统存在**已知的固有周期**（如 1 分钟采样、15 分钟负荷统计等）
3. 纯数据驱动的周期发现可能遗漏先验知识

**改进策略**:
- 将**电网领域知识**与**数据驱动方法**相结合
- 采用混合周期发现机制

### 2.2 电网固有周期

| 周期名称 | 采样点数 (1s采样) | 物理意义 |
|---------|:-----------------:|----------|
| 短期波动 | 60 (1分钟) | 瞬时负荷变化 |
| 中短期 | 300 (5分钟) | 小负荷切换 |
| 中期 | 900 (15分钟) | 统计周期 |
| 长期 | 3600 (1小时) | 负荷日变化 |

### 2.3 混合周期发现策略

**算法设计**:

```python
def hybrid_period_discovery(x, top_k=5, preset_weight=0.3):
    # 1. FFT 动态发现周期
    fft_periods, fft_weights = fft_for_period(x, k=top_k)

    # 2. 预设电网周期
    preset_periods = [60, 300, 900, 3600]  # 采样点数
    preset_weights = [0.3, 0.25, 0.25, 0.2]  # 先验权重

    # 3. 混合融合
    # 70% FFT 动态周期 + 30% 预设周期
    combined_periods = (1 - preset_weight) * fft_periods + \
                       preset_weight * preset_periods
    combined_weights = (1 - preset_weight) * fft_weights + \
                       preset_weight * preset_weights

    return combined_periods, combined_weights
```

**权重分配**:
- FFT 动态发现: 70%
- 预设领域周期: 30%

### 2.4 时域平滑卷积层

**设计目的**: 抑制电压信号中的高频噪声，保留主要趋势

```python
class TemporalSmoothLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 深度可分离卷积实现平滑
        self.smooth = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                     padding=1, groups=channels),  # 深度卷积
            nn.Conv1d(channels, channels, kernel_size=1)   # 点卷积
        )

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.smooth(x)
        return x.transpose(1, 2)  # [B, T, C]
```

### 2.5 VoltageTimesNet 架构图

```
输入 X ∈ R^(B×T×16)  # 16维电压特征
    ↓
TokenEmbedding
    ↓
PositionalEmbedding
    ↓
TemporalSmoothLayer  ← 新增：时域平滑
    ↓
┌─────────────────────────────────────────┐
│        VoltageTimesBlock × N            │
│  ┌─────────────────────────────────────┐│
│  │ 混合周期发现:                       ││
│  │   70% FFT + 30% 预设电网周期        ││
│  │     ↓                               ││
│  │ 1D → 2D 重塑                        ││
│  │     ↓                               ││
│  │ Inception 2D 卷积                   ││
│  │     ↓                               ││
│  │ 2D → 1D 还原                        ││
│  │     ↓                               ││
│  │ 混合权重聚合                        ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
    ↓
LayerNorm + 线性投影
    ↓
输出 Y ∈ R^(B×T×16)
```

---

## 三、TPATimesNet (三相注意力机制)

### 3.1 设计动机

**三相电压特性**:
- 理想情况下，Va, Vb, Vc 相位差 120°
- 三相不平衡是重要的电能质量异常
- 原始 TimesNet 未显式建模三相关系

### 3.2 三相交叉注意力

**注意力计算**:

$$\text{TPA}(V_a, V_b, V_c) = \text{softmax}\left(\frac{Q_a K_b^T}{\sqrt{d_k}}\right) V_c + \text{softmax}\left(\frac{Q_b K_c^T}{\sqrt{d_k}}\right) V_a + \ldots$$

**物理意义**:
- 显式建模三相电压间的相互依赖
- 捕获三相不平衡模式

### 3.3 可学习相位偏置

```python
class PhaseAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 可学习的 120° 相位偏置
        self.phase_bias = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.phase_bias, std=0.02)

    def forward(self, x):
        # x: [B, T, 3, d_model] 三相特征
        # 添加相位偏置
        x = x + self.phase_bias.unsqueeze(0).unsqueeze(0)
        return x
```

---

## 四、MTSTimesNet (多尺度时序建模)

### 4.1 设计动机

**时序多尺度特性**:
- 短期 (秒级): 瞬时波动、骤降骤升
- 中期 (分钟级): 负荷切换引起的变化
- 长期 (小时级): 日负荷曲线变化

### 4.2 多尺度分解架构

```
输入序列
    │
    ├──────────────────────────────────────┐
    │                                      │
    ↓                                      ↓
短期分支 (小窗口)                    长期分支 (大窗口)
seq_len=50                          seq_len=200
    │                                      │
    ↓                                      ↓
TimesBlock_short                    TimesBlock_long
    │                                      │
    ↓                                      ↓
短期特征                             长期特征
    │                                      │
    └──────────────┬───────────────────────┘
                   ↓
              特征融合层
                   ↓
              输出预测
```

### 4.3 跨尺度残差连接

```python
class MultiScaleTimesNet(nn.Module):
    def __init__(self):
        self.short_branch = TimesBlock(seq_len=50)
        self.medium_branch = TimesBlock(seq_len=100)
        self.long_branch = TimesBlock(seq_len=200)
        self.fusion = nn.Linear(3 * d_model, d_model)

    def forward(self, x):
        # 多尺度特征提取
        short_feat = self.short_branch(x[:, -50:])
        medium_feat = self.medium_branch(x[:, -100:])
        long_feat = self.long_branch(x)

        # 特征对齐与融合
        combined = torch.cat([short_feat, medium_feat, long_feat], dim=-1)
        output = self.fusion(combined)

        return output
```

---

## 五、基于重构误差的异常检测框架

### 5.1 训练阶段

**目标**: 学习正常数据的时序模式

**损失函数**:
$$\mathcal{L} = \frac{1}{T \times C} \sum_{t=1}^{T} \sum_{c=1}^{C} (x_{t,c} - \hat{x}_{t,c})^2$$

### 5.2 检测阶段

**重构误差计算**:
$$e_t = \frac{1}{C} \sum_{c=1}^{C} (x_{t,c} - \hat{x}_{t,c})^2$$

**阈值设定**:
$$\theta = \text{percentile}(E_{train}, 100 - \alpha)$$

其中 $\alpha$ 是预期异常比例。

**异常判定**:
$$\hat{y}_t = \begin{cases} 1 & \text{if } e_t > \theta \\ 0 & \text{otherwise} \end{cases}$$

### 5.3 点调整评估策略

**问题**: 时序异常通常连续发生，逐点评估可能低估性能

**点调整规则**: 如果异常段内任意一点被正确检测，则整段标记为正确检测

```python
def point_adjustment(pred, gt):
    # 找到所有异常段
    anomaly_segments = find_contiguous_regions(gt == 1)

    adjusted_pred = pred.copy()
    for start, end in anomaly_segments:
        # 如果该段内有任何正确检测
        if np.any(pred[start:end] == 1):
            adjusted_pred[start:end] = 1

    return adjusted_pred
```

---

## 六、论文写作建议

### 6.1 第三章结构建议

```
3.1 时序异常检测方法概述
    3.1.1 异常检测问题形式化
    3.1.2 基于重构的异常检测原理
    3.1.3 深度学习时序模型演进

3.2 TimesNet 模型原理深度解析
    3.2.1 模型整体架构
    3.2.2 FFT 周期发现机制
    3.2.3 1D-to-2D 时序转换
    3.2.4 Inception 2D 卷积特征提取
    3.2.5 自适应多周期聚合

3.3 面向电压检测的 TimesNet 改进方法
    3.3.1 VoltageTimesNet: 领域知识引导的周期发现
    3.3.2 TPATimesNet: 三相注意力机制
    3.3.3 MTSTimesNet: 多尺度时序建模

3.4 基于重构误差的异常检测框架
    3.4.1 重构误差计算
    3.4.2 动态阈值设定
    3.4.3 点调整评估策略

3.5 模型训练与优化策略

3.6 本章小结
```

### 6.2 公式排版规范

- 重要公式单独成行，使用 `equation` 环境
- 算法步骤使用 `algorithm` 环境
- 变量使用斜体，矩阵使用粗体

### 6.3 图表建议

| 图表 | 位置 | 内容 |
|------|------|------|
| 图3-1 | 3.2.1 | TimesNet 整体架构图 |
| 图3-2 | 3.2.3 | 1D→2D 转换示意图 |
| 图3-3 | 3.3.1 | VoltageTimesNet 架构对比图 |
| 图3-4 | 3.4 | 异常检测流程图 |
| 表3-1 | 3.3.1 | 电网固有周期定义 |
| 算法1 | 3.2.2 | FFT 周期发现算法 |

---

*文档最后更新: 2026-01-26*
