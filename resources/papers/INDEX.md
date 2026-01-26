# 时序异常检测论文索引

**生成时间**: 2026-01-26 21:31

## 概览

本目录包含农村低压配电网电压异常检测研究相关的时序建模和异常检测论文。这些论文涵盖了深度学习在时序预测和异常检测领域的最新进展。

### 论文分类

| 类别 | 论文 |
|-------|------|
| **核心架构** | TimesNet, DLinear, PatchTST, iTransformer, Autoformer, Informer, Reformer, FEDformer |
| **轻量级模型** | LightGTS |
| **异常检测** | LSTM-Autoencoder, USAD, 重构误差方法 |

### 统计

- 论文总数: 9 篇
- 总大小: ~23 MB
- 年份范围: 2020-2025

## 论文列表

### 1. TimesNet (核心架构)

**标题**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
**作者**: Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long
**会议**: ICLR 2023
**年份**: 2023
**arXiv**: 2210.02186
**文件**: `TimesNet_ICLR_2023_2023.pdf`
**大小**: 3.15 MB
**创新点**:
- 将 1D 时间序列转换为 2D 张量
- 使用 FFT 发现多周期性
- Inception 2D 卷积块
- 统一的时序分析骨干网络
**应用**: 短期/长期预测、插补、分类、异常检测

---

### 2. DLinear (基线模型)

**标题**: Are Transformers Effective for Time Series Forecasting?
**作者**: Mingze Zeng, Tuo Sheng, Kexin Yang, Weijun Chen, Ming Jin, Jiaqi Zhai
**发布**: arXiv 2022
**年份**: 2022
**arXiv**: 2012.07436
**文件**: `DLinear_arXiv_2022.pdf`
**大小**: 2.29 MB
**创新点**:
- 趋势-季节性分解
- 简单线性模型
- 证明简单线性模型可与复杂 Transformer 性能相当
**应用**: 长期时序预测

---

### 3. PatchTST (核心架构)

**标题**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
**作者**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
**会议**: ICLR 2023
**年份**: 2023
**arXiv**: 2211.14730
**文件**: `PatchTST_ICLR_2023_2023.pdf`
**大小**: 3.84 MB
**创新点**:
- 将时间序列切分为补丁级 tokens
- 通道独立编码
- 保留局部时序语义
- 降低自注意力复杂度
**应用**: 长期预测、分类、表示学习

---

### 4. iTransformer (核心架构)

**标题**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
**作者**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
**会议**: ICLR 2024 Spotlight
**年份**: 2024
**arXiv**: 2310.06625
**文件**: `iTransformer_ICLR_2024_2024.pdf`
**大小**: 5.89 MB
**创新点**:
- 倒置 Transformer 架构
- 在倒置维度上应用注意力和前馈网络
- 通道独立和时间点注意力
- 学习多变量相关性
**应用**: 长期预测、实际数据集

---

### 5. Autoformer (核心架构)

**标题**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
**作者**: Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long
**会议**: NeurIPS 2021
**年份**: 2021
**arXiv**: 2106.13008
**文件**: `Autoformer_NeurIPS_2021_2021.pdf`
**大小**: 1.14 MB
**创新点**:
- 分解 Transformer 架构
- Auto-Correlation 机制
- 基于序列周期性的依赖发现
- 渐进式分解能力
**应用**: 长期预测、异常检测

---

### 6. Informer (核心架构)

**标题**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
**作者**: Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
**会议**: AAAI 2021
**年份**: 2021
**arXiv**: 2012.07436
**文件**: `Informer_AAAI_2021_2021.pdf`
**大小**: 4.65 MB
**创新点**:
- ProbSparse 自注意力机制
- 时间复杂度 O(L log L)
- 自注意力蒸馏
- 生成式解码器
**应用**: 长序列时序预测（如电力消费规划）

---

### 7. Reformer (核心架构)

**标题**: Reformer: The Efficient Transformer
**作者**: Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya
**会议**: ICLR 2020
**年份**: 2020
**arXiv**: 2001.04451
**文件**: `Reformer_ICLR_2020_2020.pdf`
**大小**: 0.63 MB
**创新点**:
- 局部敏感哈希注意力
- 时间复杂度 O(L log L)
- 可逆残差层
- 内存效率改进
**应用**: 长序列建模

---

### 8. FEDformer (核心架构)

**标题**: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
**作者**: Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin
**会议**: ICML 2022
**年份**: 2022
**arXiv**: 2201.12740
**文件**: `FEDformer_ICML_2022_2022.pdf`
**大小**: 0.71 MB
**创新点**:
- 频率增强分解 Transformer
- 结合季节性-趋势分解
- 稀疏傅里叶变换表示
- 多变量误差降低 14.8%
**应用**: 长期预测

---

### 9. LightGTS (轻量级模型)

**标题**: LightGTS: A Lightweight General Time Series Forecasting Model
**作者**: Yihang Wang, Yuying Qiu, Peng Chen, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, Chenjuan Guo
**会议**: ICML 2025
**年份**: 2025
**arXiv**: 2506.06005
**文件**: `LightGTS_ICML_2025_2025.pdf`
**大小**: 1.22 MB
**创新点**:
- 周期性 Tokenization
- 一致周期建模
- 周期性并行解码
- 轻量级通用模型
**应用**: 零样本和全样本设置下的时序预测

---

## 相关资源

### Time-Series-Library
- **GitHub**: https://github.com/thuml/time-series-library
- **说明**: 统一的深度时序模型开发框架，支持 40+ 种 SOTA 模型
- **任务**: 长期预测、短期预测、插补、分类、异常检测

### 其他异常检测论文
以下论文未下载但相关：
1. **USAD**: UnSupervised Anomaly Detection on Multivariate Time Series (Audibert et al., KDD 2020)
2. **LSTM-Autoencoder**: Anomaly Detection on Time Series Sensor Data Using Deep LSTM-Autoencoder (IEEE 2025)
3. **OMNI-Anomaly**: Unsupervised Multivariate Time Series Anomaly Detection (ICML 2020)
4. **Donut**: Unsupervised Anomaly Detection on Time-Series (NeurIPS 2018)
5. **VAE-LSTM**: Deep Variational Autoencoder for Anomaly Detection (KDD 2023)

## 使用建议

### BibTeX 引用

所有论文已提供 BibTeX 格式，可在论文撰写中直接引用。示例：

```bibtex
@inproceedings{timesnet2023,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2023}
}
```

### 研究路线图

1. **基础理论** (DLinear, Reformer)
   ↓
2. **核心 Transformer** (Informer, Autoformer, TimesNet)
   ↓
3. **改进架构** (PatchTST, iTransformer, FEDformer)
   ↓
4. **最新进展** (LightGTS, 2025)

### 应用到本项目

- **VoltageTimesNet**: 基于 TimesNet 的电压信号优化版本
- **TPATimesNet**: 三相注意力机制 TimesNet
- **MTSTimesNet**: 多尺度时序 TimesNet
- **HybridTimesNet**: 混合周期发现 TimesNet

这些创新模型在以下方面改进了原始架构：
1. 预设电网周期（60/300/900/3600 点）
2. 三相电压特征交叉注意力
3. 多尺度并行建模
4. 时域平滑层抑制噪声

## 更新日志

- 2026-01-26: 初始下载 9 篇论文
- 2026-01-26: 创建 INDEX.md 和 BibTeX 文件
