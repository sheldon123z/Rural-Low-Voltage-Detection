# Pecan Street Dataport 数据集申请指南

## 关于 Pecan Street

Pecan Street Inc. 是一家非营利研究组织，运营着世界上最大的住宅能源和水数据研究网络。
Dataport 是其数据门户，提供高分辨率（1秒/1分钟级）电力数据。

**官网**: https://www.pecanstreet.org/dataport/

## 数据集特点

- **高时间分辨率**: 1秒/15秒/1分钟间隔数据
- **多维度数据**: 电压、电流、功率、功率因数等
- **长时间跨度**: 多年连续记录
- **真实家庭数据**: 来自美国德克萨斯州 Austin 等地的真实住宅

## 申请流程

### Step 1: 注册 Dataport 账号

1. 访问 https://www.pecanstreet.org/dataport/
2. 点击 **Register** 或 **Request Access**
3. 选择 **Academic/Research** 类别

### Step 2: 选择访问级别

| 级别 | 费用 | 数据范围 | 适用对象 |
|------|------|----------|----------|
| **Free Tier** | 免费 | 样本数据 | 初步评估 |
| **Academic** | 免费/低价 | 完整数据 | 学术研究 |
| **Commercial** | 付费 | 完整数据 | 商业用途 |

**推荐**: 申请 Academic 级别（学术研究免费或低成本）

### Step 3: 提交研究计划

---

## 申请模板（英文）

### Research Project Description

```
Project Title: Deep Learning for Voltage Anomaly Detection in Low-Voltage
Distribution Networks

Research Institution: [Your Institution]
Principal Investigator: [Your Name]
Email: [Your Academic Email]
Duration: [Expected Duration]

Project Description:
This research project focuses on developing advanced deep learning methods for
detecting voltage anomalies in distribution networks. We are specifically
interested in the high-resolution voltage data from Pecan Street Dataport to:

1. Study real-world voltage fluctuation patterns in residential settings
2. Train and validate TimesNet-based anomaly detection models
3. Compare synthetic data performance with real-world data
4. Analyze power quality disturbances in residential environments

The data will be used exclusively for academic research and will contribute to
improving power quality monitoring in distribution networks.
```

### Data Requirements

```
Requested Data Types:
- Voltage measurements (Va, Vb, Vc for three-phase where available)
- Current measurements
- Power factor
- Frequency
- Timestamps at 1-second or 1-minute resolution

Requested Time Period:
- Minimum 1 year of continuous data preferred
- Any available historical data for voltage anomaly events

Requested Coverage:
- Multiple households to ensure diversity
- Both normal operation and any recorded anomaly events
```

### Intended Use Statement

```
The Pecan Street data will be used for:

1. Academic Research
   - Training deep learning models for voltage anomaly detection
   - Publishing findings in peer-reviewed journals
   - Thesis/dissertation research

2. Data Usage Commitments
   - Data will NOT be redistributed to third parties
   - Data will NOT be used for commercial purposes
   - Individual household identities will remain anonymized
   - All publications will acknowledge Pecan Street Dataport

3. Expected Outcomes
   - Improved anomaly detection algorithms
   - Academic publications
   - Open-source model code (without data)
```

---

## 申请模板（中文，供参考）

### 研究项目描述

项目名称：低压配电网电压异常检测的深度学习方法

研究机构：[机构名称]
负责人：[姓名]
邮箱：[学术邮箱]
研究周期：[预计周期]

项目描述：
本研究项目专注于开发先进的深度学习方法来检测配电网中的电压异常。
我们特别对 Pecan Street Dataport 的高分辨率电压数据感兴趣，用于：

1. 研究住宅环境中真实的电压波动模式
2. 训练和验证基于 TimesNet 的异常检测模型
3. 比较合成数据与真实数据的性能差异
4. 分析住宅环境中的电能质量扰动

数据将仅用于学术研究，并有助于改进配电网的电能质量监测。

---

## 数据下载方式

### 在线查询

1. 登录 Dataport: https://dataport.pecanstreet.org/
2. 使用 SQL 查询界面选择数据
3. 导出为 CSV 格式

### API 访问（高级用户）

```python
# Pecan Street API 示例
import requests

API_KEY = "your_api_key"
BASE_URL = "https://api.pecanstreet.org/dataport"

params = {
    "dataid": "home_id",
    "time_resolution": "1min",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
}

response = requests.get(BASE_URL, params=params,
                       headers={"Authorization": f"Bearer {API_KEY}"})
```

---

## 注意事项

1. **审核时间**: 学术申请通常需要 1-2 周审核
2. **数据协议**: 需签署数据使用协议（DUA）
3. **引用要求**: 发表论文时需引用 Pecan Street
4. **隐私保护**: 严禁尝试识别具体家庭身份

## 引用格式

```bibtex
@misc{pecanstreet,
  author = {{Pecan Street Inc.}},
  title = {Dataport},
  year = {2024},
  url = {https://www.pecanstreet.org/dataport/}
}
```

## 替代方案

如果 Pecan Street 申请周期较长，可考虑：

1. **REFIT Dataset** (UK): https://www.refitsmarthomes.org/
2. **UK-DALE**: https://jack-kelly.com/data/
3. **REDD**: http://redd.csail.mit.edu/
