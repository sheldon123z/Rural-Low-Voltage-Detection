# 数据集目录

请将您的数据集放置在此目录下。

## 支持的数据集

### PSM（服务器池指标数据集）
```
PSM/
├── train.csv       # 训练数据
├── test.csv        # 测试数据
└── test_label.csv  # 测试标签
```

### MSL（火星科学实验室数据集）
```
MSL/
├── MSL_train.npy      # 训练数据
├── MSL_test.npy       # 测试数据
└── MSL_test_label.npy # 测试标签
```

### SMAP（土壤水分主被动探测数据集）
```
SMAP/
├── SMAP_train.npy      # 训练数据
├── SMAP_test.npy       # 测试数据
└── SMAP_test_label.npy # 测试标签
```

### SMD（服务器设备数据集）
```
SMD/
├── SMD_train.npy      # 训练数据
├── SMD_test.npy       # 测试数据
└── SMD_test_label.npy # 测试标签
```

### SWAT（安全水处理数据集）
```
SWAT/
├── swat_train2.csv  # 训练数据
└── swat2.csv        # 测试数据
```

### RuralVoltage（自定义农村电网数据集）
```
RuralVoltage/
├── train.csv      # 正常运行数据
├── test.csv       # 包含异常的测试数据
└── test_label.csv # 异常标签

# CSV 格式说明：
# 时间戳, Va, Vb, Vc, Ia, Ib, Ic, P, Q, S, PF, THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance
#
# 各字段说明：
# - Va, Vb, Vc: 三相电压（V）
# - Ia, Ib, Ic: 三相电流（A）
# - P: 有功功率（W）
# - Q: 无功功率（Var）
# - S: 视在功率（VA）
# - PF: 功率因数
# - THD_Va, THD_Vb, THD_Vc: 三相电压总谐波畸变率（%）
# - Freq: 频率（Hz）
# - V_unbalance: 电压不平衡因子（%）
# - I_unbalance: 电流不平衡因子（%）
```

## 标签映射（适用于 RuralVoltage 数据集）

| 标签值 | 异常类型 | 说明 |
|--------|----------|------|
| 0 | 正常 | 电压在正常范围内 |
| 1 | 欠压 | 电压 < 198V（低于额定电压 10%） |
| 2 | 过压 | 电压 > 242V（高于额定电压 10%） |
| 3 | 电压骤降 | 短时电压大幅下降 |
| 4 | 谐波畸变 | 总谐波畸变率 THD > 5% |
| 5 | 三相不平衡 | 不平衡因子 > 4% |

## 数据集来源

- **PSM/MSL/SMAP/SMD/SWAT**：均为公开的时序异常检测基准数据集，详见 Time-Series-Library 项目
- **RuralVoltage**：本项目自定义的农村低压配电网电压异常检测数据集

## 数据格式要求

所有数据集需满足以下格式：
1. 训练数据不含标签列（纯特征数据）
2. 测试数据不含标签列（纯特征数据）
3. 测试标签文件为单列，0 表示正常，非 0 表示异常
