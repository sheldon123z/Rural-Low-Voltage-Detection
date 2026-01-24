# 农村低压配电网电压异常检测研究

基于深度学习的农村低压配电网电压异常检测研究。本项目采用 TimesNet 及其变体模型，实现对电压异常的精准检测。

## 项目亮点

- **5 种 TimesNet 变体模型**：针对电压信号特点设计的创新模型
- **15 种深度学习模型**：全面的模型对比基准
- **完整实验框架**：基于清华 Time-Series-Library 构建
- **农村电压数据集**：17 维特征，5 种异常类型

## 项目结构

```
Rural-Low-Voltage-Detection/
├── code/voltage_anomaly_detection/     # 核心代码
│   ├── models/                         # 15+ 深度学习模型
│   │   ├── TimesNet.py                 # 基础 TimesNet
│   │   ├── VoltageTimesNet.py          # 电压优化版
│   │   ├── TPATimesNet.py              # 三相注意力版
│   │   ├── MTSTimesNet.py              # 多尺度时序版
│   │   └── HybridTimesNet.py           # 混合周期发现版
│   ├── exp/                            # 实验类
│   ├── data_provider/                  # 数据加载
│   ├── scripts/                        # 训练脚本
│   └── results/                        # 实验结果
├── thesis/                             # 论文 (北林模板)
├── resources/                          # 研究资料
├── CLAUDE.md                           # Claude Code 指南
└── README.md
```

## TimesNet 模型家族

| 模型 | 核心创新 | 适用场景 |
|------|---------|---------|
| **TimesNet** | FFT 周期发现 + 2D 卷积 | 通用时序异常检测 |
| **VoltageTimesNet** | 预设电网周期 + FFT 混合 | 电力系统电压监测 |
| **TPATimesNet** | 三相交叉注意力机制 | 三相不平衡异常检测 |
| **MTSTimesNet** | 多尺度并行 + 自适应融合 | 复杂多尺度异常 |
| **HybridTimesNet** | 置信度融合周期发现 | 鲁棒周期检测 |

## 快速开始

### 环境配置

```bash
conda create -n tslib python=3.11
conda activate tslib
pip install -r code/voltage_anomaly_detection/requirements.txt
```

### 训练模型

```bash
cd code/voltage_anomaly_detection

# 使用 TimesNet 训练 PSM 数据集
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64

# 使用 VoltageTimesNet 训练农村电压数据集
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ --enc_in 17 --c_out 17

# 运行模型对比实验
bash scripts/compare_timesnet_psm.sh
```

### 分析实验结果

```bash
python scripts/analyze_comparison_results.py --result_dir ./results/psm_comparison
```

## 数据集

### 标准数据集
- **PSM**: 服务器机器数据集 (25 特征)
- **MSL/SMAP**: NASA 航天器数据集
- **SMD**: 服务器机器数据集
- **SWAT**: 安全水处理数据集

### 农村电压数据集 (RuralVoltage)

| 特征类别 | 特征 | 说明 |
|---------|------|-----|
| 三相电压 | Va, Vb, Vc | 200-240V |
| 三相电流 | Ia, Ib, Ic | 10-20A |
| 功率指标 | P, Q, S, PF | 有功/无功/视在功率 |
| 电能质量 | THD_Va/Vb/Vc | 谐波失真率 |
| 不平衡因子 | V/I_unbalance | 三相不平衡度 |
| 频率 | Freq | 50Hz 标准 |

**异常类型**: 欠压、过压、电压骤降、谐波畸变、三相不平衡

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet top-k 周期数 |
| `--anomaly_ratio` | 1.0 | 预期异常比例 (%) |

## 技术栈

- **框架**: PyTorch 2.x + CUDA
- **基础库**: Time-Series-Library (清华)
- **可视化**: Matplotlib, Seaborn
- **论文模板**: BJFUThesis (北京林业大学)

## 论文结构

- 第一章: 绪论
- 第二章: 数据采集与预处理
- 第三章: 基于 TimesNet 的电压异常检测算法
- 第四章: 实验设计与结果分析
- 第五章: 农村电网低电压监管平台设计
- 第六章: 结论与展望

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@thesis{rural_voltage_detection,
  title={基于深度学习的农村低压配电网电压异常检测研究},
  author={郑晓东},
  school={北京林业大学},
  year={2025}
}
```

## 许可证

本项目仅供学术研究使用。
