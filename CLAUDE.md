# CLAUDE.md

农村低压配电网电压异常检测项目。核心代码位于 `code/`，基于 Time-Series-Library 框架。

## 常用命令

```bash
cd code

# 训练 TimesNet
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 --train_epochs 10

# 训练 VoltageTimesNet（农村电压数据）
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 --seq_len 100

# 训练 VoltageTimesNet_v2（召回率优化版，推荐）
python run.py --is_training 1 --model VoltageTimesNet_v2 --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 --anomaly_ratio 3.0

# 训练 Kaggle 电力质量数据集
python run.py --is_training 1 --model TimesNet --data KagglePQ \
  --root_path ./dataset/Kaggle_PowerQuality_2/ --enc_in 128 --c_out 128 --seq_len 64

# 仅测试
python run.py --is_training 0 --model TimesNet --data PSM --root_path ./dataset/PSM/

# 批量对比实验
bash scripts/PSM/run_comparison.sh
```

## 代码结构

```
code/
├── run.py                          # 主入口
├── exp/exp_anomaly_detection.py    # 训练/测试逻辑
├── models/                         # 模型实现
│   ├── TimesNet.py                 # 核心：FFT + 2D卷积
│   ├── VoltageTimesNet.py          # 预设周期 + FFT 混合
│   ├── VoltageTimesNet_v2.py       # 召回率优化版（新增）
│   ├── TPATimesNet.py              # 三相注意力
│   ├── MTSTimesNet.py              # 多尺度时序
│   └── DLinear.py                  # 轻量级基线
├── data_provider/data_loader.py    # 数据加载
├── layers/                         # 网络组件
└── scripts/                        # 训练脚本
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet 周期数 |
| `--enc_in` | 25 | 输入特征数（RuralVoltage=16） |
| `--num_workers` | 10 | 数据加载线程（卡死时设为 0） |

## 数据集

| 数据集 | 特征数 | 训练集 | 测试集 | 说明 |
|--------|:------:|-------:|-------:|------|
| PSM | 25 | 132,481 | 87,841 | 服务器指标 |
| RuralVoltage/realistic_v2 | 16 | 50,000 | 10,000 | 农村电压（改进版） |
| KagglePQ | 128 | 2,400 | 9,598 | 电力质量波形（新增） |
| MSL | 55 | 58,317 | 73,729 | NASA 航天器 |
| SMAP | 25 | 135,183 | 427,617 | NASA 航天器 |

## TimesNet 原理

```python
# 1. FFT 发现周期
xf = torch.fft.rfft(x, dim=1)
period = T // frequency_list.topk(k).indices

# 2. 1D → 2D 重塑
x_2d = x.reshape(B, period, T//period, C)

# 3. Inception 2D 卷积
x_2d = Inception_Block_V1(x_2d)

# 4. 多周期加权融合
output = Σ(softmax(weights) × x_1d_i)
```

**异常检测**：学习正常模式，重构误差超阈值即判定异常。

## 添加新模型

1. 在 `models/` 创建 `NewModel.py`，实现 `Model` 类
2. 在 `models/__init__.py` 的 `model_dict` 中注册
3. `forward` 返回维度需与输入相同

## 实验结果

结果保存：`results/PSM_comparison_YYYYMMDD_HHMMSS/`

分析脚本：
```bash
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_XXXXXX
```

## 论文材料

论文写作材料位于 `code/docs/`：
- `实验结果汇总_论文写作材料.md` - 实验数据
- `TimesNet算法描述_论文写作材料.md` - 模型原理
- `项目状态报告_20260126.md` - 进度跟踪
