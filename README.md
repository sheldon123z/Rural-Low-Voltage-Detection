# 农村低压配电网电压异常检测

基于 TimesNet 的时间序列异常检测方法，针对农村低压配电网电压质量问题进行研究。

> 北京林业大学本科毕业论文项目

## 在线资源

| 资源 | 链接 |
|------|------|
| 交互演示 | [HuggingFace Space](https://huggingface.co/spaces/Sheldon123z/rural-voltage-demo) |
| 项目主页 | [GitHub Pages](https://sheldon123z.github.io/Rural-Low-Voltage-Detection/) |
| 预训练模型 | [Sheldon123z/rural-voltage-detection-models](https://huggingface.co/Sheldon123z/rural-voltage-detection-models) |
| 数据集 | [Sheldon123z/rural-voltage-datasets](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets) |

## 实验结果

### RuralVoltage 数据集

| 排名 | 模型 | Precision | Recall | F1 | Accuracy |
|:----:|------|:---------:|:------:|:--:|:--------:|
| 1 | **DLinear** | 0.7936 | **0.9837** | **0.8785** | **0.9599** |
| 2 | **VoltageTimesNet_v2** | 0.7371 | 0.9110 | 0.8149 | 0.9393 |
| 3 | TPATimesNet | **0.8983** | 0.4723 | 0.6191 | 0.8303 |
| 4 | TimesNet | 0.8939 | 0.4582 | 0.6059 | 0.8259 |
| 5 | MTSTimesNet | 0.8509 | 0.3251 | 0.4705 | 0.7863 |
| 6 | VoltageTimesNet | 0.8636 | 0.1906 | 0.3123 | 0.7549 |

*VoltageTimesNet_v2 经 Optuna 30-trial TPE 超参搜索优化 (Trial 10)*

### PSM 数据集

| 排名 | 模型 | Precision | Recall | F1 | Accuracy |
|:----:|------|:---------:|:------:|:--:|:--------:|
| 1 | TimesNet | 0.9848 | 0.9625 | 0.9735 | 0.9855 |
| 2 | VoltageTimesNet | 0.9840 | 0.9625 | 0.9731 | 0.9853 |
| 3 | TPATimesNet | 0.9844 | 0.9612 | 0.9727 | 0.9850 |
| 4 | DLinear | 0.9864 | 0.9466 | 0.9661 | 0.9816 |
| 5 | MTSTimesNet | 0.9756 | 0.9506 | 0.9629 | 0.9797 |

## 模型

| 模型 | 核心创新 | 来源 |
|------|---------|------|
| **TimesNet** | FFT 周期发现 + 2D 卷积 | Wu et al., ICLR 2023 |
| **VoltageTimesNet** | 预设电网周期 + FFT 混合 | 本研究 |
| **VoltageTimesNet_v2** | 可学习权重 + 召回率优化 | 本研究 |
| **TPATimesNet** | 三相交叉注意力机制 | 本研究 |
| **MTSTimesNet** | 多尺度并行 + 自适应融合 | 本研究 |
| **DLinear** | 趋势-季节性分解线性模型 | Zeng et al., 2022 |

另支持 Autoformer、Informer、PatchTST、iTransformer、Reformer 等 15+ 种模型。

## 快速开始

```bash
# 环境配置
conda create -n tslib python=3.11
conda activate tslib
pip install -r code/requirements.txt

# 训练 VoltageTimesNet_v2 (推荐)
cd code
python run.py --is_training 1 --model VoltageTimesNet_v2 --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 --anomaly_ratio 3.0

# 训练 TimesNet (PSM 基准)
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64

# 批量对比实验
bash scripts/PSM/run_comparison.sh
```

### 下载预训练模型

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="Sheldon123z/rural-voltage-detection-models",
    filename="RuralVoltage/VoltageTimesNet_v2_sl100_dm64/checkpoint.pth"
)
```

## 项目结构

```
code/
├── run.py                      # 主入口
├── exp/                        # 训练/测试逻辑
├── models/                     # 20+ 模型实现
│   ├── TimesNet.py             # FFT + 2D 卷积
│   ├── VoltageTimesNet.py      # 预设周期 + FFT 混合
│   ├── VoltageTimesNet_v2.py   # 召回率优化版
│   ├── TPATimesNet.py          # 三相注意力
│   ├── MTSTimesNet.py          # 多尺度时序
│   └── DLinear.py              # 轻量级基线
├── data_provider/              # 数据加载
├── layers/                     # 网络组件
├── scripts/                    # 训练脚本
├── tests/                      # 冒烟测试
└── dataset/                    # 数据集
```

## 数据集

| 数据集 | 特征数 | 训练集 | 测试集 | 领域 |
|--------|:------:|-------:|-------:|------|
| **RuralVoltage** | 16 | 50,000 | 10,000 | 农村电压 |
| PSM | 25 | 132,481 | 87,841 | 服务器监控 |
| MSL | 55 | 58,317 | 73,729 | NASA 航天器 |
| SMAP | 25 | 135,183 | 427,617 | NASA 航天器 |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--seq_len` | 100 | 输入序列长度 |
| `--d_model` | 64 | 隐藏维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | TimesNet top-k 周期数 |
| `--enc_in` | 25 | 输入特征数 (RuralVoltage=16) |
| `--anomaly_ratio` | 1.0 | 预期异常比例 (%) |

## 技术栈

PyTorch 2.x / Time-Series-Library / Plotly / Gradio / HuggingFace

## 引用

```bibtex
@inproceedings{timesnet2023,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICLR},
  year={2023}
}
```

## License

本项目基于 [Time-Series-Library](https://github.com/thuml/time-series-library) 框架开发。
