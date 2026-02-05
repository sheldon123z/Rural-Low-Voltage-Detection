# CLAUDE.md

农村低压配电网电压异常检测项目。核心代码位于 `code/`，基于 Time-Series-Library 框架。

## 论文命名规范

**重要**：论文中统一使用以下命名：
- 代码中的 `VoltageTimesNet_v2` 在论文中统一称为 **VoltageTimesNet**
- **不要提及** DLinear 模型（论文对比实验不包含）
- **不要提及** 旧版 VoltageTimesNet（代码中的 `VoltageTimesNet.py`，论文中不存在 v1/v2 区分）
- 对比模型仅包含：VoltageTimesNet、TimesNet、LSTMAutoEncoder、Isolation Forest、One-Class SVM

## 论文主模型：VoltageTimesNet

**VoltageTimesNet**（代码中为 VoltageTimesNet_v2）是本论文的核心模型，经 Optuna 30-trial 超参数优化后在 RuralVoltage 数据集上取得最优性能：

| 指标 | 值 | 说明 |
|------|:--:|------|
| **F1** | **0.8149** | 综合性能最优 |
| **Recall** | **0.9110** | 检出 91.1% 的异常事件 |
| **Precision** | **0.7371** | 误报率可控 |
| **Accuracy** | **0.9393** | 整体准确率 93.9% |

最优参数: `d_model=128, e_layers=3, d_ff=256, seq_len=50, top_k=2, num_kernels=8, lr=0.004188, batch_size=32, dropout=0.017, anomaly_ratio=2.085`

模型权重: `code/newest_models/best_voltagetimesnet_v2.pth`
配置文件: `code/newest_models/best_model_config.json`

## 在线资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 数据集 | [Sheldon123z/rural-voltage-datasets](https://huggingface.co/datasets/Sheldon123z/rural-voltage-datasets) | 包含 RuralVoltage、PSM 等数据集 |
| 模型检查点 | [Sheldon123z/rural-voltage-detection-models](https://huggingface.co/Sheldon123z/rural-voltage-detection-models) | 训练好的模型权重 |
| 代码仓库 | [GitHub](https://github.com/sheldon123z/Rural-Low-Voltage-Detection) | 源代码 |

## 常用命令

```bash
cd code

# 训练 TimesNet
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 --train_epochs 10

# 训练 VoltageTimesNet（农村电压数据）
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 --seq_len 100

# 训练 VoltageTimesNet_v2（Optuna 优化参数，论文主模型）
python run.py --is_training 1 --model VoltageTimesNet_v2 --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/realistic_v2/ --enc_in 16 --c_out 16 \
  --seq_len 50 --d_model 128 --e_layers 3 --d_ff 256 --top_k 2 --num_kernels 8 \
  --learning_rate 0.004188 --batch_size 32 --dropout 0.017 --anomaly_ratio 2.085

# 训练 Kaggle 电力质量数据集
python run.py --is_training 1 --model TimesNet --data KagglePQ \
  --root_path ./dataset/Kaggle_PowerQuality_2/ --enc_in 128 --c_out 128 --seq_len 64

# 仅测试
python run.py --is_training 0 --model TimesNet --data PSM --root_path ./dataset/PSM/

# 批量对比实验
bash scripts/PSM/run_comparison.sh
```

## 下载预训练模型

```python
from huggingface_hub import hf_hub_download

# 下载最优模型 (VoltageTimesNet_v2)
checkpoint = hf_hub_download(
    repo_id="Sheldon123z/rural-voltage-detection-models",
    filename="RuralVoltage/VoltageTimesNet_v2_sl100_dm64/checkpoint.pth"
)
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
| RuralVoltage/realistic_v2 | 16 | 50,000 | 10,000 | 农村电压（14.6%异常） |
| KagglePQ | 128 | 2,400 | 9,598 | 电力质量波形 |
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

## 最新实验结果 (2026-02-03)

### RuralVoltage 数据集（论文主实验）

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|------|:------:|:------:|:------:|:------:|
| **VoltageTimesNet** | **0.9393** | **0.7371** | **0.9110** | **0.8149** |
| TimesNet | 0.8584 | 0.5143 | 0.7115 | 0.5970 |
| LSTMAutoEncoder | 0.7905 | 0.3654 | 0.5712 | 0.4457 |
| Isolation Forest | 0.3474 | 0.3474 | 1.0000 | 0.5157 |
| One-Class SVM | 0.3474 | 0.3474 | 1.0000 | 0.5157 |

### Optuna 超参数搜索

搜索记录: `code/results/optuna/search_progress.md`
最佳配置: `code/newest_models/best_model_config.json`

结果保存：`results/PSM_comparison_YYYYMMDD_HHMMSS/`

分析脚本：
```bash
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_XXXXXX
```

## 论文材料

论文写作材料位于 `code/docs/`：
- `实验结果汇总_论文写作材料.md` - 核心实验数据（最新）
- `README_论文写作材料索引.md` - 所有写作材料索引
- `TimesNet算法描述_论文写作材料.md` - 模型原理
- `VoltageTimesNet_v2_results.md` - V2模型详细结果
- `supplementary_experiment_report.md` - 补充实验报告

### 论文图表

**统一图表管理**：所有图表脚本位于 `code/figures/scripts/`，输出到 `code/figures/output/chapX/`

#### 目录结构
```
code/figures/
├── scripts/                      # 图表生成脚本
│   ├── thesis_style.py           # 统一样式配置
│   ├── generate_all_figures.py   # 批量生成
│   ├── sync_to_thesis.py         # 同步到论文项目
│   ├── chap2/                    # 第2章脚本
│   │   ├── fig_2_1_*.py
│   │   └── fig_2_2_*.py
│   ├── chap3/                    # 第3章脚本
│   │   ├── fig_3_1_*.py
│   │   └── ...
│   └── chap4/                    # 第4章脚本 (15个)
│       ├── fig_4_1_*.py
│       └── ...
├── output/                       # 生成的图片
│   ├── chap2/                    # 第2章：数据采集
│   ├── chap3/                    # 第3章：模型方法
│   └── chap4/                    # 第4章：实验结果
└── architecture/                 # draw.io 架构图源文件
```

#### 图表清单

**第2章：数据采集与预处理**
| 图号 | 文件名 | 说明 |
|:----:|--------|------|
| Fig 2-1 | fig_2_1_data_collection_architecture.png | 数据采集分层架构 |
| Fig 2-2 | fig_2_2_voltage_anomaly_types.png | 电压异常类型示意 |

**第3章：模型方法**
| 图号 | 文件名 | 说明 |
|:----:|--------|------|
| Fig 3-1 | fig_3_1_sliding_window.png | 滑动窗口预测示意图 |
| Fig 3-2 | fig_3_2_1d_to_2d_conversion.png | 1D→2D 时序转换 |
| Fig 3-3 | fig_3_3_voltage_timesnet_comparison.png | VoltageTimesNet 与 TimesNet 对比 |
| Fig 3-7 | fig_3_7_anomaly_detection_framework.png | 异常检测框架流程 |
| - | fig_timesnet_architecture.png | TimesNet 架构图 (draw.io) |
| - | fig_voltagetimesnet_architecture.png | VoltageTimesNet 架构图 (draw.io) |
| - | fig_fft_period_discovery.png | FFT 周期发现示意 (draw.io) |
| - | fig_2d_conv_inception.png | 2D 卷积 Inception 模块 (draw.io) |

**第4章：实验结果**
| 图号 | 文件名 | 说明 |
|:----:|--------|------|
| Fig 4-1 | fig_4_1_f1_comparison.png | 多模型 F1 分数对比 |
| Fig 4-2 | fig_4_2_roc_pr_curves.png | ROC/PR 曲线对比 |
| Fig 4-3 | fig_4_3_confusion_matrices.png | 混淆矩阵 |
| Fig 4-4 | fig_4_4_precision_recall_tradeoff.png | 精确率-召回率权衡 |
| Fig 4-5 | fig_4_5_radar_comparison.png | 雷达图多维对比 |
| Fig 4-6 | fig_4_6_score_distribution.png | 异常分数分布 |
| Fig 4-7 | fig_4_7_threshold_sensitivity.png | 阈值敏感性分析 |
| Fig 4-8 | fig_4_8_detection_visualization.png | 检测结果可视化 |
| Fig 4-9 | fig_4_9_training_loss.png | 训练损失曲线 |
| Fig 4-10 | fig_4_10_seq_len_ablation.png | 序列长度消融实验 |
| Fig 4-11 | fig_4_11_alpha_ablation.png | alpha 参数消融实验 |
| Fig 4-12 | fig_4_12_variant_bar_comparison.png | 模型变体对比 |
| Fig 4-13 | fig_4_13_variant_training_loss.png | 变体训练损失 |
| Fig 4-14 | fig_4_14_phase_attention_heatmap.png | 相位注意力热图 |
| Fig 4-15 | fig_4_15_multiscale_contribution.png | 多尺度贡献分析 |

#### 图表生成工作流

```bash
cd code/figures/scripts

# 生成所有章节图表
python generate_all_figures.py

# 生成单个章节 (2, 3, 4)
python generate_all_figures.py 4

# 生成单个图表
python fig_4_1_f1_comparison.py

# 同步到论文项目 (自动检测更新)
python sync_to_thesis.py

# 预览同步 (不实际复制)
python sync_to_thesis.py --dry-run
```

#### 架构图渲染 (draw.io)

架构图源文件位于 `code/figures/architecture/*.drawio`，通过 Playwright 渲染为 PNG：

```bash
cd code/figures/architecture
python render_architecture_figures.py
```

### 论文绘图规范

所有论文图表必须遵循以下规范：

| 项目 | 规范 |
|------|------|
| **格式** | 仅 PNG，300 DPI |
| **中文字体** | 五号宋体 (10.5pt SimSun/Noto Serif CJK) |
| **英文字体** | 五号 Times New Roman (10.5pt) |
| **坐标轴单位** | 使用 "/" 分隔，如 `时间步/s`、`电压/V` |
| **标题** | 无（论文中使用图注） |
| **子图** | 禁止（一个图一个文件） |
| **配色** | 朴素科研风格，柔和色调 |
| **表格** | 三线表，表头宋体加粗 |

**推荐配色方案**（柔和科研色）：
```python
THESIS_COLORS = {
    'primary': '#4878A8',    # 柔和蓝
    'secondary': '#72A86D',  # 柔和绿
    'accent': '#C4785C',     # 柔和橙
    'warning': '#D4A84C',    # 柔和黄
    'neutral': '#808080',    # 中性灰
    'light_gray': '#B0B0B0', # 浅灰
}
```

**Matplotlib 配置模板**：
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 字体配置
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'Times New Roman']
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False

# 图表样式
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 保存为 PNG
fig.savefig('figure.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
```
