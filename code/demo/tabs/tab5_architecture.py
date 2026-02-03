"""
模型结构展示标签页
展示 6 个模型的架构图和详细说明

Author: Rural Voltage Detection Project
Date: 2026
"""

import gradio as gr
import sys
from pathlib import Path

# 路径设置
DEMO_DIR = Path(__file__).parent.parent
CODE_DIR = DEMO_DIR.parent
SVG_DIR = CODE_DIR / "docs" / "model_architectures" / "svg"

# ============================================================================
# 模型信息配置
# ============================================================================

MODEL_INFO = {
    "TimesNet": {
        "svg_file": "01_TimesNet.svg",
        "title": "TimesNet",
        "subtitle": "FFT Period Discovery + 2D Convolution",
        "params": "~4.7M",
        "paper": "ICLR 2023",
        "description": """
## TimesNet 基线模型

### 核心思想
TimesNet 将一维时间序列转换为二维张量，利用 2D 卷积捕获周期内和周期间的变化模式。

### 关键创新
1. **FFT 周期发现**: 使用快速傅里叶变换 (FFT) 自动发现时间序列中的主要周期
2. **1D → 2D 重塑**: 将一维序列按发现的周期重塑为二维张量
3. **Inception 卷积**: 使用多尺度 2D 卷积 (1×1, 3×3, 5×5) 提取特征
4. **自适应聚合**: 使用 Softmax 权重融合多周期分支的输出

### 数据流
```
Input [B, T, C]
    ↓
Instance Normalization
    ↓
Data Embedding [B, T, d_model]
    ↓
TimesBlock × e_layers
    ├── FFT Period Discovery → Top-K periods
    ├── 1D → 2D Reshape
    ├── Inception Conv2D
    └── Adaptive Aggregation
    ↓
Output Projection [B, T, C]
```

### 参数配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 64 | 隐藏维度 |
| e_layers | 2 | 编码器层数 |
| top_k | 5 | 发现的周期数 |
| num_kernels | 6 | Inception 核数量 |
""",
    },
    "VoltageTimesNet": {
        "svg_file": "02_VoltageTimesNet.svg",
        "title": "VoltageTimesNet",
        "subtitle": "Preset Periods + FFT Hybrid",
        "params": "~4.8M",
        "paper": "本研究",
        "description": """
## VoltageTimesNet 改进模型

### 核心改进
在 TimesNet 基础上引入**电力系统领域知识**，使用预设周期与 FFT 发现周期的混合机制。

### 关键创新
1. **预设周期**: 基于电力系统先验知识设置固定周期 (日周期、小时周期等)
2. **混合融合**: 将预设周期与 FFT 发现周期按权重融合
3. **领域适配**: 针对农村电压数据特点优化

### 预设周期配置
| 周期名称 | 采样点数 | 物理含义 |
|----------|:--------:|----------|
| 日周期 | 288 | 日负荷模式 (5分钟采样) |
| 1小时 | 12 | 小时波动 |
| 5分钟 | 1 | 短期波动 |

### 融合公式
```python
final_periods = alpha * FFT_periods + (1-alpha) * preset_periods
# alpha = 0.7 (默认)
```

### 性能提升
- F1 分数: 0.6509 (相比 TimesNet 0.6520 略有下降，但召回率提升)
- 召回率: 0.5726 (相比 TimesNet 0.5705 提升 0.4%)
""",
    },
    "VoltageTimesNet_v2": {
        "svg_file": "03_VoltageTimesNet_v2.svg",
        "title": "VoltageTimesNet v2",
        "subtitle": "Recall-Optimized Version",
        "params": "~4.8M",
        "paper": "本研究 (最优)",
        "description": """
## VoltageTimesNet v2 召回率优化版

### 核心改进
在 VoltageTimesNet 基础上进一步优化，**针对召回率进行专项提升**。

### 关键创新
1. **异常比例调整**: 增大 anomaly_ratio 参数 (3.0)，降低检测阈值
2. **损失函数优化**: 调整重构损失权重，更关注异常样本
3. **阈值策略优化**: 使用百分位数阈值替代固定阈值

### 召回率优化策略
```python
# 调整异常比例，降低漏检率
anomaly_ratio = 3.0  # 默认 1.0

# 百分位数阈值
threshold = np.percentile(scores, 100 - anomaly_ratio)
```

### 性能对比
| 指标 | TimesNet | VoltageTimesNet | **VoltageTimesNet_v2** |
|------|:--------:|:---------------:|:----------------------:|
| F1 | 0.6520 | 0.6509 | **0.6622** (+1.6%) |
| Recall | 0.5705 | 0.5726 | **0.5858** (+2.7%) |
| Precision | 0.7606 | 0.7541 | 0.7614 |
| AUC | 0.8389 | 0.8412 | **0.8523** (+1.6%) |

### 为什么召回率更重要?
在电力系统异常检测中，**漏检**可能导致设备损坏或电网故障，代价远高于误报。
""",
    },
    "TPATimesNet": {
        "svg_file": "04_TPATimesNet.svg",
        "title": "TPATimesNet",
        "subtitle": "Three-Phase Attention Enhanced",
        "params": "~5.1M",
        "paper": "本研究",
        "description": """
## TPATimesNet 三相注意力增强版

### 核心思想
针对**三相电压数据**的特点，引入三相注意力机制，建模相间关系。

### 关键创新
1. **三相注意力**: 专门建模 A、B、C 三相电压之间的相关性
2. **跨相特征融合**: 允许不同相之间的信息交互
3. **相位敏感**: 对相位偏移和不平衡敏感

### 三相注意力机制
```python
# 输入: [B, T, 3, C//3] (三相分离)
# Q, K, V 分别来自三相
attention = softmax(Q @ K.T / sqrt(d)) @ V
# 输出: [B, T, 3, C//3] (三相融合)
```

### 适用场景
- 三相不平衡检测
- 相间故障识别
- 三相电压质量监测

### 性能
| 指标 | 值 |
|------|:--:|
| F1 | 0.6402 |
| Recall | 0.5612 |
| Precision | 0.7456 |

*注: 在单相数据上效果不如 VoltageTimesNet_v2*
""",
    },
    "MTSTimesNet": {
        "svg_file": "05_MTSTimesNet.svg",
        "title": "MTSTimesNet",
        "subtitle": "Multi-Scale Temporal Modeling",
        "params": "~5.3M",
        "paper": "本研究",
        "description": """
## MTSTimesNet 多尺度时序版

### 核心思想
引入**多尺度时序建模**，同时捕获短期波动和长期趋势。

### 关键创新
1. **多尺度分支**: 并行处理不同时间尺度的特征
2. **下采样策略**: 通过池化获取不同粒度的时序特征
3. **尺度融合**: 自适应融合多尺度特征

### 多尺度分支
```python
scales = [1, 2, 4, 8]  # 不同下采样率

for scale in scales:
    x_scaled = avg_pool(x, scale)
    features.append(process(x_scaled))

output = adaptive_fusion(features)
```

### 尺度说明
| 尺度 | 下采样率 | 捕获模式 |
|:----:|:--------:|----------|
| 1 | 1× | 高频波动 |
| 2 | 2× | 短期趋势 |
| 4 | 4× | 中期模式 |
| 8 | 8× | 长期趋势 |

### 性能
| 指标 | 值 |
|------|:--:|
| F1 | 0.6328 |
| Recall | 0.5534 |
| Precision | 0.7389 |

*注: 计算开销较大，适合离线分析*
""",
    },
    "DLinear": {
        "svg_file": "06_DLinear.svg",
        "title": "DLinear",
        "subtitle": "Lightweight Linear Baseline",
        "params": "~0.1M",
        "paper": "AAAI 2023",
        "description": """
## DLinear 轻量级基线

### 核心思想
使用**简单的线性分解**进行时间序列建模，证明简单方法的有效性。

### 关键设计
1. **趋势-季节分解**: 将序列分解为趋势和季节成分
2. **线性映射**: 对每个成分使用独立的线性层
3. **极简设计**: 无注意力、无卷积、无循环

### 模型结构
```python
# 分解
trend = moving_avg(x)
seasonal = x - trend

# 线性映射
trend_out = linear1(trend)
seasonal_out = linear2(seasonal)

# 合并
output = trend_out + seasonal_out
```

### 优势
- **速度快**: 推理速度是 TimesNet 的 10 倍以上
- **参数少**: 仅 ~0.1M 参数
- **易部署**: 适合边缘设备

### 性能
| 指标 | 值 |
|------|:--:|
| F1 | 0.6071 |
| Recall | 0.5289 |
| Precision | 0.7123 |

*注: 性能略低但速度优势明显，适合实时场景*
""",
    },
}

# 模型顺序
MODEL_ORDER = [
    "TimesNet",
    "VoltageTimesNet",
    "VoltageTimesNet_v2",
    "TPATimesNet",
    "MTSTimesNet",
    "DLinear",
]


# ============================================================================
# 辅助函数
# ============================================================================

def load_svg_content(model_name: str) -> str:
    """加载 SVG 文件内容"""
    info = MODEL_INFO.get(model_name)
    if not info:
        return "<p>模型信息未找到</p>"

    svg_path = SVG_DIR / info["svg_file"]
    if not svg_path.exists():
        return f"<p>SVG 文件未找到: {svg_path}</p>"

    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # 添加居中和缩放样式
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 20px; background: #fafafa; border-radius: 8px;">
        <div style="max-width: 100%; overflow: auto;">
            {svg_content}
        </div>
    </div>
    """


def get_model_summary(model_name: str) -> str:
    """获取模型摘要信息"""
    info = MODEL_INFO.get(model_name, {})
    return f"""
### {info.get('title', model_name)}

**副标题**: {info.get('subtitle', 'N/A')}

**参数量**: {info.get('params', 'N/A')}

**来源**: {info.get('paper', 'N/A')}
"""


# ============================================================================
# 主标签页函数
# ============================================================================

def create_architecture_tab():
    """
    创建模型结构展示标签页

    包含:
    1. Dropdown 选择模型
    2. SVG 架构图显示
    3. 模型详细说明
    """
    with gr.Tab("模型结构"):
        gr.Markdown("""
        # 模型架构展示

        本页面展示研究中使用的 6 个异常检测模型的网络结构。

        选择模型查看其架构图和详细说明。
        """)

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=MODEL_ORDER,
                value="TimesNet",
                label="选择模型",
                info="选择要查看的模型架构"
            )

        # 模型摘要
        with gr.Row():
            model_summary = gr.Markdown(
                value=get_model_summary("TimesNet"),
                label="模型摘要"
            )

        # 架构图
        with gr.Row():
            gr.Markdown("### 网络架构图")

        with gr.Row():
            architecture_svg = gr.HTML(
                value=load_svg_content("TimesNet"),
                label="架构图"
            )

        # 详细说明
        with gr.Row():
            gr.Markdown("### 模型详解")

        with gr.Row():
            model_description = gr.Markdown(
                value=MODEL_INFO["TimesNet"]["description"],
                label="详细说明"
            )

        # 模型对比表格
        with gr.Accordion("模型对比一览", open=False):
            comparison_table = """
| 模型 | 参数量 | F1 | Recall | Precision | 特点 |
|:-----|:------:|:--:|:------:|:---------:|:-----|
| **VoltageTimesNet_v2** | ~4.8M | **0.6622** | **0.5858** | 0.7614 | 召回率优化，最优模型 |
| TimesNet | ~4.7M | 0.6520 | 0.5705 | 0.7606 | 基线，FFT + 2D卷积 |
| VoltageTimesNet | ~4.8M | 0.6509 | 0.5726 | 0.7541 | 预设周期混合 |
| TPATimesNet | ~5.1M | 0.6402 | 0.5612 | 0.7456 | 三相注意力 |
| MTSTimesNet | ~5.3M | 0.6328 | 0.5534 | 0.7389 | 多尺度建模 |
| DLinear | ~0.1M | 0.6071 | 0.5289 | 0.7123 | 轻量级基线 |
"""
            gr.Markdown(comparison_table)

        # 交互逻辑
        def update_model_view(model_name: str):
            """更新模型展示"""
            svg_html = load_svg_content(model_name)
            summary = get_model_summary(model_name)
            description = MODEL_INFO.get(model_name, {}).get("description", "")
            return svg_html, summary, description

        # 绑定事件
        model_selector.change(
            fn=update_model_view,
            inputs=[model_selector],
            outputs=[architecture_svg, model_summary, model_description]
        )

    return {
        "model_selector": model_selector,
        "architecture_svg": architecture_svg,
        "model_summary": model_summary,
        "model_description": model_description,
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试模型结构页面...")

    # 检查 SVG 文件
    for model_name in MODEL_ORDER:
        info = MODEL_INFO[model_name]
        svg_path = SVG_DIR / info["svg_file"]
        status = "✓" if svg_path.exists() else "✗"
        print(f"  {status} {model_name}: {svg_path}")

    print("\n测试完成!")
