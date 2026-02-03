"""
模型结构展示标签页
展示 6 个模型的架构图和详细说明

Author: Rural Voltage Detection Project
Date: 2026
"""

import gradio as gr
import sys
from pathlib import Path

# 路径设置 (HuggingFace Spaces 兼容)
DEMO_DIR = Path(__file__).parent.parent
SVG_DIR = DEMO_DIR / "docs" / "model_architectures" / "svg"

# ============================================================================
# 模型信息配置
# ============================================================================

MODEL_INFO = {
    "TimesNet": {
        "svg_file": "01_TimesNet.svg",
        "title": "TimesNet",
        "subtitle": "FFT周期发现 + 二维卷积",
        "params": "~4.7M",
        "paper": "ICLR 2023",
        "description": """
## TimesNet 基线模型

### 核心思想
TimesNet 将一维时间序列转换为二维张量，利用 2D 卷积捕获周期内和周期间的变化模式。

### 关键创新
1. **FFT 周期发现**: 使用快速傅里叶变换自动发现主要周期
2. **1D → 2D 重塑**: 将一维序列按周期重塑为二维张量
3. **Inception 卷积**: 多尺度 2D 卷积提取特征
4. **自适应聚合**: Softmax 权重融合多周期分支

### 参数配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 64 | 隐藏维度 |
| e_layers | 2 | 编码器层数 |
| top_k | 5 | 发现的周期数 |
""",
        "structure_explanation": """
### 架构解读

**数据流向**: 输入 → 实例归一化 → 数据嵌入 → TimesBlock(×2层) → 输出投影 → 输出

**TimesBlock 内部流程**:
1. **FFT周期发现**: 对输入信号进行快速傅里叶变换，选取幅值最大的 Top-K 个频率分量，计算对应的周期长度。例如，若发现频率 f=10Hz，则周期 p=T/10
2. **多周期并行处理**: 对每个发现的周期，将一维信号按该周期长度重塑为二维矩阵（行=周期间变化，列=周期内变化），然后用 Inception V1 模块进行多尺度二维卷积（1×1, 3×3, 5×5 三种卷积核）
3. **自适应聚合**: 使用 Softmax 对各周期分支的 FFT 幅值进行归一化，作为权重对各分支输出进行加权求和
4. **残差连接**: 将聚合结果与输入相加，保留原始信息

**异常检测原理**: 模型学习正常时序模式的重构能力，重构误差超过阈值即判定为异常
""",
    },
    "VoltageTimesNet": {
        "svg_file": "02_VoltageTimesNet.svg",
        "title": "VoltageTimesNet",
        "subtitle": "预设周期 + FFT 混合发现",
        "params": "~4.8M",
        "paper": "本研究",
        "description": """
## VoltageTimesNet 改进模型

### 核心改进
引入**电力系统领域知识**，使用预设周期与 FFT 发现周期的混合机制。

### 融合公式
```
final_periods = alpha * FFT_periods + (1-alpha) * preset_periods
# alpha = 0.7 (默认)
```

### 性能
- F1: 0.6509
- Recall: 0.5726
""",
        "structure_explanation": """
### 架构解读

**与 TimesNet 的关键差异**: 在周期发现环节引入了电力系统领域知识

**混合周期发现机制**:
1. **FFT发现分支** (权重 1-α): 与 TimesNet 相同，通过数据驱动方式自动发现信号中的主要周期
2. **预设周期分支** (权重 α): 基于电力系统先验知识，预设了4个关键周期：
   - **60秒**: 捕捉瞬时电压波动（如电器启停）
   - **300秒**: 捕捉负荷变化周期（5分钟级负荷波动）
   - **900秒**: 对应电力需量计量周期（15分钟）
   - **3600秒**: 对应日负荷模式的小时级变化
3. **混合策略**: 通过固定权重 α=0.3 将两种周期融合

**时序平滑卷积**: 在二维卷积后增加了一维时序平滑卷积（kernel=3），减少因周期重塑引入的边界不连续性

**设计动机**: 纯数据驱动的 FFT 可能遗漏电力系统的固有周期特征，领域知识的引入有助于模型关注电压信号的关键时间尺度
""",
    },
    "VoltageTimesNet_v2": {
        "svg_file": "03_VoltageTimesNet_v2.svg",
        "title": "VoltageTimesNet v2",
        "subtitle": "召回率优化版",
        "params": "~4.8M",
        "paper": "本研究 (最优)",
        "description": """
## VoltageTimesNet v2 召回率优化版

### 核心改进
**针对召回率进行专项提升**。

### 优化策略
- 异常比例调整: anomaly_ratio = 3.0
- 损失函数优化
- 阈值策略优化

### 性能对比
| 指标 | TimesNet | **VoltageTimesNet_v2** |
|------|:--------:|:----------------------:|
| F1 | 0.6520 | **0.6622** (+1.6%) |
| Recall | 0.5705 | **0.5858** (+2.7%) |
""",
        "structure_explanation": """
### 架构解读

**本研究的最优模型**，在 VoltageTimesNet 基础上进行了多项针对性优化

**五大创新模块**:

1. **电能质量编码器** (Power Quality Encoder):
   - 将16维输入特征按物理含义分组编码：电压编码器处理 Va/Vb/Vc（→d/2维），THD编码器处理三相谐波畸变率（→d/4维），不平衡编码器处理电压不平衡度（→d/4维）
   - 通过分组编码让模型更好地理解各特征的物理意义

2. **可学习混合周期发现** (Learnable Hybrid Period):
   - 将 v1 中固定的混合权重 α 替换为可学习参数 α=sigmoid(θ)
   - 模型在训练过程中自动调节 FFT 发现周期与预设周期的权重比例

3. **多尺度时序卷积** (Multi-Scale Temporal Conv):
   - 使用三种不同卷积核（k=3, 5, 7）并行提取不同时间尺度的特征
   - 拼接后通过线性层融合，捕获从短期波动到长期趋势的多尺度信息

4. **异常敏感度放大器** (Anomaly Sensitivity Amplifier):
   - 通过 Linear(d→2d) → ReLU → Linear(2d→d) → Sigmoid 的瓶颈结构
   - 输出 [0,1] 范围的注意力权重，放大异常特征、抑制正常特征，提升召回率

5. **相位约束模块** (Phase Constraint):
   - 引入可学习的 3×3 相位相关矩阵，建模三相电压 (Va, Vb, Vc) 之间的物理约束关系
   - 三相平衡系统中各相应满足120°相位差，该模块帮助模型学习这种约束
""",
    },
    "TPATimesNet": {
        "svg_file": "04_TPATimesNet.svg",
        "title": "TPATimesNet",
        "subtitle": "三相注意力增强",
        "params": "~5.1M",
        "paper": "本研究",
        "description": """
## TPATimesNet 三相注意力增强版

针对**三相电压数据**引入三相注意力机制。

### 性能
- F1: 0.6402
- Recall: 0.5612
""",
        "structure_explanation": """
### 架构解读

**核心创新**: 在标准 TimesNet 的卷积分支之外，增加了专门针对三相电压的注意力分支

**双分支并行架构**:

1. **卷积分支** (左侧):
   - 与标准 TimesNet 相同的多周期二维卷积处理
   - 负责捕获时序模式和周期特征

2. **三相注意力分支** (右侧):
   - **Q/K/V 投影**: 对特征进行查询、键、值三种线性变换
   - **注意力得分**: 计算标准的缩放点积注意力 Q×K^T/√d
   - **相位偏置矩阵**: 这是核心创新——引入一个可学习的 3×3 偏置矩阵，编码三相电压间的先验关系。初始化为对角线1.0、其余0.5（反映同相关联强、异相关联弱的物理特性）
   - **加权输出**: Softmax 归一化后与 V 相乘得到注意力输出

3. **融合门控** (Fusion Gate):
   - 使用 sigmoid 门控机制动态融合卷积分支和注意力分支
   - gate = σ(W·[conv_out; attn_out])
   - 最终输出 = gate × conv_out + (1-gate) × attn_out
   - 让模型自动学习在不同时刻偏重时序模式还是相间关系
""",
    },
    "MTSTimesNet": {
        "svg_file": "05_MTSTimesNet.svg",
        "title": "MTSTimesNet",
        "subtitle": "多尺度时序建模",
        "params": "~5.3M",
        "paper": "本研究",
        "description": """
## MTSTimesNet 多尺度时序版

引入**多尺度时序建模**，同时捕获短期波动和长期趋势。

### 性能
- F1: 0.6328
- Recall: 0.5534
""",
        "structure_explanation": """
### 架构解读

**核心思想**: 将 FFT 发现的周期按长度分为三个尺度范围，分别独立处理后融合

**三尺度并行分支**:

1. **短期尺度** (周期 2-20):
   - 捕捉瞬时波动，如电器启停造成的快速电压变化
   - FFT 仅保留短周期分量，进行独立的 1D→2D→Conv2D→1D 处理

2. **中期尺度** (周期 20-60):
   - 捕捉负荷变化周期，如家庭用电的分钟级规律
   - 独立处理中等周期的时序特征

3. **长期尺度** (周期 60-200):
   - 捕捉日周期模式，如昼夜用电差异
   - 关注长时间跨度的电压变化趋势

**跨尺度连接** (Cross-Scale Connection):
- 将三个尺度的输出拼接后通过 Conv1d(3d→d) 进行维度压缩
- 再求和融合，使不同尺度间的信息可以交互

**自适应融合门控** (Adaptive Fusion Gate):
- 对三个尺度的输出取均值，输入 MLP 网络
- 通过 Softmax 产生三个权重 (w_s, w_m, w_l)
- 最终输出 = w_s × short_out + w_m × medium_out + w_l × long_out
- 模型自动学习在不同场景下各尺度的重要程度
""",
    },
    "DLinear": {
        "svg_file": "06_DLinear.svg",
        "title": "DLinear",
        "subtitle": "轻量级线性基线",
        "params": "~0.1M",
        "paper": "AAAI 2023",
        "description": """
## DLinear 轻量级基线

使用**简单的线性分解**进行时间序列建模。

### 优势
- 速度快: 推理速度是 TimesNet 的 10 倍以上
- 参数少: 仅 ~0.1M 参数

### 性能
- F1: 0.6071
- Recall: 0.5289
""",
        "structure_explanation": """
### 架构解读

**设计哲学**: "简单是否足够？" —— 用最简单的线性模型作为基线，验证复杂模型的必要性

**序列分解** (Series Decomposition):
- 使用移动平均滤波器（窗口大小=25）提取趋势分量
- 季节分量 = 原始信号 - 趋势分量
- 将时间序列分解为"缓慢变化的趋势"和"快速波动的季节性"两部分

**双分支线性映射**:

1. **趋势分支**:
   - 将维度从 [B,T,C] 转置为 [B,C,T]
   - 对每个通道独立施加线性变换 Linear(T→T)
   - 转置回 [B,T,C]

2. **季节分支**:
   - 与趋势分支结构完全相同
   - 独立学习季节性模式的变换

**输出合成**: 直接将趋势分支和季节分支的输出相加

**参数分析**:
- 无注意力机制、无卷积操作，仅有两个线性层
- 参数量约为 TimesNet 的 1/200（O(C×T²) vs O(d²×layers)）
- 推理速度极快，适合资源受限场景
- 在本实验中 F1=0.6071，说明复杂模型（如 VoltageTimesNet_v2 的 F1=0.6622）确实带来了有意义的性能提升
""",
    },
}

MODEL_ORDER = ["TimesNet", "VoltageTimesNet", "VoltageTimesNet_v2", "TPATimesNet", "MTSTimesNet", "DLinear"]


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

**副标题**: {info.get('subtitle', 'N/A')} | **参数量**: {info.get('params', 'N/A')} | **来源**: {info.get('paper', 'N/A')}
"""


def create_architecture_tab():
    """创建模型结构展示标签页"""
    with gr.Tab("模型结构"):
        gr.Markdown("""
        # 模型架构展示

        本页面展示研究中使用的 6 个异常检测模型的网络结构。选择模型查看其架构图和详细说明。
        """)

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=MODEL_ORDER,
                value="TimesNet",
                label="选择模型",
                info="选择要查看的模型架构"
            )

        with gr.Row():
            model_summary = gr.Markdown(value=get_model_summary("TimesNet"))

        with gr.Row():
            gr.Markdown("### 网络架构图")

        with gr.Row():
            architecture_svg = gr.HTML(value=load_svg_content("TimesNet"))

        with gr.Row():
            gr.Markdown("### 架构解读")

        with gr.Row():
            structure_explanation = gr.Markdown(
                value=MODEL_INFO["TimesNet"]["structure_explanation"]
            )

        with gr.Row():
            gr.Markdown("### 模型详解")

        with gr.Row():
            model_description = gr.Markdown(value=MODEL_INFO["TimesNet"]["description"])

        with gr.Accordion("模型对比一览", open=False):
            gr.Markdown("""
| 模型 | 参数量 | F1 | Recall | Precision | 特点 |
|:-----|:------:|:--:|:------:|:---------:|:-----|
| **VoltageTimesNet_v2** | ~4.8M | **0.6622** | **0.5858** | 0.7614 | 召回率优化，最优模型 |
| TimesNet | ~4.7M | 0.6520 | 0.5705 | 0.7606 | 基线，FFT + 2D卷积 |
| VoltageTimesNet | ~4.8M | 0.6509 | 0.5726 | 0.7541 | 预设周期混合 |
| TPATimesNet | ~5.1M | 0.6402 | 0.5612 | 0.7456 | 三相注意力 |
| MTSTimesNet | ~5.3M | 0.6328 | 0.5534 | 0.7389 | 多尺度建模 |
| DLinear | ~0.1M | 0.6071 | 0.5289 | 0.7123 | 轻量级基线 |
""")

        def update_model_view(model_name: str):
            svg_html = load_svg_content(model_name)
            summary = get_model_summary(model_name)
            description = MODEL_INFO.get(model_name, {}).get("description", "")
            explanation = MODEL_INFO.get(model_name, {}).get("structure_explanation", "")
            return svg_html, summary, explanation, description

        model_selector.change(
            fn=update_model_view,
            inputs=[model_selector],
            outputs=[architecture_svg, model_summary, structure_explanation, model_description]
        )

    return {"model_selector": model_selector, "architecture_svg": architecture_svg}
