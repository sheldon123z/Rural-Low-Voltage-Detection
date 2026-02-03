"""
原理演示标签页 - TimesNet 算法原理可视化

功能:
1. 展示 TimesNet 的 FFT 周期发现原理
2. 可视化 1D → 2D 重塑过程
3. 演示 2D 卷积处理

UI 组件:
- Radio: 选择信号类型 (正弦波、方波、农村电压模拟)
- Slider: 调节 top-k 周期数 (1-10, 默认 5)
- Plot: FFT 分析结果
- Plot: 2D 重塑可视化
- Markdown: 原理说明文字
"""

import gradio as gr
import numpy as np

import sys
from pathlib import Path
DEMO_DIR = Path(__file__).parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from visualization.fft_plots import (
    create_fft_visualization,
    create_2d_reshape_visualization,
    create_period_analysis_plot,
    fft_for_period,
)


# ======================== 信号生成函数 ========================

def generate_sine_wave(length: int = 500) -> np.ndarray:
    """
    生成多周期叠加正弦波信号

    包含周期 T=20 和 T=50 的叠加

    Args:
        length: 信号长度

    Returns:
        生成的信号数组
    """
    t = np.arange(length)
    signal = (
        np.sin(2 * np.pi * t / 20) * 2.0 +      # 周期 T=20, 振幅 2.0
        np.sin(2 * np.pi * t / 50) * 1.5 +      # 周期 T=50, 振幅 1.5
        np.random.randn(length) * 0.2           # 轻微噪声
    )
    return signal


def generate_square_wave(length: int = 500) -> np.ndarray:
    """
    生成方波信号

    周期 T=30 的方波

    Args:
        length: 信号长度

    Returns:
        生成的信号数组
    """
    t = np.arange(length)
    period = 30
    # 使用 sign 函数生成方波
    signal = np.sign(np.sin(2 * np.pi * t / period)) * 2.0
    # 添加轻微噪声使其更真实
    signal = signal + np.random.randn(length) * 0.1
    return signal


def generate_rural_voltage(length: int = 500) -> np.ndarray:
    """
    生成农村电压模拟信号

    特点:
    - 基础电压 220V (归一化后为 0)
    - 日周期波动 (周期 ~288, 模拟 24 小时, 5 分钟采样)
    - 随机负载波动
    - 偶发异常

    Args:
        length: 信号长度

    Returns:
        生成的信号数组
    """
    t = np.arange(length)

    # 日周期波动 (模拟早晚高峰)
    # 假设 288 个点代表一天 (5 分钟采样间隔)
    day_period = 144  # 半天周期更明显
    day_pattern = np.sin(2 * np.pi * t / day_period) * 1.5

    # 短周期波动 (模拟设备开关)
    short_period = 24
    short_pattern = np.sin(2 * np.pi * t / short_period) * 0.5

    # 基础随机波动
    noise = np.random.randn(length) * 0.3

    # 组合信号
    signal = day_pattern + short_pattern + noise

    # 添加少量尖峰异常 (模拟瞬时过压/欠压)
    num_spikes = max(1, length // 100)
    spike_indices = np.random.choice(length, num_spikes, replace=False)
    spike_values = np.random.choice([-3.0, 3.0], num_spikes)
    signal[spike_indices] += spike_values

    return signal


def get_signal_by_type(signal_type: str, length: int = 500) -> np.ndarray:
    """
    根据类型生成信号

    Args:
        signal_type: 信号类型 ("正弦波", "方波", "农村电压模拟")
        length: 信号长度

    Returns:
        生成的信号数组
    """
    signal_generators = {
        "正弦波": generate_sine_wave,
        "方波": generate_square_wave,
        "农村电压模拟": generate_rural_voltage,
    }

    generator = signal_generators.get(signal_type, generate_sine_wave)
    return generator(length)


# ======================== 回调函数 ========================

def update_fft_plot(signal_type: str, top_k: int):
    """
    更新 FFT 分析图

    Args:
        signal_type: 信号类型
        top_k: 显示的主要周期数量

    Returns:
        FFT 可视化图表
    """
    signal = get_signal_by_type(signal_type)
    fig = create_fft_visualization(signal, top_k=top_k)
    return fig


def update_2d_reshape_plot(signal_type: str, top_k: int):
    """
    更新 2D 重塑可视化图

    使用 FFT 发现的最强周期进行重塑

    Args:
        signal_type: 信号类型
        top_k: 用于 FFT 分析的周期数量

    Returns:
        2D 重塑可视化图表
    """
    signal = get_signal_by_type(signal_type)

    # 使用 FFT 发现最强周期
    periods, amplitudes, _, _ = fft_for_period(signal, k=top_k)

    # 选择最强周期 (确保合理范围)
    main_period = periods[0] if periods else 20
    main_period = max(5, min(100, main_period))  # 限制在 5-100 之间

    fig = create_2d_reshape_visualization(signal, period=main_period)
    return fig


def update_all_plots(signal_type: str, top_k: int):
    """
    同时更新所有图表

    Args:
        signal_type: 信号类型
        top_k: 显示的主要周期数量

    Returns:
        (FFT 图, 2D 重塑图)
    """
    fft_fig = update_fft_plot(signal_type, top_k)
    reshape_fig = update_2d_reshape_plot(signal_type, top_k)
    return fft_fig, reshape_fig


# ======================== 原理说明文本 ========================

PRINCIPLE_INTRO_MD = """
## TimesNet 算法原理

TimesNet 是一种基于 **时间序列 2D 变换** 的深度学习模型，其核心思想是将一维时间序列转换为二维张量，
从而利用成熟的 2D 卷积网络捕获时间模式。

### 核心步骤

1. **FFT 周期发现**: 使用快速傅里叶变换 (FFT) 自动发现时间序列中的主要周期
2. **1D → 2D 重塑**: 根据发现的周期，将一维序列折叠成二维矩阵
3. **2D 卷积处理**: 使用 Inception 模块对 2D 表示进行特征提取
4. **多周期融合**: 对多个周期的处理结果进行加权融合
"""

FFT_EXPLANATION_MD = """
### FFT 周期发现原理

快速傅里叶变换将时间域信号转换为频率域，通过分析频谱可以发现信号中的周期性成分：

```python
# TimesNet 中的 FFT 周期发现
xf = torch.fft.rfft(x, dim=1)  # 实数 FFT
frequency = torch.abs(xf).mean(0).mean(-1)  # 计算频率强度
top_list = frequency.topk(k).indices  # 选取 top-k 频率
period_list = (seq_len / top_list).int()  # 转换为周期
```

**图中解读**:
- **上图**: 原始时间序列信号
- **下图**: FFT 频谱，星号标记为发现的主要周期对应的频率
"""

RESHAPE_EXPLANATION_MD = """
### 1D → 2D 重塑过程

根据发现的周期 T，将长度为 N 的一维序列重塑为 (N/T, T) 的二维矩阵：

```python
# 重塑过程
x_2d = x.reshape(batch, num_periods, period, channels)
# 形状: [B, N/T, T, C]
```

**重塑的意义**:
- **行方向**: 不同周期的数据
- **列方向**: 周期内的时间演变
- 这种排列使得 2D 卷积可以同时捕获 **周期内模式** 和 **周期间变化**

**图中解读**:
- **热力图**: 重塑后的 2D 矩阵，颜色深浅表示数值大小
- **叠加图**: 将各周期数据叠加显示，可以观察周期性重复模式
"""

CONVOLUTION_EXPLANATION_MD = """
### 2D 卷积处理

TimesNet 使用 Inception 模块对 2D 表示进行处理：

```python
# Inception Block 结构
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        # 多尺度卷积核: 1, 3, 5, 7, ...
        kernels = [1, 3, 5, 7, ...][:num_kernels]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, (k, k), padding=(k//2, k//2))
            for k in kernels
        ])

    def forward(self, x):
        # 多尺度特征融合
        outputs = [conv(x) for conv in self.convs]
        return torch.stack(outputs, dim=-1).mean(-1)
```

**多尺度卷积的作用**:
- 小卷积核 (1×1, 3×3): 捕获局部细节
- 大卷积核 (5×5, 7×7): 捕获较长范围的模式
- 多尺度融合: 综合不同尺度的时间特征

### 异常检测原理

在异常检测任务中，TimesNet 学习正常数据的重构模式：

```
异常分数 = ||原始输入 - 重构输出||
```

- **正常数据**: 重构误差小
- **异常数据**: 重构误差大，超过阈值则判定为异常
"""


# ======================== 标签页创建函数 ========================

def create_principle_tab():
    """
    创建原理演示标签页

    Returns:
        包含所有 UI 组件的字典
    """
    with gr.Tab("原理演示"):
        # 页面标题
        gr.Markdown(PRINCIPLE_INTRO_MD)

        # 控制面板
        with gr.Row():
            with gr.Column(scale=1):
                signal_type = gr.Radio(
                    choices=["正弦波", "方波", "农村电压模拟"],
                    value="正弦波",
                    label="选择信号类型",
                    info="选择不同类型的示例信号来观察 FFT 分析效果"
                )

            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K 周期数",
                    info="FFT 分析中显示的主要周期数量"
                )

            with gr.Column(scale=1):
                analyze_btn = gr.Button(
                    "开始分析",
                    variant="primary"
                )

        # 信号类型说明
        with gr.Accordion("信号类型说明", open=False):
            gr.Markdown("""
            | 信号类型 | 说明 | 主要周期 |
            |---------|------|---------|
            | **正弦波** | 多周期叠加的正弦信号 | T=20, T=50 |
            | **方波** | 周期性方波信号 | T=30 |
            | **农村电压模拟** | 模拟真实农村电压波动 | T=144 (日周期), T=24 (短周期) |
            """)

        gr.Markdown("---")

        # FFT 分析部分
        gr.Markdown(FFT_EXPLANATION_MD)
        fft_plot = gr.Plot(
            label="FFT 周期发现可视化",
            show_label=True
        )

        gr.Markdown("---")

        # 2D 重塑部分
        gr.Markdown(RESHAPE_EXPLANATION_MD)
        reshape_plot = gr.Plot(
            label="1D → 2D 重塑可视化",
            show_label=True
        )

        gr.Markdown("---")

        # 2D 卷积说明
        gr.Markdown(CONVOLUTION_EXPLANATION_MD)

        # 事件绑定
        analyze_btn.click(
            fn=update_all_plots,
            inputs=[signal_type, top_k_slider],
            outputs=[fft_plot, reshape_plot]
        )

        # 参数变化时自动更新
        signal_type.change(
            fn=update_all_plots,
            inputs=[signal_type, top_k_slider],
            outputs=[fft_plot, reshape_plot]
        )

        top_k_slider.release(
            fn=update_all_plots,
            inputs=[signal_type, top_k_slider],
            outputs=[fft_plot, reshape_plot]
        )

    # 返回组件引用
    components = {
        "signal_type": signal_type,
        "top_k_slider": top_k_slider,
        "analyze_btn": analyze_btn,
        "fft_plot": fft_plot,
        "reshape_plot": reshape_plot,
    }

    return components


# ======================== 测试代码 ========================

if __name__ == "__main__":
    # 测试信号生成
    print("测试信号生成...")

    for signal_type in ["正弦波", "方波", "农村电压模拟"]:
        signal = get_signal_by_type(signal_type)
        print(f"  {signal_type}: shape={signal.shape}, "
              f"min={signal.min():.2f}, max={signal.max():.2f}")

    print("\n测试 FFT 分析...")
    signal = generate_sine_wave()
    periods, amplitudes, _, _ = fft_for_period(signal, k=5)
    print(f"  发现的周期: {periods}")
    print(f"  对应振幅: {[f'{a:.2f}' for a in amplitudes]}")

    print("\n原理演示标签页模块加载成功!")
