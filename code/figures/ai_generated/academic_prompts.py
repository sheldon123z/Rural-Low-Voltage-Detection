#!/usr/bin/env python3
"""
IEEE Smart Grid 学术风格图表生成提示词库

为农村低压配电网电压异常检测论文生成学术级结构图和示意图。
风格参考：IEEE Transactions on Smart Grid, IEEE Power & Energy Magazine

设计原则：
1. 清晰简洁的技术风格
2. 专业的配色方案（蓝色系为主，灰色辅助）
3. 矢量化外观，适合学术出版
4. 无装饰性元素，纯技术表达
"""

# IEEE 学术风格基础参数
IEEE_STYLE_BASE = """
Style: Professional IEEE academic illustration for power systems research.
Visual characteristics:
- Clean vector-like appearance with sharp edges
- Color palette: IEEE blue (#0076A8), dark gray (#333333), light gray (#E5E5E5), white background
- Minimal shadows, flat design aesthetic
- Sans-serif technical labels (Arial/Helvetica style)
- High contrast for print reproduction
- No decorative elements, purely technical
Resolution: 4K, ultra-sharp, print-quality
Background: Pure white (#FFFFFF)
"""

# 图表提示词定义
PROMPTS = {
    # ============================================================
    # 第2章：数据采集与预处理
    # ============================================================

    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "description": "Rural Low-Voltage Distribution Network Data Collection Architecture",
        "prompt": f"""
Create a professional technical diagram showing a three-layer data collection architecture for rural low-voltage power distribution network monitoring.

Structure (top to bottom):
LAYER 1 - Field Layer (底层 - 现场层):
- Smart meters icon (简洁的智能电表图标)
- Voltage sensors distributed along power lines
- Multiple rural households connected to distribution transformers
- Show 3-phase power lines (A, B, C phases) in different colors

LAYER 2 - Communication Layer (中层 - 通信层):
- Data concentrators collecting from field devices
- Wireless communication symbols (4G/NB-IoT icons)
- Data aggregation nodes

LAYER 3 - Platform Layer (顶层 - 平台层):
- Central data server/cloud platform
- Database storage icon
- Data processing module
- Anomaly detection system block

Connection arrows: Vertical data flow arrows between layers, labeled with "voltage data", "status signals"

{IEEE_STYLE_BASE}
Layout: Vertical hierarchy, balanced composition
Labels: English technical terms with clean typography
No text labels on the image itself - pure visual diagram
""",
        "negative_prompt": "cartoon, 3D realistic, photorealistic, shadows, gradients, decorative elements, text labels, Chinese characters"
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "description": "Voltage Anomaly Types in Rural Distribution Networks",
        "prompt": f"""
Create a technical illustration showing 4 types of voltage anomalies in a 2x2 grid layout for power systems research.

Grid layout (4 panels):
Panel 1 (Top-Left) - Voltage Sag/Dip:
- Waveform showing sudden voltage drop (to 70-80% of nominal)
- Duration marker showing short-term event
- Clean sine wave with depression in the middle

Panel 2 (Top-Right) - Voltage Swell:
- Waveform showing voltage increase (to 110-120% of nominal)
- Overvoltage region highlighted
- Clean sine wave with elevated section

Panel 3 (Bottom-Left) - Voltage Flicker:
- Waveform showing amplitude modulation
- Oscillating envelope around the sine wave
- Periodic amplitude variation

Panel 4 (Bottom-Right) - Voltage Interruption:
- Waveform showing complete loss of voltage
- Zero-voltage period clearly marked
- Recovery transient shown

Each panel:
- Clean coordinate axes (time vs voltage)
- Reference line for nominal voltage (220V or 1.0 p.u.)
- Anomaly region highlighted in light red/orange tint
- Grid lines for professional appearance

{IEEE_STYLE_BASE}
Color coding: Normal voltage in IEEE blue, anomaly regions in muted orange/red
""",
        "negative_prompt": "3D, photorealistic, cartoon, excessive colors, decorative, hand-drawn, sketchy"
    },

    # ============================================================
    # 第3章：模型方法
    # ============================================================

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "description": "Sliding Window Mechanism for Time Series Prediction",
        "prompt": f"""
Create a technical diagram illustrating the sliding window mechanism for time series anomaly detection.

Main elements:
1. Long horizontal time series signal (voltage waveform over time)
   - X-axis: Time steps (t₁, t₂, ... tₙ)
   - Y-axis: Voltage amplitude
   - Show realistic voltage fluctuations

2. Sliding window visualization:
   - Rectangular window frame (highlighted in IEEE blue)
   - Window length labeled as "L" (sequence length)
   - Show 3 window positions: past, current, future
   - Arrows showing window sliding direction (left to right)

3. Input-Output relationship:
   - Input window: "Historical sequence" bracket
   - Output: "Predicted/Reconstructed" single point or short sequence
   - Arrow from input to output showing the prediction flow

4. Overlap indication:
   - Show stride/step size between consecutive windows
   - Overlap region in lighter shade

{IEEE_STYLE_BASE}
Layout: Horizontal composition, left-to-right flow
Annotations: Mathematical notation style (L, t, Δt)
""",
        "negative_prompt": "3D perspective, photorealistic, cartoon style, excessive decoration, Chinese text"
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "1D到2D时序转换示意图",
        "description": "1D to 2D Time Series Transformation for TimesNet",
        "prompt": f"""
Create a technical diagram showing the transformation from 1D time series to 2D representation in TimesNet architecture.

Three-stage visualization (left to right):

STAGE 1 - Original 1D Signal:
- Long horizontal waveform (voltage time series)
- X-axis: Time (t)
- Periodic pattern visible in the signal
- Length: T time steps

STAGE 2 - Period Detection (FFT):
- Frequency spectrum bar chart
- Top-k dominant frequencies highlighted
- Arrow showing "FFT Analysis"
- Detected period values: p₁, p₂, ...

STAGE 3 - 2D Reshaping:
- 1D signal folded into 2D matrix
- Matrix dimensions: (period × T/period)
- Show the folding process with curved arrows
- Resulting 2D "image" with temporal patterns visible
- Color gradient showing signal amplitude

Transformation arrows between stages:
- Stage 1 → Stage 2: "Frequency Analysis"
- Stage 2 → Stage 3: "Reshape by Period"

{IEEE_STYLE_BASE}
Mathematical notation: Use subscripts and Greek letters
Show dimensional annotations: (B, T, C) → (B, p, T/p, C)
""",
        "negative_prompt": "photorealistic, 3D rendering, cartoon, hand-drawn, decorative elements"
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "description": "End-to-End Anomaly Detection Framework",
        "prompt": f"""
Create a professional flowchart showing the complete anomaly detection framework for voltage time series.

Flow structure (left to right or top to bottom):

BLOCK 1 - Data Input:
- Icon: Time series waveform
- Label concept: "Raw Voltage Data"
- Multiple channels (16 features)

BLOCK 2 - Preprocessing:
- Normalization module
- Sliding window segmentation
- Missing value handling

BLOCK 3 - Feature Extraction (VoltageTimesNet):
- FFT period detection sub-block
- 1D→2D transformation sub-block
- 2D Inception convolution sub-block
- Multi-scale feature fusion

BLOCK 4 - Reconstruction:
- Decoder network
- Output: Reconstructed time series

BLOCK 5 - Anomaly Scoring:
- Reconstruction error calculation
- Point-wise and window-wise scoring
- Formula concept: ||x - x̂||

BLOCK 6 - Detection Output:
- Threshold comparison
- Binary anomaly labels
- Anomaly visualization

Arrows: Directional flow arrows between blocks
Feedback loop: Optional training loss feedback arrow

{IEEE_STYLE_BASE}
Layout: Horizontal pipeline or vertical flowchart
Blocks: Rounded rectangles with consistent sizing
""",
        "negative_prompt": "3D, photorealistic, cartoon, excessive colors, decorative borders, Chinese characters"
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet 网络架构图",
        "description": "TimesNet Neural Network Architecture",
        "prompt": f"""
Create a detailed neural network architecture diagram for TimesNet model.

Architecture components (vertical stack):

INPUT LAYER:
- Input tensor shape: (Batch, Time, Channels)
- Embedding layer block

TIMESBLOCK (repeated N times):
- FFT module: "Period Discovery via FFT"
- Reshape operation: 1D → 2D
- 2D Inception Block:
  * Multiple parallel convolution paths
  * Different kernel sizes (1×1, 3×3, 5×5)
  * Concatenation
- Reshape back: 2D → 1D
- Adaptive aggregation across periods
- Residual connection (skip arrow)
- Layer normalization

OUTPUT LAYER:
- Projection layer
- Output tensor shape: (Batch, Time, Channels)

Visual elements:
- Tensor flow arrows with dimension annotations
- Inception block shown as parallel paths merging
- Residual connections as curved bypass arrows
- Stacked blocks to show repetition

{IEEE_STYLE_BASE}
Style: Clean neural network diagram similar to original papers
Blocks: Rectangular modules with rounded corners
""",
        "negative_prompt": "3D perspective, photorealistic, cartoon, hand-drawn, excessive decoration"
    },

    "fig_voltagetimesnet_architecture": {
        "title": "VoltageTimesNet 网络架构图",
        "description": "VoltageTimesNet Architecture with Domain-Specific Enhancements",
        "prompt": f"""
Create an architecture diagram for VoltageTimesNet, highlighting differences from standard TimesNet.

Architecture (similar to TimesNet but with enhancements):

INPUT:
- Voltage time series: (B, T, 16) - 16 voltage features
- Highlight: "16-channel voltage input"

ENHANCED TIMESBLOCK:
- Standard FFT period detection
- ADDITION 1: "Domain Prior Injection"
  * Pre-defined periods: 50Hz (20ms), daily (24h)
  * Merge with FFT-detected periods
- 1D→2D transformation
- 2D Inception convolution
- ADDITION 2: "Enhanced Feature Weighting"
  * Learnable period importance weights
  * Softmax attention over periods
- 2D→1D transformation
- Residual + LayerNorm

OUTPUT:
- Reconstruction head for anomaly detection
- Anomaly score computation

Side-by-side comparison callout (optional):
- Left: Standard TimesNet block (simplified)
- Right: VoltageTimesNet block (with enhancements highlighted)

{IEEE_STYLE_BASE}
Highlight color: Use orange/yellow to mark enhancements
Annotations: "Domain-specific" labels near enhanced components
""",
        "negative_prompt": "3D, photorealistic, cartoon, excessive shadows, decorative elements"
    },

    "fig_fft_period_discovery": {
        "title": "FFT 周期发现示意图",
        "description": "Period Discovery via Fast Fourier Transform",
        "prompt": f"""
Create a technical illustration showing how FFT discovers periodic patterns in voltage time series.

Three-panel layout (top to bottom or left to right):

PANEL 1 - Time Domain Signal:
- Voltage waveform with visible periodic patterns
- X-axis: Time (samples or seconds)
- Y-axis: Voltage amplitude
- Show multiple overlapping periodicities

PANEL 2 - Frequency Domain (FFT Result):
- Amplitude spectrum (|FFT|)
- X-axis: Frequency (Hz) or period (samples)
- Y-axis: Magnitude
- Dominant peaks highlighted with markers
- Top-k peaks labeled: f₁, f₂, f₃, ...
- Threshold line for peak selection

PANEL 3 - Detected Periods:
- Bar chart or markers showing selected periods
- Period values: p₁ = T/f₁, p₂ = T/f₂, ...
- Importance weights (softmax normalized)

Transformation arrows:
- Panel 1 → Panel 2: "FFT" arrow
- Panel 2 → Panel 3: "Top-k Selection" arrow

{IEEE_STYLE_BASE}
Mathematical annotations: FFT formula hint, frequency-period relationship
Color: Peaks in IEEE blue, secondary in gray
""",
        "negative_prompt": "3D, photorealistic, cartoon, hand-drawn style, decorative"
    },

    "fig_2d_conv_inception": {
        "title": "2D卷积 Inception 模块示意图",
        "description": "2D Inception Convolution Block for Temporal Patterns",
        "prompt": f"""
Create a technical diagram showing the Inception-style 2D convolution block used in TimesNet.

Module structure:

INPUT:
- 2D feature map: (B, C, H, W) representing reshaped time series
- H = period, W = T/period

PARALLEL PATHS (4 branches):
Branch 1: 1×1 Convolution
- Pointwise convolution for channel mixing
- Output: same spatial size

Branch 2: 1×1 Conv → 3×3 Conv
- Dimension reduction then 3×3 spatial
- Captures local patterns

Branch 3: 1×1 Conv → 5×5 Conv
- Dimension reduction then 5×5 spatial
- Captures medium-range patterns

Branch 4: 3×3 MaxPool → 1×1 Conv
- Pooling for translation invariance
- 1×1 conv for channel adjustment

CONCATENATION:
- All branches merged along channel dimension
- Final 1×1 conv for channel reduction (optional)

OUTPUT:
- Fused feature map with multi-scale information

Visual style:
- Parallel vertical paths
- Merge point shown as concatenation symbol
- Kernel size annotations on each conv block

{IEEE_STYLE_BASE}
Layout: Vertical flow with parallel horizontal branches
Blocks: Rounded rectangles for operations
""",
        "negative_prompt": "3D rendering, photorealistic, cartoon, hand-drawn, excessive colors"
    },
}

# 模型配置
IMAGE_GENERATION_CONFIG = {
    "model": "google/gemini-2.0-flash-exp:free",  # 免费模型
    "backup_models": [
        "google/gemini-flash-1.5",
        "meta-llama/llama-4-maverick:free"
    ],
    "image_size": "1024x1024",
    "quality": "hd",
    "style": "natural"
}

def get_prompt(figure_key: str) -> dict:
    """获取指定图表的提示词配置"""
    if figure_key not in PROMPTS:
        raise KeyError(f"Unknown figure key: {figure_key}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[figure_key]

def list_available_prompts() -> list:
    """列出所有可用的图表提示词"""
    return [(key, PROMPTS[key]["title"]) for key in PROMPTS]

if __name__ == "__main__":
    print("可用的学术图表提示词：")
    print("=" * 60)
    for key, title in list_available_prompts():
        print(f"  {key}")
        print(f"    → {title}")
    print("=" * 60)
    print(f"总计: {len(PROMPTS)} 个提示词")
