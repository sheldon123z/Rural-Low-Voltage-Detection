#!/usr/bin/env python3
"""
论文图表生成器 - 通过 OpenRouter API 调用图像生成模型

要求：
- 所有文字必须使用中文（宋体，12号以上）
- 4K 高清（4096x4096 或适合 A4 论文的尺寸）
- 图片边缘裁剪，减少空白
- 适合 A4 论文排版

支持的模型：
- google/gemini-2.5-flash-image (推荐)
- google/gemini-3-pro-image-preview
- openai/gpt-5-image
"""

import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import time
import re

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# API 配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyBTIQLM1WClyvwCTRwvgjrGNa81HHdDBtQ"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Clash 代理配置
PROXY_CONFIG = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# API 提供商选择: "google" 或 "openrouter"
API_PROVIDER = "google"

# Google AI Studio 图像生成模型（Nano Banana Pro）
GOOGLE_IMAGE_MODELS = [
    "gemini-3-pro-image-preview",  # Nano Banana Pro 官方模型
]

# OpenRouter 图像生成模型
OPENROUTER_IMAGE_MODELS = [
    "google/gemini-2.5-flash-image",
]

# ============================================================
# 论文图表风格要求（中文版）
# ============================================================
THESIS_STYLE_REQUIREMENTS = """
【图片风格要求 - 必须严格遵守】

1. 文字要求：
   - 所有文字标签必须使用简体中文
   - 禁止出现任何英文字母或单词
   - 字体风格：宋体（SimSun）
   - 字号：12号字以上，确保清晰可读
   - 文字颜色：黑色或深灰色

2. 配色方案：
   - 主色调：学术蓝 (#0076A8)
   - 辅助色：深灰色 (#333333)
   - 背景色：纯白色 (#FFFFFF)
   - 强调色：橙色 (#E07020) 仅用于高亮

3. 视觉风格：
   - 扁平化设计，无立体阴影
   - 矢量风格，边缘锐利清晰
   - 专业学术图表风格
   - 适合黑白印刷（高对比度）

4. 布局要求：
   - 图片边缘紧凑，减少空白区域
   - 元素居中对齐，布局平衡
   - 适合 A4 论文排版（宽度约 15cm）

5. 分辨率：
   - 4K 高清质量
   - 300 DPI 印刷质量
   - 输出尺寸：约 1800x1200 像素（适合论文单栏）

6. 禁止元素：
   - 禁止英文
   - 禁止装饰性花纹
   - 禁止卡通风格
   - 禁止过多颜色
"""

# ============================================================
# 论文图表提示词（Nano Banana Pro 优化版 v2）
# 基于 Nano Banana Pro 库的 "Infographic with Technical Annotation" 模板
# 融入论文实际内容：农村低压配电网电压异常检测、VoltageTimesNet 模型
# ============================================================

# 统一风格参数（基于 Nano Banana Pro 流程图样式指南优化）
NANO_BANANA_STYLE_GUIDE = """
【Style Core Parameters - Nano Banana Pro Infographic Style】

Visual Style:
- Style: Clean technical infographic, IEEE academic diagram, engineering manual aesthetic
- Rendering: Photorealistic 3D elements with hand-drawn annotation overlays
- Composition: Balanced spacing, center aligned, generous negative space

Color Palette:
- Background: Pure white (#FFFFFF)
- Primary: Deep blue (#0076A8) for main elements
- Accent: Orange (#E07020) for highlights and emphasis
- Lines/Borders: Dark gray (#333333) or black (#000000)
- Fill: White or semi-transparent blue for shapes

Lines/Connectors:
- Type: Orthogonal/Elbow lines with right-angle corners
- Thickness: 2px for main lines, 1px for secondary
- Arrow: Solid small triangle arrowhead, black or blue

Shapes:
- Normal Node: Rounded rectangle (corner radius 8-12px)
- Decision Node: Diamond shape
- Start/End: Capsule or pill shape

Typography:
- NO TEXT LABELS in the image
- Pure visual diagram only
- All meaning conveyed through icons and shapes

Quality:
- Resolution: 4K equivalent, 300 DPI print-ready
- Ultra-crisp edges, no blur
- High contrast for black/white printing
"""

THESIS_FIGURES: Dict[str, Dict[str, Any]] = {

    # ============================================================
    # 第2章：数据采集与预处理
    # ============================================================

    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "chapter": 2,
        "description": "农村低压配电网数据采集系统三层架构：平台层→通信层→现场层",
        "prompt": """Create a professional three-tier IoT architecture infographic for rural power grid monitoring system.

SCENE DESCRIPTION:
A vertical three-layer technical architecture diagram showing data flow from field devices to cloud platform. The diagram uses a clean engineering manual aesthetic with realistic 3D icons and technical annotation style.

LAYER 1 - PLATFORM LAYER (Top):
- Large cloud shape containing:
  * Database cylinder icon (data storage)
  * Server rack icon (computing)
  * Brain/neural network icon (anomaly detection AI)
- Blue gradient fill (#0076A8 to light blue)
- Dashed border indicating cloud boundary

LAYER 2 - COMMUNICATION LAYER (Middle):
- Three data concentrator boxes arranged horizontally
- Each box has antenna/wireless symbol on top
- Connected by horizontal dashed lines
- Icons showing: 4G/NB-IoT signal waves
- Orange accent (#E07020) for wireless signals

LAYER 3 - FIELD LAYER (Bottom):
- Row of 6 traditional Chinese rural houses (simple pitched roof style)
- Each house has small smart meter icon attached
- Central transformer symbol (standard electrical transformer icon)
- Power lines connecting all houses through transformer
- Small monitoring sensor icons near power lines

DATA FLOW:
- Thick upward arrows between layers showing data flow direction
- Arrow from field to communication: voltage data (16 features)
- Arrow from communication to platform: aggregated time series

VISUAL STYLE:
- Clean white background
- 3D isometric icons with subtle shadows
- Black technical annotation lines connecting elements
- Professional engineering diagram aesthetic
- NO text labels - pure visual representation

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "1800x1600"
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "chapter": 2,
        "description": "四种典型电压异常波形：骤降(<198V)、骤升(>242V)、闪变、中断",
        "prompt": """Create a 2x2 grid technical diagram showing four types of voltage anomalies in power systems.

SCENE DESCRIPTION:
A professional oscilloscope-style display showing four panels, each demonstrating a different voltage waveform anomaly. Uses realistic signal visualization with engineering annotation overlay.

PANEL LAYOUT (2x2 Grid with thin gray borders):

TOP-LEFT - VOLTAGE SAG (Undervoltage):
- Clean sinusoidal waveform (blue #0076A8 line, 3px thick)
- Normal amplitude at start and end
- Middle section: amplitude drops to ~70% of normal
- Semi-transparent orange (#E07020) rectangle highlighting the sag region
- Horizontal dashed reference lines showing normal vs. reduced amplitude
- X-axis: time progression, Y-axis: voltage amplitude

TOP-RIGHT - VOLTAGE SWELL (Overvoltage):
- Clean sinusoidal waveform (blue line)
- Middle section: amplitude rises to ~120% of normal
- Orange highlight on the swell region
- Reference lines showing normal vs. elevated amplitude

BOTTOM-LEFT - VOLTAGE FLICKER:
- Sinusoidal waveform with modulated envelope
- Amplitude gradually varies up and down (envelope modulation effect)
- Shows 3-4 cycles of the modulation pattern
- Orange shading on varying envelope region

BOTTOM-RIGHT - VOLTAGE INTERRUPTION:
- Sinusoidal waveform at start
- Sudden drop to zero (flat line) in middle section
- Waveform resumes at end
- Orange highlight on the zero-voltage gap

VISUAL ELEMENTS:
- Each panel has coordinate axes (gray thin lines)
- Light gray grid background in each panel
- Waveforms are smooth, clean sine curves
- Engineering graph paper aesthetic
- NO text labels - patterns speak for themselves

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "1800x1400"
    },

    # ============================================================
    # 第3章：模型方法
    # ============================================================

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "chapter": 3,
        "description": "时间序列滑动窗口机制：50步输入窗口，逐步滑动进行预测",
        "prompt": """Create a technical diagram illustrating the sliding window mechanism for time series anomaly detection.

SCENE DESCRIPTION:
A horizontal technical illustration showing how a fixed-size window slides across a continuous voltage time series signal. Engineering annotation style with clean vector graphics.

MAIN ELEMENTS:

TIME SERIES SIGNAL (spanning full width):
- Continuous wavy line representing voltage readings over time
- Blue (#0076A8) thick line (3px)
- Shows clear periodic patterns (sine-like oscillations with some variation)
- Approximately 200 time steps visible
- Subtle gray grid background

THREE WINDOW POSITIONS:
- Window 1 (left): Semi-transparent blue rectangle with blue border
- Window 2 (center): Same style, highlighted more prominently
- Window 3 (right): Same style
- Each window covers exactly 50 time steps (shown by bracket below)
- Windows overlap slightly to show sliding motion

SLIDING DIRECTION:
- Large horizontal arrow below the signal pointing RIGHT
- Shows the direction of window movement
- Arrow has engineering style (clean triangular head)

INPUT-OUTPUT RELATIONSHIP (for center window):
- Bracket underneath center window indicating "Input Window"
- Arrow pointing from window to a single point on the right
- Small dot or marker indicating "Prediction Point"
- Shows: window content → model → next value prediction

VISUAL ANNOTATIONS:
- Dashed vertical lines marking window boundaries
- Step indicators showing window positions
- Clean orthogonal connecting lines

STYLE:
- Pure white background
- Minimal, technical illustration aesthetic
- High contrast for printing
- NO text - pure visual representation

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "2000x1000"
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "一维到二维时序转换示意图",
        "chapter": 3,
        "description": "TimesNet核心：1D时序→FFT周期发现→2D张量重塑",
        "prompt": """Create a three-stage transformation diagram showing the core mechanism of TimesNet: converting 1D time series to 2D representation through FFT period discovery.

SCENE DESCRIPTION:
A horizontal left-to-right transformation pipeline showing three distinct stages connected by large arrows. Technical infographic style with 3D elements and engineering annotations.

STAGE 1 - 1D TIME SERIES (Left):
- Horizontal wavy line representing original voltage signal
- Blue (#0076A8) colored line with periodic oscillations
- Shows approximately 100 time steps
- Small coordinate axes (X: time, Y: amplitude)
- Enclosed in rounded rectangle frame
- Label indicator: "T × C" shape notation

STAGE 2 - FFT FREQUENCY SPECTRUM (Center):
- Vertical bar chart style visualization
- 6-8 bars of different heights
- 2-3 TALL bars in blue (dominant frequencies/periods)
- Remaining bars in light gray (minor frequencies)
- The tall bars represent detected periods (e.g., 60, 300, 900 samples)
- Small coordinate axes (X: frequency, Y: amplitude)
- Enclosed in rounded rectangle frame

STAGE 3 - 2D TENSOR REPRESENTATION (Right):
- Square grid matrix (8×8 or similar)
- Blue gradient color fill showing 2D pattern
- Represents the reshaped time series based on detected period
- Shows how 1D sequence becomes 2D matrix
- Grid cells with varying blue intensity
- Enclosed in rounded rectangle frame
- Label indicator: "p × (T/p) × C" shape notation

TRANSFORMATION ARROWS:
- Large bold arrow from Stage 1 to Stage 2 (FFT transform)
- Large bold arrow from Stage 2 to Stage 3 (Reshape by period)
- Arrows are blue with black outline
- Engineering style arrow heads

VISUAL STYLE:
- Clean white background
- Stages are clearly separated with equal spacing
- Professional technical diagram aesthetic
- NO text labels in the image

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "2200x1000"
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "chapter": 3,
        "description": "端到端异常检测流程：输入→预处理→编码→解码→评分→检测",
        "prompt": """Create a horizontal flowchart showing the complete anomaly detection pipeline using reconstruction-based approach.

SCENE DESCRIPTION:
A professional process flowchart with 6 distinct processing modules connected by arrows, showing data flow from raw input to anomaly detection output. Uses Nano Banana Pro's minimalist flowchart style.

FLOWCHART MODULES (Left to Right):

MODULE 1 - DATA INPUT:
- Rounded rectangle shape
- Inside: Waveform/signal icon (sine wave symbol)
- Represents: Raw voltage time series (16 channels)
- Blue (#0076A8) border, white fill

MODULE 2 - PREPROCESSING:
- Rounded rectangle shape
- Inside: Filter/funnel icon with small gear
- Represents: Normalization, windowing, feature extraction
- Blue border, white fill

MODULE 3 - ENCODER:
- Rounded rectangle shape
- Inside: Three stacked horizontal bars converging (compression)
- Represents: VoltageTimesNet encoder (feature extraction)
- Blue border, light blue fill (emphasized)

MODULE 4 - DECODER:
- Rounded rectangle shape
- Inside: Three stacked horizontal bars diverging (expansion)
- Represents: Reconstruction decoder
- Blue border, light blue fill

MODULE 5 - ANOMALY SCORING:
- Rounded rectangle shape
- Inside: Two overlapping waveforms with difference highlighted
- Represents: Reconstruction error computation (MSE)
- Blue border, white fill

MODULE 6 - DETECTION OUTPUT:
- Pill/capsule shape (end node)
- Inside: Alert/bell icon or checkmark
- Represents: Binary anomaly classification
- Orange (#E07020) border (output highlight), white fill

CONNECTIONS:
- Horizontal arrows between each consecutive module
- Arrows are blue with solid triangle heads
- Equal spacing between modules
- Single straight flow line

ADDITIONAL ELEMENTS:
- Thin dashed line below showing data transformation stages
- Small icons representing data shape at each stage
- Clean orthogonal alignment

STYLE:
- Pure white background
- Minimalist flowchart aesthetic
- High contrast, print-ready
- NO text labels

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "2400x800"
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet网络架构图",
        "chapter": 3,
        "description": "TimesNet基线模型：FFT周期发现+2D卷积+多周期聚合",
        "prompt": """Create a vertical neural network architecture diagram showing the complete TimesNet model structure.

SCENE DESCRIPTION:
A detailed vertical flowchart showing TimesNet's architecture from input to output, including the core TimesBlock with FFT, 2D convolution, and multi-period aggregation. Professional deep learning diagram style.

ARCHITECTURE (Top to Bottom):

INPUT LAYER:
- Rectangle at top
- Shows input tensor shape icon (horizontal bars)
- Blue (#0076A8) fill

EMBEDDING LAYER:
- Small rectangle below input
- Represents: Linear projection to d_model dimensions
- Light blue fill

MAIN TIMESBLOCK (Enclosed in dashed border - repeated L times):

  A. FFT PERIOD DISCOVERY:
  - Small rectangle with wave→bars transformation icon
  - Shows: Time domain → Frequency domain

  B. TOP-K PERIOD SELECTION:
  - Small rectangle with selection/filter icon
  - Represents: Selecting top-k dominant periods

  C. 2D RESHAPE:
  - Icon showing 1D line transforming to 2D grid
  - Represents: Reshape by each detected period

  D. INCEPTION 2D CONVOLUTION (4 parallel paths):
  - Path 1: Small square (1×1 conv)
  - Path 2: Small→Medium squares (1×1→3×3)
  - Path 3: Small→Large squares (1×1→5×5)
  - Path 4: Grid→Small (MaxPool→1×1)
  - All paths merge into concatenation bar

  E. 2D TO 1D RESHAPE:
  - Icon showing 2D grid transforming back to 1D

  F. ADAPTIVE AGGREGATION:
  - Weighted sum icon (multiple inputs with weights)
  - Represents: Softmax-weighted combination of periods

RESIDUAL CONNECTION:
- Curved arrow on the LEFT side
- Connects from before TimesBlock to after
- Dashed blue line

LAYER NORMALIZATION:
- Thin rectangle after residual addition
- Represents: Post-block normalization

REPETITION INDICATOR:
- Dotted vertical line on right side
- Shows "×L" (L layers stacked)

OUTPUT LAYER:
- Rectangle at bottom
- Represents: Final projection layer
- Blue fill

STYLE:
- White background
- Clean vector graphics
- Professional ML architecture diagram style
- NO text labels
- Clear visual hierarchy

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "1400x2200"
    },

    "fig_voltagetimesnet_architecture": {
        "title": "VoltageTimesNet网络架构图",
        "chapter": 3,
        "description": "VoltageTimesNet改进：混合周期发现(70%FFT+30%电网先验)+自适应周期加权",
        "prompt": """Create a side-by-side architecture comparison diagram showing TimesNet (baseline) vs VoltageTimesNet (proposed improvement).

SCENE DESCRIPTION:
Two vertical neural network architectures placed side by side, with the left showing baseline TimesNet and the right showing enhanced VoltageTimesNet with two key improvements highlighted in orange.

LEFT COLUMN - TIMESNET BASELINE:

Vertical flow from top to bottom:
1. INPUT rectangle (blue)
2. EMBEDDING rectangle (light blue)
3. TIMESBLOCK (dashed border):
   - FFT block
   - Period selection
   - 2D Conv (Inception style)
   - Aggregation
4. OUTPUT rectangle (blue)
5. Skip connection curve on left side

All blocks in blue (#0076A8) color scheme

RIGHT COLUMN - VOLTAGETIMESNET (Enhanced):

Same vertical flow with TWO HIGHLIGHTED IMPROVEMENTS:

IMPROVEMENT 1 - HYBRID PERIOD DISCOVERY (after FFT):
- Orange (#E07020) bordered rectangle
- Shows: FFT periods + Grid preset periods merging
- Icon: Wave bars + fixed period markers combining
- Represents: 70% data-driven + 30% domain knowledge
- Preset periods: 60 (1min), 300 (5min), 900 (15min), 3600 (1hr)

IMPROVEMENT 2 - ADAPTIVE PERIOD WEIGHTING (before aggregation):
- Orange (#E07020) bordered rectangle
- Shows: Attention mechanism icon (multiple inputs with varying weights)
- Represents: Learnable importance weighting for different periods
- Icon: Bars with gradient heights showing attention weights

Rest of the architecture same as baseline:
- 2D Conv blocks in blue
- Skip connection on right side
- Output in blue

VISUAL SEPARATION:
- Vertical dashed line between left and right columns
- Clear visual distinction between baseline (all blue) and enhanced (blue + orange highlights)

CONNECTING ELEMENTS:
- Horizontal double-headed arrow between corresponding blocks
- Shows: "Same structure" vs "Enhanced module"

STYLE:
- White background
- Professional neural network diagram
- Orange highlights draw attention to innovations
- NO text labels
- High contrast for printing

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "2000x1800"
    },

    "fig_fft_period_discovery": {
        "title": "快速傅里叶变换周期发现示意图",
        "chapter": 3,
        "description": "FFT变换：时域电压信号→频域→发现主导周期(如60采样点=1分钟周期)",
        "prompt": """Create a two-panel FFT analysis diagram showing the transformation from time domain to frequency domain for period discovery.

SCENE DESCRIPTION:
A vertical two-panel technical illustration showing how FFT reveals hidden periodic patterns in voltage time series data. Top panel shows the original signal, bottom panel shows the frequency spectrum with detected periods.

TOP PANEL - TIME DOMAIN SIGNAL:

- Large rectangular plot area with light gray grid
- X-axis: Time (showing ~200 time steps)
- Y-axis: Voltage amplitude
- SIGNAL: Blue (#0076A8) continuous wavy line
  * Shows clear but complex periodic pattern
  * Multiple overlapping periodicities visible
  * Resembles real voltage data with daily/hourly cycles
- Coordinate axes in dark gray
- Clean engineering graph aesthetic

TRANSFORMATION ARROW:

- Large bold downward arrow between panels
- Blue with black outline
- Shows FFT transformation direction
- Arrow contains small wave→bar icon

BOTTOM PANEL - FREQUENCY SPECTRUM:

- Large rectangular plot area with light gray grid
- X-axis: Frequency (or Period in samples)
- Y-axis: Amplitude/Power
- SPECTRUM as vertical bar chart:
  * 2-3 TALL prominent bars in blue (#0076A8) - dominant periods
    - These represent: 60 samples (1-minute cycle)
    - 300 samples (5-minute cycle)
    - 900 samples (15-minute cycle)
  * Orange (#E07020) circles/markers on top of dominant bars
  * Remaining ~10 shorter bars in light gray - minor frequencies
- Clear visual hierarchy: dominant peaks stand out

VISUAL ANNOTATIONS:
- Dashed horizontal reference lines on frequency panel
- Small arrows pointing to dominant peaks
- Bracket showing "Top-K selected" range

STYLE:
- White background
- Scientific visualization aesthetic
- Print-ready high contrast
- NO text labels

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "1600x1600"
    },

    "fig_2d_conv_inception": {
        "title": "二维卷积Inception模块示意图",
        "chapter": 3,
        "description": "Inception Block：多尺度2D卷积(1×1, 3×3, 5×5)+MaxPool并行处理",
        "prompt": """Create a detailed Inception module structure diagram showing multi-scale 2D convolution paths.

SCENE DESCRIPTION:
A vertical neural network module diagram showing the Inception-style architecture used in TimesNet for multi-scale temporal pattern extraction. Shows four parallel processing paths with different receptive field sizes.

STRUCTURE (Top to Bottom):

INPUT (Top):
- Wide rectangle representing input 2D feature map
- Blue (#0076A8) fill
- Shows: H × W × C tensor shape icon

FOUR PARALLEL PATHS (arranged horizontally):

PATH 1 - 1×1 Convolution:
- Single small blue square
- Represents: Point-wise convolution (channel mixing)
- Smallest receptive field

PATH 2 - 1×1 → 3×3 Convolution:
- Small square (1×1) connected to medium square (3×3)
- Two-stage processing
- Vertical arrow between them
- Medium receptive field

PATH 3 - 1×1 → 5×5 Convolution:
- Small square (1×1) connected to large square (5×5)
- Two-stage processing
- Vertical arrow between them
- Large receptive field

PATH 4 - MaxPool → 1×1:
- Grid pattern icon (3×3 max pooling) connected to small square (1×1)
- Two-stage processing
- Captures local max features

VISUAL REPRESENTATION:
- Square sizes literally represent kernel sizes
- Path 1 square is smallest
- Path 3 square is largest
- Path 4 grid pattern distinct from convolutions

CONCATENATION (Merge point):
- All four paths converge via arrows pointing down
- Wide horizontal rectangle showing concatenated features
- Different colored sections showing contribution from each path
- Orange (#E07020) border highlighting the fusion

OUTPUT (Bottom):
- Rectangle representing output feature map
- Blue fill
- Shows: H × W × (C1+C2+C3+C4) combined channels

CONNECTING ARROWS:
- Vertical arrows showing data flow
- From input splitting to 4 paths
- From 4 paths merging to concatenation
- From concatenation to output

STYLE:
- White background
- Clean vector graphics
- Professional deep learning diagram
- Different visual weights for different kernel sizes
- NO text labels

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "1600x2000"
    },

    "fig_3_3_voltage_timesnet_comparison": {
        "title": "VoltageTimesNet与TimesNet周期检测对比",
        "chapter": 3,
        "description": "对比：纯FFT周期 vs 混合周期(FFT+电网先验)；均匀权重 vs 自适应注意力权重",
        "prompt": """Create a 2×2 comparison grid showing the key differences between TimesNet and VoltageTimesNet in period detection and weighting.

SCENE DESCRIPTION:
A four-panel comparison diagram illustrating two key improvements: (1) hybrid period discovery vs pure FFT, and (2) adaptive weighting vs uniform weighting. Left column shows TimesNet baseline, right column shows VoltageTimesNet improvements.

PANEL LAYOUT:

TOP-LEFT - TimesNet Period Detection:
- Frequency spectrum bar chart
- 6-8 bars of varying heights
- Only 2-3 BLUE bars selected (pure FFT top-k)
- Remaining bars in light gray (not selected)
- Shows: Only data-driven period selection
- Simple selection based on amplitude

TOP-RIGHT - VoltageTimesNet Hybrid Period Detection:
- Same frequency spectrum base
- BLUE bars from FFT (data-driven)
- PLUS additional ORANGE bars at fixed positions
  * These represent grid-specific periods:
  * 60 samples (1-minute)
  * 300 samples (5-minute)
  * 900 samples (15-minute)
  * 3600 samples (1-hour)
- Orange (#E07020) markers/circles on preset periods
- Shows: 70% FFT + 30% domain knowledge

BOTTOM-LEFT - TimesNet Uniform Weighting:
- Bar chart showing period weights
- ALL bars have EQUAL height
- Uniform blue color
- Shows: Equal importance for all periods
- No adaptation based on data

BOTTOM-RIGHT - VoltageTimesNet Adaptive Weighting:
- Bar chart showing period weights
- Bars have VARYING heights (attention weights)
- Important periods (e.g., daily cycle) have taller bars
- Less important periods have shorter bars
- Gradient colors from blue to light blue based on weight
- Small attention icon above (softmax symbol)
- Shows: Learnable, data-dependent weighting

VISUAL SEPARATORS:
- Vertical dashed line between left and right columns
- Horizontal dashed line between top and bottom rows
- Clear grid structure

COMPARISON INDICATORS:
- Left column: Gray/blue scheme (baseline)
- Right column: Blue + Orange scheme (improved)
- Visual emphasis on the differences

STYLE:
- White background
- Clean chart aesthetic
- High contrast for printing
- NO text labels
- Clear visual distinction between baseline and improved

""" + NANO_BANANA_STYLE_GUIDE,
        "size": "2000x1600"
    },
}


def call_google_api(prompt: str, model: str = "gemini-2.0-flash-exp-image-generation") -> Optional[bytes]:
    """
    直接调用 Google AI Studio API 生成图像
    """
    # 构建 API URL
    api_url = f"{GOOGLE_API_BASE_URL}/models/{model}:generateContent?key={GOOGLE_API_KEY}"

    headers = {
        "Content-Type": "application/json",
    }

    # Google API 请求格式
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.7,
        }
    }

    try:
        print(f"    调用 Google API: {model}")
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=180,
            proxies=PROXY_CONFIG
        )

        if response.status_code == 200:
            result = response.json()

            # 解析 Google API 响应
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                for part in parts:
                    # 检查是否有图像数据
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        mime_type = inline_data.get("mimeType", "")
                        data_b64 = inline_data.get("data", "")

                        if "image" in mime_type and data_b64:
                            print(f"    ✓ 成功获取 Google API 图像 (长度: {len(data_b64)} 字符)")
                            return base64.b64decode(data_b64)

                # 如果没有图像，检查是否有文本响应
                for part in parts:
                    if "text" in part:
                        print(f"    ℹ Google API 仅返回文本: {part['text'][:100]}...")

            return None

        else:
            error_text = response.text[:500]
            print(f"    ✗ Google API HTTP {response.status_code}: {error_text}")
            return None

    except Exception as e:
        print(f"    ✗ Google API 请求错误: {e}")
        return None


def call_imagen_api(prompt: str) -> Optional[bytes]:
    """
    调用 Google Imagen 3 API 生成图像
    """
    model = "imagen-3.0-generate-002"
    api_url = f"{GOOGLE_API_BASE_URL}/models/{model}:predict?key={GOOGLE_API_KEY}"

    headers = {
        "Content-Type": "application/json",
    }

    # Imagen API 请求格式
    data = {
        "instances": [
            {"prompt": prompt}
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "personGeneration": "dont_allow",
            "safetySetting": "block_few",
        }
    }

    try:
        print(f"    调用 Imagen 3 API...")
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=180,
            proxies=PROXY_CONFIG
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])

            if predictions:
                for pred in predictions:
                    if "bytesBase64Encoded" in pred:
                        data_b64 = pred["bytesBase64Encoded"]
                        print(f"    ✓ 成功获取 Imagen 3 图像 (长度: {len(data_b64)} 字符)")
                        return base64.b64decode(data_b64)

            return None
        else:
            error_text = response.text[:500]
            print(f"    ✗ Imagen API HTTP {response.status_code}: {error_text}")
            return None

    except Exception as e:
        print(f"    ✗ Imagen API 请求错误: {e}")
        return None


def call_image_generation_api(prompt: str, model: str, size: str = "1792x1024") -> Optional[bytes]:
    """
    调用 OpenRouter 图像生成 API
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rural-voltage-detection",
        "X-Title": "Rural Voltage Detection Thesis Figures"
    }

    # 解析尺寸
    width, height = map(int, size.split("x"))

    # 使用简单的字符串消息格式（Gemini 图像生成需要）
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=180,
            proxies=PROXY_CONFIG
        )

        if response.status_code == 200:
            result = response.json()

            # 检查是否有错误
            if "error" in result:
                print(f"    ✗ API 错误: {result['error']}")
                return None

            # 调试：打印完整响应结构
            print(f"    [调试] 响应结构: {list(result.keys())}")

            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            # 首先检查 message.images 数组（Gemini 返回格式）
            images = message.get("images", [])
            if images:
                print(f"    [调试] 找到 {len(images)} 个图像")
                for img in images:
                    if isinstance(img, dict):
                        img_type = img.get("type", "")
                        if img_type == "image_url" or "image" in img_type:
                            image_url = img.get("image_url", {}).get("url", "")
                            if image_url and image_url.startswith("data:image"):
                                base64_data = image_url.split(",")[1]
                                print(f"    ✓ 成功提取 Gemini 图像 (长度: {len(base64_data)} 字符)")
                                return base64.b64decode(base64_data)

            # 调试：检查内容类型
            print(f"    [调试] 内容类型: {type(content)}")
            if isinstance(content, list):
                print(f"    [调试] 内容列表长度: {len(content)}")
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        print(f"    [调试] 项目{i}类型: {item.get('type', 'unknown')}")

            # 检查是否返回了图像（base64 格式）
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")

                        # 处理 inline_data 格式（Gemini 常用）
                        if "inline_data" in item:
                            inline = item["inline_data"]
                            if "data" in inline:
                                print(f"    ✓ 找到 inline_data 图像")
                                return base64.b64decode(inline["data"])

                        # 处理 image_url 格式
                        if item_type == "image_url" or "image" in item_type:
                            image_url = item.get("image_url", {}).get("url", "")
                            if not image_url and "url" in item:
                                image_url = item["url"]
                            if image_url:
                                if image_url.startswith("data:image"):
                                    base64_data = image_url.split(",")[1]
                                    print(f"    ✓ 找到 base64 图像 URL")
                                    return base64.b64decode(base64_data)
                                else:
                                    # 下载图像
                                    print(f"    ⬇ 下载图像: {image_url[:50]}...")
                                    img_resp = requests.get(image_url, timeout=60, proxies=PROXY_CONFIG)
                                    if img_resp.status_code == 200:
                                        return img_resp.content

                        # 处理 image 类型
                        if item_type == "image" and "data" in item:
                            print(f"    ✓ 找到 image 数据")
                            return base64.b64decode(item["data"])

            # 检查是否在文本中返回了 base64 图像
            if isinstance(content, str):
                # 尝试提取 base64 图像数据
                base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                match = re.search(base64_pattern, content)
                if match:
                    print(f"    ✓ 从文本中提取到 base64 图像")
                    return base64.b64decode(match.group(1))

            # 如果只有文本响应，返回 None 而不是文本
            if content:
                print(f"    ℹ 模型仅返回文本响应，无图像")
            return None

        elif response.status_code == 429:
            print(f"    ⏳ 速率限制，等待30秒后重试...")
            time.sleep(30)
            return None
        else:
            print(f"    ✗ HTTP {response.status_code}: {response.text[:300]}")
            return None

    except Exception as e:
        print(f"    ✗ 请求错误: {e}")
        return None


def generate_figure(figure_key: str, output_dir: Path, models: List[str] = None, provider: str = None) -> Dict[str, Any]:
    """
    生成单个图表
    """
    if figure_key not in THESIS_FIGURES:
        print(f"未知的图表键: {figure_key}")
        return {"success": False, "error": "Unknown figure key"}

    config = THESIS_FIGURES[figure_key]
    provider = provider or API_PROVIDER

    print(f"\n{'='*60}")
    print(f"正在生成: {config['title']}")
    print(f"章节: 第{config['chapter']}章")
    print(f"描述: {config['description']}")
    print(f"API 提供商: {provider}")
    print(f"{'='*60}")

    result = {
        "key": figure_key,
        "title": config["title"],
        "chapter": config["chapter"],
        "success": False,
        "model_used": None,
        "output_path": None,
        "timestamp": datetime.now().isoformat()
    }

    response = None

    if provider == "google":
        # 使用 Google AI Studio API
        models_to_try = models or GOOGLE_IMAGE_MODELS

        for model in models_to_try:
            print(f"  尝试模型: {model}")

            if "imagen" in model.lower():
                response = call_imagen_api(config["prompt"])
            else:
                response = call_google_api(config["prompt"], model)

            if response and isinstance(response, bytes) and len(response) > 1000:
                output_path = output_dir / f"{figure_key}.png"
                with open(output_path, "wb") as f:
                    f.write(response)
                print(f"  ✓ 图像已保存: {output_path}")
                result["success"] = True
                result["model_used"] = model
                result["output_path"] = str(output_path)
                result["file_size"] = len(response)
                return result

            time.sleep(2)

    else:
        # 使用 OpenRouter API
        models_to_try = models or OPENROUTER_IMAGE_MODELS

        for model in models_to_try:
            print(f"  尝试模型: {model}")

            response = call_image_generation_api(
                config["prompt"],
                model,
                config.get("size", "1800x1200")
            )

            if response:
                if isinstance(response, bytes) and len(response) > 1000:
                    output_path = output_dir / f"{figure_key}.png"
                    with open(output_path, "wb") as f:
                        f.write(response)
                    print(f"  ✓ 图像已保存: {output_path}")
                    result["success"] = True
                    result["model_used"] = model
                    result["output_path"] = str(output_path)
                    result["file_size"] = len(response)
                    return result
                else:
                    text_path = output_dir / f"{figure_key}_response.txt"
                    if isinstance(response, bytes):
                        response = response.decode("utf-8", errors="ignore")
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(f"Model: {model}\n\n{response}")
                    print(f"  ℹ 模型返回文本响应: {text_path}")

            time.sleep(3)

    if not result["success"]:
        print(f"  ✗ 所有模型均未能生成图像")
        result["error"] = "All models failed"

    return result


def generate_all_figures(output_dir: Optional[Path] = None, chapters: List[int] = None) -> None:
    """
    生成所有或指定章节的图表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "thesis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("论文图表批量生成")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"API 提供商: {API_PROVIDER}")
    if API_PROVIDER == "google":
        print(f"图像模型: {GOOGLE_IMAGE_MODELS}")
    else:
        print(f"图像模型: {OPENROUTER_IMAGE_MODELS}")
    if chapters:
        print(f"生成章节: {chapters}")
    print("=" * 70)

    # 筛选要生成的图表
    figures_to_generate = THESIS_FIGURES
    if chapters:
        figures_to_generate = {
            k: v for k, v in THESIS_FIGURES.items()
            if v["chapter"] in chapters
        }

    results = []
    for i, (figure_key, config) in enumerate(figures_to_generate.items(), 1):
        print(f"\n[{i}/{len(figures_to_generate)}] ", end="")
        result = generate_figure(figure_key, output_dir)
        results.append(result)
        time.sleep(5)  # 请求间隔

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_figures": len(figures_to_generate),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "output_directory": str(output_dir),
        "results": results
    }

    report_path = output_dir / "generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"生成完成！")
    print(f"  成功: {report['successful']}/{report['total_figures']}")
    print(f"  失败: {report['failed']}/{report['total_figures']}")
    print(f"  报告: {report_path}")
    print("=" * 70)


def list_figures() -> None:
    """
    列出所有可用的图表
    """
    print("\n可用的论文图表:")
    print("=" * 70)

    by_chapter = {}
    for key, config in THESIS_FIGURES.items():
        ch = config["chapter"]
        if ch not in by_chapter:
            by_chapter[ch] = []
        by_chapter[ch].append((key, config))

    for chapter in sorted(by_chapter.keys()):
        print(f"\n第{chapter}章:")
        print("-" * 40)
        for key, config in by_chapter[chapter]:
            print(f"  {key}")
            print(f"    标题: {config['title']}")
            print(f"    尺寸: {config.get('size', '1800x1200')}")

    print("\n" + "=" * 70)
    print(f"总计: {len(THESIS_FIGURES)} 个图表")


def export_prompts(output_file: Optional[Path] = None) -> None:
    """
    导出所有提示词到 JSON 文件
    """
    if output_file is None:
        output_file = Path(__file__).parent / "thesis_prompts_chinese.json"

    export_data = []
    for key, config in THESIS_FIGURES.items():
        export_data.append({
            "id": key,
            "title": config["title"],
            "chapter": config["chapter"],
            "description": config["description"],
            "prompt": config["prompt"],
            "size": config.get("size", "1800x1200"),
            "style": "IEEE学术风格，中文标注，宋体字",
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"提示词已导出到: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="论文图表生成器 - 通过 OpenRouter API 生成学术图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_thesis_figures.py --list                # 列出所有图表
  python generate_thesis_figures.py --all                 # 生成所有图表
  python generate_thesis_figures.py --chapter 3           # 生成第3章图表
  python generate_thesis_figures.py --figure fig_timesnet_architecture  # 生成单个图表
  python generate_thesis_figures.py --export              # 导出提示词
"""
    )

    parser.add_argument("--list", action="store_true", help="列出所有可用图表")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--chapter", type=int, nargs="+", help="生成指定章节的图表")
    parser.add_argument("--figure", type=str, help="生成指定的单个图表")
    parser.add_argument("--export", action="store_true", help="导出提示词到 JSON")
    parser.add_argument("--output", type=str, help="输出目录")

    args = parser.parse_args()

    if args.list:
        list_figures()
    elif args.export:
        export_prompts()
    elif args.figure:
        output_dir = Path(args.output) if args.output else Path(__file__).parent / "output" / "thesis_figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_figure(args.figure, output_dir)
    elif args.all:
        output_dir = Path(args.output) if args.output else None
        generate_all_figures(output_dir)
    elif args.chapter:
        output_dir = Path(args.output) if args.output else None
        generate_all_figures(output_dir, chapters=args.chapter)
    else:
        parser.print_help()
