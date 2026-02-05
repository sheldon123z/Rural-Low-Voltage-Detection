# Nano Banana Pro 学术图表生成提示词

> 农村低压配电网电压异常检测论文 - IEEE Smart Grid 学术风格

## 使用说明

1. 打开 [Nano Banana Pro](https://nanobanana.ai/) 或 Google AI Studio
2. 选择 **Thought Mode** 或 **Pro Mode** 以获得更稳定的输出
3. 复制下方的提示词粘贴到输入框
4. 生成后下载 PNG 格式图片
5. 确保分辨率为 300 DPI 以满足论文印刷要求

## 统一风格指南

所有图表需遵循以下风格：
- **配色**: IEEE 蓝 (#0076A8)、深灰 (#333333)、白色背景
- **风格**: 扁平化、矢量风格、清晰锐利的边缘
- **禁止**: 任何文字标签、注释、装饰元素
- **用途**: 学术论文插图（IEEE Transactions on Smart Grid 风格）

---

## 第2章：数据采集与预处理

### Fig 2-1: 数据采集分层架构图

```
Generate a professional technical diagram showing a three-layer architecture for rural power distribution network data collection system.

STRUCTURE (vertical, top to bottom):

LAYER 3 - TOP (Platform Layer):
- Central cloud/server icon in IEEE blue (#0076A8)
- Small database cylinder icon next to it
- Represent the data center and analytics platform

LAYER 2 - MIDDLE (Communication Layer):
- 3 wireless router/antenna icons evenly spaced
- Connected by horizontal lines
- Show data concentration points

LAYER 1 - BOTTOM (Field Layer):
- Row of 6 simple house silhouettes
- Small meter symbol (circle with line) at each house
- Horizontal power line connecting all houses
- Distribution transformer symbol in center

CONNECTIONS:
- Vertical blue arrows between layers (pointing upward = data flow)
- Thin gray lines for infrastructure connections

VISUAL STYLE:
- Pure white background (#FFFFFF)
- Primary color: IEEE blue (#0076A8)
- Secondary: Dark gray (#333333)
- Flat design, no shadows, no gradients
- Clean vector-like appearance
- NO text, NO labels, NO annotations
- High resolution, suitable for academic publication

OUTPUT: Technical architecture diagram, white background, 4K quality
```

### Fig 2-2: 电压异常类型示意图

```
Generate a 2x2 grid showing four types of voltage anomaly waveforms for an IEEE academic paper.

LAYOUT: 4 equal panels in 2x2 grid

TOP-LEFT PANEL (Voltage Sag):
- Sine wave with sudden amplitude drop (to ~70%) in the middle section
- Normal regions in IEEE blue (#0076A8)
- Dip region highlighted with light orange tint (#FFE4B5)
- Clean X-Y coordinate axes (no labels)

TOP-RIGHT PANEL (Voltage Swell):
- Sine wave with elevated amplitude section (to ~120%)
- Normal regions in IEEE blue
- Swell region highlighted with light orange tint
- Clean X-Y coordinate axes

BOTTOM-LEFT PANEL (Voltage Flicker):
- Sine wave with oscillating amplitude envelope
- Amplitude varies periodically up and down
- IEEE blue color throughout
- Shows amplitude modulation effect

BOTTOM-RIGHT PANEL (Voltage Interruption):
- Sine wave with zero-voltage gap in the middle
- Clear interruption where wave goes to zero
- Interruption region in light orange tint
- Recovery transient visible

EACH PANEL:
- Light gray grid lines for reference
- Horizontal dashed line for nominal voltage level
- Clean axis lines without text labels

VISUAL STYLE:
- White background, flat design
- IEEE blue (#0076A8) for waveforms
- Light orange (#FFE4B5) for anomaly highlights
- NO text, NO labels, NO annotations
- Professional, clean, suitable for print
```

---

## 第3章：模型方法

### Fig 3-1: 滑动窗口预测示意图

```
Generate a technical diagram showing the sliding window mechanism for time series processing.

MAIN ELEMENTS:

1. TIME SERIES SIGNAL:
- Long horizontal continuous wavy line (voltage-like signal)
- Spans full width of image
- IEEE blue (#0076A8) color
- Shows realistic oscillation pattern

2. SLIDING WINDOWS (3 windows):
- Three rectangular frames at different positions along the signal
- Left window (past position) - light blue fill, blue border
- Center window (current position) - medium blue fill, blue border
- Right window (future position) - light blue fill, blue border
- Equal spacing between windows
- Semi-transparent fill so signal is visible through

3. DIRECTION ARROW:
- Large horizontal arrow below the windows
- Points from left to right
- Shows sliding direction
- Dark gray color

4. INPUT-OUTPUT:
- Bracket symbol under center window (input region)
- Small dot or short line segment to the right (output prediction)
- Arrow connecting input bracket to output

VISUAL STYLE:
- Pure white background
- Clean, minimal design
- Flat design, no 3D effects
- NO text labels or annotations
- Professional technical illustration
```

### Fig 3-2: 1D到2D时序转换示意图

```
Generate a three-stage transformation diagram showing 1D to 2D conversion for TimesNet.

LAYOUT: Three distinct sections arranged horizontally (left to right)

STAGE 1 (Left) - 1D TIME SERIES:
- Horizontal wavy line representing time series signal
- Blue color (#0076A8)
- Shows periodic pattern in the oscillation
- Represents: (B, T, C) tensor

STAGE 2 (Middle) - FREQUENCY SPECTRUM:
- Vertical bar chart (frequency domain)
- 5-6 bars of varying heights
- 2-3 tallest bars in IEEE blue (dominant frequencies)
- Remaining bars in light gray
- Represents FFT analysis result

STAGE 3 (Right) - 2D MATRIX:
- 8x8 or 10x10 grid/matrix
- Cells colored as heatmap (blue gradient: white to IEEE blue)
- Shows the reshaped 2D representation
- Visible periodic pattern in the grid
- Represents: (B, p, T/p, C) tensor

CONNECTIONS:
- Arrow from Stage 1 to Stage 2 (FFT transformation)
- Arrow from Stage 2 to Stage 3 (Reshape operation)
- Arrows in dark gray

VISUAL STYLE:
- White background, horizontal flow
- Clean, professional, flat design
- NO text, NO labels
- Suitable for IEEE publication
```

### Fig 3-3: VoltageTimesNet vs TimesNet 对比

```
Generate a side-by-side comparison diagram of two neural network architectures.

LAYOUT: Two parallel vertical flows, separated by vertical dashed line

LEFT SIDE - STANDARD TIMESNET:
- Simple vertical flow
- Boxes connected by arrows (top to bottom):
  * Input box (rectangle)
  * FFT box (small square)
  * Reshape box (rectangle)
  * Conv block (4 parallel small boxes merging)
  * Output box (rectangle)
- All boxes in IEEE blue (#0076A8)
- Skip connection (curved arrow) from input to output

RIGHT SIDE - VOLTAGETIMESNET (Enhanced):
- Same structure as left
- PLUS two additional highlighted boxes in ORANGE (#E07020):
  * "Domain Prior" box after FFT
  * "Weight Enhancement" box before output
- Orange boxes stand out from blue
- Same skip connection

VISUAL ELEMENTS:
- Vertical dashed gray line separating left and right
- Arrows showing data flow (top to bottom)
- Skip connections as curved arrows on the sides

VISUAL STYLE:
- White background
- IEEE blue for standard components
- Orange (#E07020) for enhancements
- NO text, pure visual comparison
```

### Fig 3-7: 异常检测框架流程图

```
Generate a horizontal flowchart showing the end-to-end anomaly detection pipeline.

STRUCTURE: 6 connected blocks in a single horizontal row

BLOCKS (left to right):
1. INPUT BLOCK: Small waveform icon inside rounded rectangle
2. PREPROCESS BLOCK: Filter/funnel icon inside rounded rectangle
3. FEATURE EXTRACT BLOCK: 3 stacked horizontal bars (neural layers) inside rounded rectangle
4. RECONSTRUCT BLOCK: Expanding/diverging bars inside rounded rectangle
5. SCORING BLOCK: Two overlapping shapes or difference symbol inside rounded rectangle
6. OUTPUT BLOCK: Checkmark or alert icon inside rounded rectangle

CONNECTIONS:
- Horizontal arrows connecting each block
- Arrows in dark gray
- All blocks same size, evenly spaced

BLOCK STYLING:
- Rounded rectangles with IEEE blue (#0076A8) border
- Light blue or white fill
- Icons in dark gray

VISUAL STYLE:
- Pure white background
- Horizontal pipeline layout
- Clean, minimal, professional
- NO text labels or annotations
- Suitable for academic publication
```

### Fig: TimesNet 网络架构图

```
Generate a detailed neural network architecture diagram for TimesNet model.

STRUCTURE (vertical, top to bottom):

INPUT LAYER:
- Rectangle at top representing input tensor
- IEEE blue fill

TIMESBLOCK MODULE (main component, show in detail):
- FFT Sub-block: Small rectangle (frequency analysis)
- Down arrow
- Reshape Icon: 1D to 2D transformation symbol (rectangle becoming grid)
- Down arrow
- 2D CONV with INCEPTION structure:
  * 4 parallel vertical paths side by side
  * Different widths (representing 1x1, 3x3, 5x5, pooling paths)
  * All paths merge at bottom into single block
- Down arrow
- Aggregation block: Rectangle
- SKIP CONNECTION: Curved arrow from block input to block output (on the right side)
- Layer Norm: Small rectangle

REPETITION:
- Dotted horizontal line
- "×N" suggestion through visual repetition symbol

OUTPUT LAYER:
- Rectangle at bottom representing output tensor
- IEEE blue fill

VISUAL STYLE:
- White background
- IEEE blue (#0076A8) for main flow
- Light gray for skip connections
- Flat design, clean lines
- NO text annotations
```

### Fig: FFT 周期发现示意图

```
Generate a two-panel technical illustration showing FFT period discovery.

LAYOUT: Two panels stacked vertically

TOP PANEL - TIME DOMAIN:
- Horizontal continuous waveform
- Shows clear periodic pattern (repeating oscillations)
- IEEE blue (#0076A8) color
- Clean coordinate axes (no labels)
- X-axis represents time
- Y-axis represents amplitude

BOTTOM PANEL - FREQUENCY DOMAIN:
- Bar chart / spectrum plot
- X-axis represents frequency
- Y-axis represents magnitude
- 2-3 tall prominent peaks in IEEE blue (dominant frequencies)
- Other frequencies as shorter bars in light gray
- Clear separation between signal peaks and noise floor

CONNECTION:
- Large downward arrow between panels
- Arrow in dark gray
- Represents FFT transformation

VISUAL STYLE:
- White background
- Professional, clean appearance
- NO text, NO labels, NO annotations
- High contrast for print reproduction
- Suitable for IEEE publication
```

### Fig: 2D卷积 Inception 模块示意图

```
Generate a technical diagram of an Inception-style convolution module.

STRUCTURE (vertical flow):

INPUT:
- Single rectangle at TOP
- IEEE blue fill

FOUR PARALLEL BRANCHES (side by side):
Branch 1 (leftmost): Single small square (1×1 conv)
Branch 2: Small square → Medium square (1×1 → 3×3 conv)
Branch 3: Small square → Large square (1×1 → 5×5 conv)
Branch 4 (rightmost): Grid pattern (max pool) → Small square (1×1 conv)

- All branches start from input
- Run vertically downward in parallel
- Different sizes indicate different receptive fields

CONCATENATION:
- All 4 branches merge into single WIDE rectangle
- Shows feature concatenation

OUTPUT:
- Single rectangle at BOTTOM
- Same width as concatenated layer

VISUAL ELEMENTS:
- Down arrows showing data flow
- All conv blocks in IEEE blue
- Pooling block in light gray
- Clean, symmetrical layout

VISUAL STYLE:
- White background
- Flat design, no shadows
- Professional technical illustration
- NO text or annotations
```

---

## 使用技巧

1. **模式选择**: 使用 Thought Mode 或 Pro Mode 获得更稳定的输出
2. **迭代优化**: 如果第一次生成不理想，可以添加更具体的描述重新生成
3. **分辨率**: 下载时选择最高分辨率选项
4. **后处理**: 可在 Figma 或 PowerPoint 中进行微调和添加中文标注

## 常见问题

Q: 图像中出现了文字怎么办？
A: 在提示词末尾强调 "NO text, NO labels, absolutely no text or annotations on the image"

Q: 颜色不够准确？
A: 明确指定 hex 颜色代码，如 "IEEE blue (#0076A8)"

Q: 风格不够学术？
A: 添加 "IEEE academic publication style, technical diagram, vector-like, flat design, professional"
