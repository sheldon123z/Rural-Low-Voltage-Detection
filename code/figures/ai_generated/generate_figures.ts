/**
 * IEEE Smart Grid 学术风格图表生成器
 *
 * 使用 OpenRouter API 通过 AI 模型生成学术论文图表
 * 风格：电气工程学术期刊风格（IEEE Transactions on Smart Grid）
 */

import OpenRouter from '@openrouter/sdk';
import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';

// 加载环境变量
config({ path: path.join(__dirname, '../../../.env') });

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

if (!OPENROUTER_API_KEY) {
  console.error('错误: 未找到 OPENROUTER_API_KEY 环境变量');
  console.error('请确保 .env 文件存在并包含正确的 API 密钥');
  process.exit(1);
}

// 初始化 OpenRouter 客户端
const client = new OpenRouter({
  apiKey: OPENROUTER_API_KEY
});

// IEEE 学术风格基础参数
const IEEE_STYLE_BASE = `
Style: Professional IEEE academic illustration for power systems research.
Visual characteristics:
- Clean vector-like appearance with sharp edges
- Color palette: IEEE blue (#0076A8), dark gray (#333333), light gray (#E5E5E5), white background
- Minimal shadows, flat design aesthetic
- Sans-serif technical labels (Arial/Helvetica style)
- High contrast for print reproduction
- No decorative elements, purely technical
- No text labels or annotations on the image
Resolution: 4K, ultra-sharp, print-quality
Background: Pure white (#FFFFFF)
`;

// 图表提示词定义
interface FigurePrompt {
  title: string;
  description: string;
  prompt: string;
  negativePrompt?: string;
}

const FIGURE_PROMPTS: Record<string, FigurePrompt> = {
  "fig_2_1_data_collection_architecture": {
    title: "数据采集分层架构图",
    description: "Rural Low-Voltage Distribution Network Data Collection Architecture",
    prompt: `
Create a professional technical diagram showing a three-layer data collection architecture for rural low-voltage power distribution network monitoring.

Structure (top to bottom):
LAYER 1 - Field Layer:
- Smart meters icon (simplified meter symbols)
- Voltage sensors distributed along power lines
- Multiple rural households connected to distribution transformers
- Show 3-phase power lines (A, B, C phases) in different shades of blue

LAYER 2 - Communication Layer:
- Data concentrators collecting from field devices
- Wireless communication symbols (antenna icons)
- Data aggregation nodes

LAYER 3 - Platform Layer:
- Central data server/cloud platform icon
- Database storage icon
- Data processing module
- Anomaly detection system block

Connection arrows: Vertical data flow arrows between layers
${IEEE_STYLE_BASE}
Layout: Vertical hierarchy, balanced composition, clean technical diagram
`,
    negativePrompt: "cartoon, 3D realistic, photorealistic, shadows, gradients, decorative elements, text, labels, Chinese characters, words"
  },

  "fig_2_2_voltage_anomaly_types": {
    title: "电压异常类型示意图",
    description: "Voltage Anomaly Types in Rural Distribution Networks",
    prompt: `
Create a technical illustration showing 4 types of voltage anomalies in a 2x2 grid layout.

Grid layout (4 panels, each showing a waveform):
Panel 1 (Top-Left) - Voltage Sag: Sine wave with sudden dip to 70% amplitude in the middle
Panel 2 (Top-Right) - Voltage Swell: Sine wave with elevated section to 120% amplitude
Panel 3 (Bottom-Left) - Voltage Flicker: Sine wave with oscillating envelope (amplitude modulation)
Panel 4 (Bottom-Right) - Voltage Interruption: Sine wave with zero-voltage gap in the middle

Each panel should have:
- Clean coordinate axes
- Horizontal reference line for nominal voltage
- Anomaly region highlighted in light orange/red tint
- Grid lines for professional appearance
${IEEE_STYLE_BASE}
Color coding: Normal voltage in IEEE blue, anomaly regions in muted orange
`,
    negativePrompt: "3D, photorealistic, cartoon, excessive colors, decorative, hand-drawn, sketchy, text labels"
  },

  "fig_3_1_sliding_window": {
    title: "滑动窗口预测示意图",
    description: "Sliding Window Mechanism for Time Series Prediction",
    prompt: `
Create a technical diagram illustrating the sliding window mechanism for time series processing.

Elements:
1. Long horizontal time series signal (voltage-like waveform)
2. Three rectangular window frames shown at different positions (past, current, future)
   - Windows highlighted in IEEE blue with semi-transparent fill
   - Windows equally spaced showing the sliding motion
3. Arrows showing window movement direction (left to right)
4. Input-output relationship: bracket showing input window, arrow pointing to output point
${IEEE_STYLE_BASE}
Layout: Horizontal composition showing temporal progression
`,
    negativePrompt: "3D perspective, photorealistic, cartoon style, excessive decoration, text, labels"
  },

  "fig_3_2_1d_to_2d_conversion": {
    title: "1D到2D时序转换示意图",
    description: "1D to 2D Time Series Transformation",
    prompt: `
Create a technical diagram showing transformation from 1D time series to 2D matrix representation.

Three stages (left to right):
STAGE 1: Long horizontal 1D waveform (time series signal)
STAGE 2: Frequency spectrum bar chart showing dominant peaks
STAGE 3: 2D matrix/grid showing the reshaped signal as a heat map

Arrows between stages showing the transformation flow:
- Stage 1 to 2: FFT transformation
- Stage 2 to 3: Reshape operation

Visual elements:
- 1D signal as continuous line
- Spectrum as vertical bars
- 2D matrix as colored grid cells
${IEEE_STYLE_BASE}
`,
    negativePrompt: "photorealistic, 3D rendering, cartoon, hand-drawn, decorative elements, text"
  },

  "fig_3_7_anomaly_detection_framework": {
    title: "异常检测框架流程图",
    description: "End-to-End Anomaly Detection Framework",
    prompt: `
Create a professional flowchart showing an anomaly detection pipeline.

Flow structure (left to right):
BLOCK 1: Input data icon (waveform symbol)
BLOCK 2: Preprocessing module (filter icon)
BLOCK 3: Feature extraction (neural network icon with layers)
BLOCK 4: Reconstruction module (decoder icon)
BLOCK 5: Scoring module (comparison/difference icon)
BLOCK 6: Output (binary classification icon)

Directional arrows connecting all blocks in sequence
Each block as a rounded rectangle in IEEE blue
${IEEE_STYLE_BASE}
Layout: Horizontal pipeline flow
`,
    negativePrompt: "3D, photorealistic, cartoon, excessive colors, decorative borders, text, Chinese characters"
  },

  "fig_timesnet_architecture": {
    title: "TimesNet 网络架构图",
    description: "TimesNet Neural Network Architecture",
    prompt: `
Create a neural network architecture diagram for TimesNet.

Vertical structure:
INPUT: Rectangular input tensor block at top
TIMESBLOCK (shown as a detailed module, repeated):
- FFT sub-block
- Reshape operation arrow
- 2D Convolution blocks (multiple parallel paths like Inception)
- Merge/concatenation point
- Skip connection arrow curving around
OUTPUT: Rectangular output tensor block at bottom

Show stacking of multiple TimesBlocks
Residual connections as curved bypass arrows
${IEEE_STYLE_BASE}
Style: Clean neural network diagram
`,
    negativePrompt: "3D perspective, photorealistic, cartoon, hand-drawn, excessive decoration, text labels"
  },

  "fig_voltagetimesnet_architecture": {
    title: "VoltageTimesNet 网络架构图",
    description: "VoltageTimesNet Architecture with Enhancements",
    prompt: `
Create an architecture diagram comparing standard neural network block with enhanced version.

Two parallel vertical flows side by side:
LEFT SIDE: Standard TimesNet block (simpler, fewer components)
RIGHT SIDE: VoltageTimesNet block (same structure but with additional highlighted modules)

Enhancements highlighted in orange:
- Domain prior injection module
- Enhanced period weighting module

Both sides share similar structure:
- FFT module
- Reshape operation
- 2D convolution
- Aggregation
- Skip connection

Arrow at bottom merging to output
${IEEE_STYLE_BASE}
Highlight color: Orange for enhanced components
`,
    negativePrompt: "3D, photorealistic, cartoon, excessive shadows, decorative elements, text"
  },

  "fig_fft_period_discovery": {
    title: "FFT 周期发现示意图",
    description: "Period Discovery via FFT",
    prompt: `
Create a two-panel technical illustration for FFT analysis.

PANEL 1 (Top or Left): Time domain signal
- Continuous waveform with visible periodic patterns
- X-axis showing time progression
- Y-axis showing amplitude

PANEL 2 (Bottom or Right): Frequency domain spectrum
- Bar chart or line showing frequency magnitude
- Several dominant peaks highlighted in IEEE blue
- Other frequencies in gray
- Clear separation between signal and noise frequencies

Arrow connecting the two panels showing FFT transformation
${IEEE_STYLE_BASE}
`,
    negativePrompt: "3D, photorealistic, cartoon, hand-drawn style, decorative, text labels"
  },

  "fig_2d_conv_inception": {
    title: "2D卷积 Inception 模块示意图",
    description: "2D Inception Convolution Block",
    prompt: `
Create a technical diagram showing Inception-style convolution module.

Structure:
INPUT: Single rectangular block at top

PARALLEL BRANCHES (4 paths side by side):
- Branch 1: Small square block (1x1 conv)
- Branch 2: Small square → Medium square (1x1 then 3x3 conv)
- Branch 3: Small square → Large square (1x1 then 5x5 conv)
- Branch 4: Grid pattern → Small square (pooling then 1x1 conv)

CONCATENATION: All branches merge into single wide block

OUTPUT: Single rectangular block at bottom

Arrows showing data flow from input through branches to output
${IEEE_STYLE_BASE}
Layout: Vertical flow with parallel horizontal branches
`,
    negativePrompt: "3D rendering, photorealistic, cartoon, hand-drawn, excessive colors, text"
  }
};

// 图像生成函数
async function generateFigure(figureKey: string): Promise<string | null> {
  const figurePrompt = FIGURE_PROMPTS[figureKey];
  if (!figurePrompt) {
    console.error(`未知的图表键: ${figureKey}`);
    return null;
  }

  console.log(`\n正在生成: ${figurePrompt.title}`);
  console.log(`描述: ${figurePrompt.description}`);

  try {
    // 使用支持图像生成的模型
    const result = client.callModel({
      model: 'google/gemini-2.0-flash-exp:free',
      input: [
        {
          role: 'user',
          content: `Generate a professional technical diagram for an IEEE academic paper.

${figurePrompt.prompt}

IMPORTANT REQUIREMENTS:
1. NO text, labels, or annotations on the image
2. Pure visual diagram only
3. IEEE academic style - clean, professional, technical
4. White background
5. Use IEEE blue (#0076A8) as primary color
6. Vector-like clean appearance

${figurePrompt.negativePrompt ? `AVOID: ${figurePrompt.negativePrompt}` : ''}`
        }
      ]
    });

    const response = await result.getResponse();
    console.log(`✓ 生成完成: ${figureKey}`);

    // 返回响应文本（包含图像描述或URL）
    return response.text || null;
  } catch (error) {
    console.error(`✗ 生成失败 ${figureKey}:`, error);
    return null;
  }
}

// 批量生成所有图表
async function generateAllFigures(): Promise<void> {
  console.log('='.repeat(60));
  console.log('IEEE 学术风格图表批量生成');
  console.log('='.repeat(60));

  const outputDir = path.join(__dirname, 'output');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const results: Record<string, string | null> = {};

  for (const figureKey of Object.keys(FIGURE_PROMPTS)) {
    const result = await generateFigure(figureKey);
    results[figureKey] = result;

    // 保存结果到文件
    if (result) {
      const outputPath = path.join(outputDir, `${figureKey}_response.txt`);
      fs.writeFileSync(outputPath, result);
      console.log(`  → 已保存响应到: ${outputPath}`);
    }
  }

  // 生成汇总报告
  const reportPath = path.join(outputDir, 'generation_report.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    totalFigures: Object.keys(FIGURE_PROMPTS).length,
    results: Object.keys(results).map(key => ({
      key,
      title: FIGURE_PROMPTS[key].title,
      success: results[key] !== null
    }))
  }, null, 2));

  console.log('\n' + '='.repeat(60));
  console.log(`生成完成！报告已保存到: ${reportPath}`);
}

// 列出所有可用提示词
function listPrompts(): void {
  console.log('\n可用的学术图表提示词：');
  console.log('='.repeat(60));
  for (const [key, prompt] of Object.entries(FIGURE_PROMPTS)) {
    console.log(`  ${key}`);
    console.log(`    → ${prompt.title}`);
    console.log(`    → ${prompt.description}`);
  }
  console.log('='.repeat(60));
  console.log(`总计: ${Object.keys(FIGURE_PROMPTS).length} 个提示词\n`);
}

// 主函数
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help') {
    console.log(`
IEEE 学术风格图表生成器

用法:
  npx ts-node generate_figures.ts --list          列出所有可用提示词
  npx ts-node generate_figures.ts --all           生成所有图表
  npx ts-node generate_figures.ts <figure_key>    生成指定图表

示例:
  npx ts-node generate_figures.ts fig_timesnet_architecture
`);
    return;
  }

  if (args[0] === '--list') {
    listPrompts();
    return;
  }

  if (args[0] === '--all') {
    await generateAllFigures();
    return;
  }

  // 生成指定图表
  const figureKey = args[0];
  if (!(figureKey in FIGURE_PROMPTS)) {
    console.error(`未知的图表键: ${figureKey}`);
    console.log('使用 --list 查看所有可用的提示词');
    return;
  }

  const result = await generateFigure(figureKey);
  if (result) {
    console.log('\n生成结果:');
    console.log(result);
  }
}

main().catch(console.error);
