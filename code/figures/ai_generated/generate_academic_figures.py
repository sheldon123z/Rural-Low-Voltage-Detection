#!/usr/bin/env python3
"""
IEEE Smart Grid 学术风格图表生成器

使用 OpenRouter API 调用图像生成模型为论文创建学术级结构图。
支持的模型包括 Gemini、DALL-E 等具有图像生成能力的模型。
"""

import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY 环境变量")

# OpenRouter API 配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# IEEE 学术风格基础提示词
IEEE_STYLE_BASE = """
CRITICAL STYLE REQUIREMENTS for IEEE academic publication:
1. Visual Style: Clean vector-like appearance, sharp edges, professional technical diagram
2. Color Palette:
   - Primary: IEEE blue (#0076A8)
   - Secondary: Dark gray (#333333)
   - Background: Pure white (#FFFFFF)
   - Accent: Light gray (#E5E5E5)
3. Design: Minimal shadows, flat design, no decorative elements
4. Typography style: Sans-serif (but NO actual text on image)
5. High contrast suitable for print reproduction
6. IMPORTANT: Generate ONLY visual elements, NO text labels or annotations
"""

# 图表提示词库
PROMPTS: Dict[str, Dict[str, Any]] = {
    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "description": "三层数据采集架构：现场层、通信层、平台层",
        "prompt": f"""
Create a professional 3-layer architecture diagram for rural power distribution monitoring.

VISUAL STRUCTURE (top to bottom):
Layer 1 (Bottom - Field Layer):
- Row of 5-6 simple house icons representing rural households
- Connected by horizontal power line
- Small meter symbols at each house
- Distribution transformer symbol in center

Layer 2 (Middle - Communication Layer):
- 2-3 data concentrator boxes
- Wireless antenna symbols
- Vertical lines connecting to Layer 1

Layer 3 (Top - Platform Layer):
- Central server/cloud icon
- Database cylinder icon
- Processing module rectangle
- All connected horizontally

CONNECTIONS:
- Vertical arrows between layers showing data flow upward
- Use IEEE blue for data flow arrows
- Gray for infrastructure lines

{IEEE_STYLE_BASE}
Output: Clean technical diagram, white background, no text
""",
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "description": "四种典型电压异常波形：骤降、骤升、闪变、中断",
        "prompt": f"""
Create a 2x2 grid showing 4 voltage anomaly waveforms.

GRID LAYOUT:
Top-Left: VOLTAGE SAG
- Sine wave with sudden dip (amplitude drops to ~70%)
- Dip region lightly shaded in orange

Top-Right: VOLTAGE SWELL
- Sine wave with elevated section (amplitude rises to ~120%)
- Swell region lightly shaded in orange

Bottom-Left: VOLTAGE FLICKER
- Sine wave with oscillating envelope
- Amplitude varies periodically

Bottom-Right: VOLTAGE INTERRUPTION
- Sine wave with gap (zero voltage section)
- Interruption region clearly visible

EACH PANEL:
- Clean X-Y axes (no labels needed)
- Horizontal dashed line for nominal voltage reference
- IEEE blue for normal waveform
- Orange tint for anomaly regions
- Light gray grid lines

{IEEE_STYLE_BASE}
Output: 2x2 grid, white background, professional waveform plots
""",
    },

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "description": "时间序列滑动窗口机制可视化",
        "prompt": f"""
Create a diagram showing sliding window mechanism on time series.

VISUAL ELEMENTS:
1. Long horizontal waveform (continuous voltage-like signal)
   - Spanning full width
   - Blue line with some variation/oscillation

2. Three window frames at different positions:
   - Rectangular boxes with IEEE blue border
   - Semi-transparent blue fill
   - Positioned: left (past), center (current), right (future)
   - Equal spacing between windows

3. Directional arrow:
   - Large horizontal arrow below windows
   - Points left to right (sliding direction)

4. Input-Output indication:
   - Bracket under center window (input)
   - Small arrow pointing to single dot (output prediction)

{IEEE_STYLE_BASE}
Layout: Horizontal, clean, minimal
""",
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "1D到2D时序转换示意图",
        "description": "时间序列从一维到二维的变换过程",
        "prompt": f"""
Create a 3-stage transformation diagram (left to right).

STAGE 1 (Left): 1D TIME SERIES
- Horizontal waveform/signal
- Continuous line with oscillations
- IEEE blue color

STAGE 2 (Middle): FREQUENCY SPECTRUM
- Vertical bar chart
- 5-6 bars of varying heights
- Tallest bars (dominant frequencies) in IEEE blue
- Shorter bars in gray

STAGE 3 (Right): 2D MATRIX
- Grid/matrix representation (8x8 or similar)
- Cells colored as heatmap (blue to white gradient)
- Shows periodic pattern after reshaping

CONNECTIONS:
- Arrow from Stage 1 to Stage 2 (FFT transform)
- Arrow from Stage 2 to Stage 3 (Reshape)

{IEEE_STYLE_BASE}
Layout: Horizontal flow, 3 distinct sections
""",
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "description": "端到端异常检测流程",
        "prompt": f"""
Create a horizontal flowchart for anomaly detection pipeline.

BLOCKS (left to right, 6 rounded rectangles):
1. Input: Waveform icon inside box
2. Preprocess: Filter/cleaning icon
3. Feature Extract: Neural network layers icon (3 stacked rectangles)
4. Reconstruct: Decoder shape (expanding layers)
5. Score: Comparison/subtraction symbol
6. Output: Checkmark/alert icon

CONNECTIONS:
- Horizontal arrows connecting each block
- All arrows in IEEE blue
- Blocks have light gray fill, blue border

LAYOUT:
- Single horizontal row
- Equal spacing between blocks
- Clean, minimal design

{IEEE_STYLE_BASE}
""",
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet 网络架构图",
        "description": "TimesNet 神经网络结构",
        "prompt": f"""
Create a neural network architecture diagram (vertical flow).

STRUCTURE (top to bottom):
INPUT BLOCK:
- Rectangle at top

TIMESBLOCK MODULE (main component, shown in detail):
- FFT sub-block (small rectangle)
- Down arrow
- Reshape icon (1D to 2D transformation symbol)
- Down arrow
- 2D CONV block with 4 parallel paths (Inception style):
  * 4 vertical paths side by side
  * Different sizes representing different kernel sizes
  * Merge at bottom
- Skip connection: curved arrow from input to output of block
- LayerNorm block

REPETITION INDICATION:
- Dotted line suggesting "×N layers"

OUTPUT BLOCK:
- Rectangle at bottom

{IEEE_STYLE_BASE}
Color: IEEE blue for main flow, gray for skip connections
""",
    },

    "fig_voltagetimesnet_architecture": {
        "title": "VoltageTimesNet 网络架构图",
        "description": "VoltageTimesNet 架构（带领域增强）",
        "prompt": f"""
Create a side-by-side architecture comparison diagram.

LEFT SIDE: STANDARD BLOCK
- Simple vertical flow
- FFT → Reshape → Conv → Output
- All in IEEE blue/gray

RIGHT SIDE: ENHANCED BLOCK
- Same structure as left
- PLUS two highlighted modules in ORANGE:
  * "Domain Prior" module after FFT
  * "Enhanced Weighting" module before output
- Orange boxes stand out from blue/gray

VISUAL COMPARISON:
- Vertical dashed line separating left and right
- Right side slightly larger/emphasized
- Arrows showing enhancement locations

{IEEE_STYLE_BASE}
Highlight: Orange (#E07020) for enhancements
""",
    },

    "fig_fft_period_discovery": {
        "title": "FFT 周期发现示意图",
        "description": "快速傅里叶变换周期检测",
        "prompt": f"""
Create a two-panel FFT visualization (top and bottom).

TOP PANEL: TIME DOMAIN
- Continuous waveform with visible periodic pattern
- X-axis represents time
- Y-axis represents amplitude
- IEEE blue line

BOTTOM PANEL: FREQUENCY DOMAIN
- Bar chart / spectrum plot
- X-axis represents frequency
- Y-axis represents magnitude
- 2-3 prominent peaks in IEEE blue (dominant frequencies)
- Other frequencies in light gray
- Clear peak-to-background contrast

CONNECTION:
- Large arrow between panels
- Arrow points downward (time → frequency)

{IEEE_STYLE_BASE}
Layout: Vertical stack, two panels
""",
    },

    "fig_2d_conv_inception": {
        "title": "2D卷积 Inception 模块示意图",
        "description": "Inception 风格的多尺度卷积模块",
        "prompt": f"""
Create an Inception module diagram.

STRUCTURE (top to bottom):
INPUT:
- Single rectangle at top

FOUR PARALLEL BRANCHES:
Branch 1: Single small square (1x1 conv)
Branch 2: Small square → Medium square (1x1 → 3x3)
Branch 3: Small square → Large square (1x1 → 5x5)
Branch 4: Grid pattern → Small square (pooling → 1x1)

All branches start from input and run vertically downward

CONCATENATION:
- All 4 branches merge into single wide rectangle

OUTPUT:
- Single rectangle at bottom

VISUAL STYLE:
- Different sizes represent different kernel sizes
- Light blue fill for all conv blocks
- Arrows showing data flow

{IEEE_STYLE_BASE}
Layout: Vertical with parallel horizontal branches
""",
    },
}


def call_openrouter_chat(prompt: str, model: str = "google/gemini-flash-1.5") -> Optional[str]:
    """
    调用 OpenRouter Chat API
    注意：这主要用于获取文本描述，因为大多数模型不直接返回图像
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rural-voltage-detection",
        "X-Title": "Rural Voltage Detection - Academic Figure Generator"
    }

    # 尝试多个模型（2026年2月更新的可用模型列表）
    models_to_try = [
        "openrouter/free",  # 免费模型路由
        "google/gemini-3-flash-preview",
        "stepfun/step-3.5-flash:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-120b:free",
    ]

    for try_model in models_to_try:
        data = {
            "model": try_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }

        try:
            print(f"  尝试模型: {try_model}")
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                print(f"  ✓ 成功使用模型: {try_model}")
                return content
        except requests.exceptions.HTTPError as e:
            print(f"  模型 {try_model} 失败: {e}")
            continue
        except Exception as e:
            print(f"  模型 {try_model} 错误: {e}")
            continue

    print("所有模型均失败")
    return None


def generate_figure_description(figure_key: str) -> Optional[Dict[str, Any]]:
    """
    生成图表的详细描述（用于手动创建或进一步处理）
    """
    if figure_key not in PROMPTS:
        print(f"未知的图表键: {figure_key}")
        return None

    prompt_config = PROMPTS[figure_key]
    print(f"\n{'='*60}")
    print(f"正在处理: {prompt_config['title']}")
    print(f"描述: {prompt_config['description']}")
    print(f"{'='*60}")

    # 构建完整提示词
    full_prompt = f"""
You are a technical diagram expert specializing in IEEE academic publications.

Task: Provide a detailed textual description of a technical diagram that could be created using professional drawing tools.

FIGURE: {prompt_config['title']}
CONTEXT: {prompt_config['description']}

REQUIREMENTS:
{prompt_config['prompt']}

Please provide:
1. Exact layout dimensions and proportions
2. Specific color codes (hex values)
3. Shape specifications (rounded corners radius, line thickness)
4. Precise positioning of elements
5. Connection arrow styles
6. A step-by-step guide for recreating this diagram in draw.io or similar tools

Output should be detailed enough for someone to recreate the exact diagram.
"""

    response = call_openrouter_chat(full_prompt)

    return {
        "key": figure_key,
        "title": prompt_config["title"],
        "description": prompt_config["description"],
        "prompt": prompt_config["prompt"],
        "generated_description": response,
        "timestamp": datetime.now().isoformat()
    }


def generate_all_descriptions(output_dir: Optional[Path] = None) -> None:
    """
    为所有图表生成详细描述
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for figure_key in PROMPTS:
        result = generate_figure_description(figure_key)
        if result:
            results.append(result)

            # 保存单个结果
            output_file = output_dir / f"{figure_key}_description.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ 已保存: {output_file}")

    # 保存汇总报告
    report_file = output_dir / "generation_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_figures": len(PROMPTS),
        "generated": len(results),
        "figures": [{"key": r["key"], "title": r["title"]} for r in results]
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"生成完成！共处理 {len(results)}/{len(PROMPTS)} 个图表")
    print(f"报告已保存到: {report_file}")


def list_available_prompts() -> None:
    """
    列出所有可用的图表提示词
    """
    print("\n可用的学术图表提示词：")
    print("=" * 60)
    for i, (key, config) in enumerate(PROMPTS.items(), 1):
        print(f"{i}. {key}")
        print(f"   标题: {config['title']}")
        print(f"   描述: {config['description']}")
        print()
    print("=" * 60)
    print(f"总计: {len(PROMPTS)} 个提示词\n")


def export_prompts_for_nanobanan(output_file: Optional[Path] = None) -> None:
    """
    导出提示词为 Nano Banana Pro 友好的格式
    """
    if output_file is None:
        output_file = Path(__file__).parent / "nanobanan_prompts.json"

    export_data = []
    for key, config in PROMPTS.items():
        export_data.append({
            "name": key,
            "title": config["title"],
            "description": config["description"],
            "prompt": config["prompt"].strip(),
            "style": "IEEE academic, power systems, technical diagram",
            "negative_prompt": "cartoon, 3D realistic, photorealistic, text labels, Chinese characters, decorative elements"
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"提示词已导出到: {output_file}")
    print(f"可直接用于 Nano Banana Pro 或其他图像生成工具")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IEEE 学术风格图表生成器")
    parser.add_argument("--list", action="store_true", help="列出所有可用提示词")
    parser.add_argument("--all", action="store_true", help="生成所有图表描述")
    parser.add_argument("--export", action="store_true", help="导出提示词为 JSON 格式")
    parser.add_argument("--figure", type=str, help="生成指定图表的描述")
    parser.add_argument("--output", type=str, help="输出目录")

    args = parser.parse_args()

    if args.list:
        list_available_prompts()
    elif args.export:
        export_prompts_for_nanobanan()
    elif args.all:
        output_dir = Path(args.output) if args.output else None
        generate_all_descriptions(output_dir)
    elif args.figure:
        result = generate_figure_description(args.figure)
        if result:
            print("\n生成的描述：")
            print(result.get("generated_description", "无结果"))
    else:
        parser.print_help()
        print("\n示例:")
        print("  python generate_academic_figures.py --list")
        print("  python generate_academic_figures.py --export")
        print("  python generate_academic_figures.py --figure fig_timesnet_architecture")
        print("  python generate_academic_figures.py --all")
