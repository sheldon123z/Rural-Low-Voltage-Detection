#!/usr/bin/env python3
"""
使用 Gemini 图像生成 API 创建学术论文图表

支持的模型：
- google/gemini-3-pro-image-preview (图像生成)
- google/gemini-2.5-flash-image (图像生成)
"""

import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import time

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY 环境变量")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 支持图像生成的模型
IMAGE_MODELS = [
    "google/gemini-3-pro-image-preview",
    "google/gemini-2.5-flash-image",
]

# IEEE 学术风格提示词
IEEE_STYLE = """
STYLE: IEEE academic technical diagram for power systems research.
REQUIREMENTS:
- Clean vector-like appearance with sharp edges
- Color scheme: IEEE blue (#0076A8), dark gray (#333333), white background (#FFFFFF)
- Minimal shadows, flat design, professional
- High contrast for print reproduction (300 DPI quality)
- NO text, labels, or annotations on the image
- Pure visual diagram only
"""

# 简化的图表提示词（针对图像生成优化）
IMAGE_PROMPTS = {
    "fig_2_1_data_collection": {
        "title": "数据采集分层架构",
        "prompt": f"""
Generate a professional 3-layer architecture diagram.

Visual structure (top to bottom):
- TOP LAYER: Cloud/server icon with database symbol
- MIDDLE LAYER: 3 wireless antenna/router icons connected horizontally
- BOTTOM LAYER: Row of 6 simple house icons with meter symbols

Connections: Vertical lines/arrows between layers showing data flow upward
Layout: Balanced, centered, symmetrical
{IEEE_STYLE}
"""
    },

    "fig_2_2_voltage_anomaly": {
        "title": "电压异常波形",
        "prompt": f"""
Generate a 2x2 grid showing 4 electrical waveforms.

Each quadrant shows a sine wave with different anomaly:
- Top-left: Normal sine wave with a DIP (sudden drop in amplitude)
- Top-right: Normal sine wave with a SWELL (sudden increase in amplitude)
- Bottom-left: Sine wave with FLICKERING envelope (varying amplitude)
- Bottom-right: Sine wave with INTERRUPTION (gap with zero voltage)

Each panel has X-Y axes, wave in blue, anomaly region highlighted in orange tint.
{IEEE_STYLE}
"""
    },

    "fig_3_1_sliding_window": {
        "title": "滑动窗口机制",
        "prompt": f"""
Generate a diagram showing sliding window on time series.

Elements:
- Horizontal wavy line (signal) across full width
- 3 rectangular frames at different positions (past, current, future)
- Frames have blue border with semi-transparent fill
- Arrow below showing left-to-right movement direction

{IEEE_STYLE}
"""
    },

    "fig_3_2_1d_to_2d": {
        "title": "1D到2D转换",
        "prompt": f"""
Generate a 3-stage transformation diagram (left to right).

Stage 1 (left): Horizontal wavy line (1D signal)
Stage 2 (middle): Vertical bar chart (frequency spectrum, 5-6 bars)
Stage 3 (right): 8x8 grid with color gradient (2D matrix representation)

Arrows connecting stages showing the transformation flow.
{IEEE_STYLE}
"""
    },

    "fig_3_7_detection_flow": {
        "title": "异常检测流程",
        "prompt": f"""
Generate a horizontal flowchart with 6 connected blocks.

Blocks (left to right):
1. Waveform icon (input)
2. Filter icon (preprocessing)
3. Stacked rectangles (feature extraction)
4. Expanding layers (reconstruction)
5. Comparison symbol (scoring)
6. Alert icon (output)

All blocks connected by horizontal arrows. Blue rounded rectangles.
{IEEE_STYLE}
"""
    },

    "fig_timesnet_arch": {
        "title": "TimesNet架构",
        "prompt": f"""
Generate a neural network architecture diagram (vertical).

Structure (top to bottom):
- Input rectangle at top
- Main block containing:
  - Small rectangle (FFT)
  - Arrow down
  - Grid symbol (2D reshape)
  - 4 parallel vertical paths of different widths (Inception conv)
  - Merge point
  - Curved arrow on side (skip connection)
- Dotted line indicating repetition
- Output rectangle at bottom

{IEEE_STYLE}
"""
    },

    "fig_fft_period": {
        "title": "FFT周期发现",
        "prompt": f"""
Generate a two-panel FFT visualization (top and bottom panels).

TOP PANEL:
- Continuous wavy line (time domain signal)
- Shows periodic pattern

BOTTOM PANEL:
- Bar chart (frequency spectrum)
- 2-3 tall bars (dominant frequencies) in blue
- Other bars shorter in gray

Large arrow between panels pointing downward.
{IEEE_STYLE}
"""
    },

    "fig_inception_block": {
        "title": "Inception卷积模块",
        "prompt": f"""
Generate an Inception module diagram.

Structure:
- Single rectangle at TOP (input)
- 4 PARALLEL vertical paths below:
  - Path 1: Small square
  - Path 2: Small square -> Medium square
  - Path 3: Small square -> Large square
  - Path 4: Grid pattern -> Small square
- All paths merge into WIDE rectangle
- Single rectangle at BOTTOM (output)

Arrows showing data flow. Blue fill for all blocks.
{IEEE_STYLE}
"""
    },
}


def generate_image_with_gemini(prompt: str, output_path: Path, model: str = None) -> bool:
    """
    使用 Gemini 图像生成 API 生成图像
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rural-voltage-detection",
        "X-Title": "Rural Voltage Detection - Academic Figure Generator"
    }

    models_to_try = [model] if model else IMAGE_MODELS

    for try_model in models_to_try:
        print(f"  尝试模型: {try_model}")

        data = {
            "model": try_model,
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
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                # 检查是否返回了图像
                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")

                # 某些模型可能在 content 中返回 base64 图像
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image" and "data" in item:
                            image_data = base64.b64decode(item["data"])
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            print(f"  ✓ 图像已保存: {output_path}")
                            return True

                # 如果没有直接返回图像，保存响应内容
                if content:
                    text_path = output_path.with_suffix(".txt")
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(f"Model: {try_model}\n\n{content}")
                    print(f"  ℹ 模型返回了文本描述: {text_path}")
                    return True

            elif response.status_code == 429:
                print(f"  ⏳ 速率限制，等待后重试...")
                time.sleep(10)
                continue
            else:
                print(f"  ✗ HTTP {response.status_code}: {response.text[:200]}")
                continue

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            continue

    return False


def generate_all_images(output_dir: Optional[Path] = None) -> None:
    """
    生成所有学术图表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IEEE 学术风格图表生成")
    print("=" * 60)

    results = []
    for key, config in IMAGE_PROMPTS.items():
        print(f"\n正在处理: {config['title']}")
        print("-" * 40)

        output_path = output_dir / f"{key}.png"
        success = generate_image_with_gemini(config["prompt"], output_path)

        results.append({
            "key": key,
            "title": config["title"],
            "success": success,
            "output": str(output_path) if success else None
        })

        # 避免触发速率限制
        time.sleep(2)

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(IMAGE_PROMPTS),
        "success": sum(1 for r in results if r["success"]),
        "results": results
    }
    report_path = output_dir / "generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"生成完成: {report['success']}/{report['total']} 成功")
    print(f"报告: {report_path}")


def export_prompts_for_external_tools(output_file: Optional[Path] = None) -> None:
    """
    导出提示词供外部工具使用（如 Midjourney, DALL-E, Nano Banana Pro 等）
    """
    if output_file is None:
        output_file = Path(__file__).parent / "external_prompts.json"

    export_data = []
    for key, config in IMAGE_PROMPTS.items():
        export_data.append({
            "id": key,
            "title": config["title"],
            "prompt": config["prompt"].strip(),
            "style": "IEEE academic, power systems, technical diagram, vector art",
            "negative_prompt": "text, labels, annotations, cartoon, 3D realistic, photorealistic, decorative, Chinese characters",
            "recommended_tools": ["Midjourney", "DALL-E 3", "Nano Banana Pro", "Ideogram"]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"提示词已导出到: {output_file}")
    print("\n可用于以下工具:")
    print("  - Midjourney (/imagine prompt)")
    print("  - DALL-E 3 (ChatGPT)")
    print("  - Nano Banana Pro")
    print("  - Ideogram.ai")
    print("  - Leonardo.ai")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="学术图表图像生成器")
    parser.add_argument("--generate", action="store_true", help="使用 Gemini 生成图像")
    parser.add_argument("--export", action="store_true", help="导出提示词供外部工具使用")
    parser.add_argument("--list", action="store_true", help="列出所有可用提示词")
    parser.add_argument("--output", type=str, help="输出目录")

    args = parser.parse_args()

    if args.list:
        print("\n可用的图表提示词:")
        print("=" * 60)
        for key, config in IMAGE_PROMPTS.items():
            print(f"  {key}: {config['title']}")
        print("=" * 60)
    elif args.export:
        export_prompts_for_external_tools()
    elif args.generate:
        output_dir = Path(args.output) if args.output else None
        generate_all_images(output_dir)
    else:
        parser.print_help()
