#!/usr/bin/env python3
"""
使用 HuggingFace 推理 API 生成论文图表

HuggingFace 的 Flux 模型可以免费使用，没有地区限制。
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import time

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Clash 代理配置
PROXY_CONFIG = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# HuggingFace 推理 API（免费，无需 API 密钥也可使用部分模型）
HF_API_URL = "https://api-inference.huggingface.co/models"

# 可用的免费图像生成模型
HF_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",  # 快速版 Flux
    "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL
    "runwayml/stable-diffusion-v1-5",  # SD 1.5
]

# 论文图表提示词（英文版，适合 Flux 模型）
THESIS_FIGURES_EN: Dict[str, Dict[str, Any]] = {

    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "chapter": 2,
        "prompt": """Professional technical diagram showing a three-layer IoT architecture for smart grid monitoring.

Top layer: Cloud server icon with database symbol, labeled "Data Center"
Middle layer: Three wireless gateway devices connected horizontally
Bottom layer: Six house icons with smart meter symbols, connected by power lines with transformer in center

Style: Clean vector illustration, flat design, IEEE academic style
Colors: Primary blue (#0076A8), dark gray (#333333), white background
No text labels, pure visual diagram, high contrast, print quality
Minimal, professional, suitable for academic publication""",
        "size": "1024x768"
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "chapter": 2,
        "prompt": """Technical diagram showing four types of voltage anomalies in 2x2 grid layout.

Top-left: Sine wave with voltage sag (amplitude drops to 70%)
Top-right: Sine wave with voltage swell (amplitude rises to 120%)
Bottom-left: Sine wave with amplitude modulation (flicker effect)
Bottom-right: Sine wave with zero-voltage interruption gap

Each panel has coordinate axes, blue waveform, orange highlight for anomaly region
Style: Clean technical illustration, flat design, IEEE academic style
Colors: Blue (#0076A8) for normal, orange highlight for anomalies, white background
No text labels, pure visual diagram, high contrast""",
        "size": "1024x768"
    },

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "chapter": 3,
        "prompt": """Technical diagram showing sliding window mechanism for time series processing.

Elements:
- Horizontal wavy signal line spanning full width (voltage time series)
- Three rectangular window frames at left, center, right positions
- Windows have blue border with semi-transparent fill
- Horizontal arrow below showing sliding direction (left to right)
- Bracket under center window indicating input, arrow pointing to output dot

Style: Clean vector illustration, flat design, IEEE academic style
Colors: Blue (#0076A8), white background, gray for arrows
No text labels, minimal, professional diagram""",
        "size": "1024x512"
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "一维到二维时序转换示意图",
        "chapter": 3,
        "prompt": """Technical diagram showing 1D to 2D transformation process in three stages.

Left: Horizontal wavy line (1D time series signal)
Middle: Vertical bar chart (frequency spectrum, 5-6 bars, tallest in blue)
Right: 8x8 grid matrix with blue gradient heatmap pattern

Arrows connecting stages showing transformation flow
Style: Clean vector illustration, flat design, IEEE academic style
Colors: Blue (#0076A8), gray, white background
Horizontal layout, no text labels, professional diagram""",
        "size": "1024x512"
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "chapter": 3,
        "prompt": """Horizontal flowchart showing anomaly detection pipeline with 6 connected blocks.

Blocks left to right:
1. Waveform icon (input data)
2. Filter icon (preprocessing)
3. Stacked layers icon (feature extraction neural network)
4. Expanding layers icon (reconstruction decoder)
5. Comparison symbol (scoring)
6. Alert icon (detection output)

Rounded rectangle blocks connected by horizontal arrows
Style: Clean vector illustration, flat design, IEEE academic style
Colors: Blue (#0076A8) borders, white/light blue fill, white background
No text labels, professional flowchart""",
        "size": "1280x512"
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet网络架构图",
        "chapter": 3,
        "prompt": """Vertical neural network architecture diagram for TimesNet model.

Structure top to bottom:
- Input rectangle block
- Main module with dashed border containing:
  - Small FFT block
  - Reshape icon (1D to 2D)
  - Four parallel paths (Inception style, different sizes)
  - Merge point
  - Normalization block
- Curved skip connection arrow on the side
- Dotted line indicating layer repetition
- Output rectangle block

Style: Clean neural network diagram, flat design, IEEE academic style
Colors: Blue (#0076A8) for main flow, gray for skip connections, white background
Vertical layout, no text labels""",
        "size": "768x1024"
    },

    "fig_fft_period_discovery": {
        "title": "快速傅里叶变换周期发现示意图",
        "chapter": 3,
        "prompt": """Two-panel technical diagram showing FFT analysis.

Top panel: Continuous wavy signal line showing periodic pattern (time domain)
With coordinate axes, blue waveform

Bottom panel: Bar chart showing frequency spectrum
2-3 tall prominent bars in blue (dominant frequencies)
Other bars shorter in gray

Large downward arrow between panels (FFT transformation)

Style: Clean technical illustration, flat design, IEEE academic style
Colors: Blue (#0076A8), gray, white background
Vertical stack layout, no text labels""",
        "size": "768x1024"
    },

    "fig_2d_conv_inception": {
        "title": "二维卷积Inception模块示意图",
        "chapter": 3,
        "prompt": """Technical diagram showing Inception convolution module structure.

Top: Single input rectangle
Middle: Four parallel vertical paths side by side:
- Path 1: Small square (1x1 conv)
- Path 2: Small -> medium square (1x1 -> 3x3 conv)
- Path 3: Small -> large square (1x1 -> 5x5 conv)
- Path 4: Grid pattern -> small square (pooling -> 1x1 conv)
All paths merge into wide rectangle
Bottom: Single output rectangle

Arrows showing data flow, all blocks in blue fill
Style: Clean neural network diagram, flat design, IEEE academic style
Colors: Blue (#0076A8), white background
Vertical layout, no text labels""",
        "size": "768x1024"
    },
}


def generate_image_hf(prompt: str, model: str, output_path: Path, use_proxy: bool = True) -> bool:
    """
    使用 HuggingFace 推理 API 生成图像
    """
    api_url = f"{HF_API_URL}/{model}"

    headers = {
        "Content-Type": "application/json",
    }

    # 如果有 HuggingFace token，添加到 headers
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 4 if "schnell" in model else 25,
            "guidance_scale": 0.0 if "schnell" in model else 7.5,
        }
    }

    proxies = PROXY_CONFIG if use_proxy else None

    try:
        print(f"    发送请求到 {model}...")
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=300,
            proxies=proxies
        )

        if response.status_code == 200:
            # 检查是否是图像数据
            content_type = response.headers.get("content-type", "")
            if "image" in content_type:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"    ✓ 图像已保存: {output_path}")
                return True
            else:
                print(f"    ✗ 非图像响应: {content_type}")
                print(f"    响应内容: {response.text[:200]}")
                return False

        elif response.status_code == 503:
            # 模型正在加载
            result = response.json()
            estimated_time = result.get("estimated_time", 60)
            print(f"    ⏳ 模型正在加载，预计等待 {estimated_time:.0f} 秒...")
            time.sleep(min(estimated_time, 60))
            # 重试
            return generate_image_hf(prompt, model, output_path, use_proxy)

        elif response.status_code == 429:
            print(f"    ⏳ 速率限制，等待 60 秒...")
            time.sleep(60)
            return False

        else:
            print(f"    ✗ HTTP {response.status_code}")
            try:
                error_info = response.json()
                print(f"    错误信息: {error_info.get('error', response.text[:200])}")
            except:
                print(f"    响应: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"    ✗ 请求超时")
        return False
    except Exception as e:
        print(f"    ✗ 请求错误: {e}")
        return False


def generate_figure(figure_key: str, output_dir: Path) -> Dict[str, Any]:
    """
    生成单个图表
    """
    if figure_key not in THESIS_FIGURES_EN:
        print(f"未知的图表键: {figure_key}")
        return {"success": False, "error": "Unknown figure key"}

    config = THESIS_FIGURES_EN[figure_key]

    print(f"\n{'='*60}")
    print(f"正在生成: {config['title']}")
    print(f"章节: 第{config['chapter']}章")
    print(f"{'='*60}")

    result = {
        "key": figure_key,
        "title": config["title"],
        "success": False,
        "model_used": None,
        "output_path": None,
        "timestamp": datetime.now().isoformat()
    }

    for model in HF_IMAGE_MODELS:
        print(f"  尝试模型: {model}")
        output_path = output_dir / f"{figure_key}.png"

        if generate_image_hf(config["prompt"], model, output_path):
            result["success"] = True
            result["model_used"] = model
            result["output_path"] = str(output_path)
            return result

        time.sleep(5)

    print(f"  ✗ 所有模型均失败")
    result["error"] = "All models failed"
    return result


def generate_all_figures(output_dir: Optional[Path] = None) -> None:
    """
    生成所有图表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "hf_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("使用 HuggingFace 推理 API 生成论文图表")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"使用代理: {PROXY_CONFIG['http']}")
    print("=" * 70)

    results = []
    for i, figure_key in enumerate(THESIS_FIGURES_EN.keys(), 1):
        print(f"\n[{i}/{len(THESIS_FIGURES_EN)}]", end="")
        result = generate_figure(figure_key, output_dir)
        results.append(result)
        time.sleep(10)  # 避免速率限制

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(THESIS_FIGURES_EN),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }

    report_path = output_dir / "generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"生成完成: {report['successful']}/{report['total']} 成功")
    print(f"报告: {report_path}")


def list_figures() -> None:
    """列出所有图表"""
    print("\n可用的图表:")
    print("=" * 60)
    for key, config in THESIS_FIGURES_EN.items():
        print(f"  {key}")
        print(f"    标题: {config['title']}")
        print(f"    章节: 第{config['chapter']}章")
    print("=" * 60)
    print(f"总计: {len(THESIS_FIGURES_EN)} 个")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用 HuggingFace 生成论文图表")
    parser.add_argument("--list", action="store_true", help="列出所有图表")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--figure", type=str, help="生成单个图表")
    parser.add_argument("--output", type=str, help="输出目录")

    args = parser.parse_args()

    if args.list:
        list_figures()
    elif args.figure:
        output_dir = Path(args.output) if args.output else Path(__file__).parent / "output" / "hf_figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_figure(args.figure, output_dir)
    elif args.all:
        output_dir = Path(args.output) if args.output else None
        generate_all_figures(output_dir)
    else:
        parser.print_help()
