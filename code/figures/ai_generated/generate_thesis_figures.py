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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY 环境变量")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Clash 代理配置
PROXY_CONFIG = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# 图像生成模型优先级（gemini-2.5-flash-image 可用）
IMAGE_MODELS = [
    "google/gemini-2.5-flash-image",  # 可用，返回 base64 图像
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
# 论文图表提示词（简化英文版，适合 Gemini 图像生成）
# ============================================================
THESIS_FIGURES: Dict[str, Dict[str, Any]] = {

    # --------------------------------------------------------
    # 第2章：数据采集与预处理
    # --------------------------------------------------------

    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "chapter": 2,
        "description": "农村低压配电网数据采集系统三层架构",
        "prompt": "Technical diagram: Three-layer IoT architecture. Top layer: cloud server with database icon. Middle layer: 3 wireless gateways connected horizontally. Bottom layer: 6 houses with smart meters, power lines, transformer in center. Vertical arrows between layers showing data flow upward. Clean flat design, blue (#0076A8) color scheme, white background, no text labels, vector style, professional academic diagram.",
        "size": "1800x1400"
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "chapter": 2,
        "description": "四种典型电压异常波形对比",
        "prompt": "Technical diagram: 2x2 grid showing 4 voltage waveform anomalies. Top-left: sine wave with voltage sag (70% dip). Top-right: sine wave with voltage swell (120% rise). Bottom-left: sine wave with amplitude modulation (flicker). Bottom-right: sine wave with interruption gap. Each panel has X-Y axes, blue waveform, orange highlight on anomaly region. Clean flat design, white background, no text, academic style.",
        "size": "1800x1400"
    },

    # --------------------------------------------------------
    # 第3章：模型方法
    # --------------------------------------------------------

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "chapter": 3,
        "description": "时间序列滑动窗口机制可视化",
        "prompt": "Technical diagram: Sliding window mechanism for time series. Horizontal wavy signal line spanning full width. Three blue rectangular window frames at left, center, right positions with semi-transparent fill. Horizontal arrow below showing sliding direction. Bracket under center window with arrow pointing to output dot. Clean flat design, blue (#0076A8), white background, no text, academic style.",
        "size": "1800x1000"
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "一维到二维时序转换示意图",
        "chapter": 3,
        "description": "时间序列从一维到二维的变换过程",
        "prompt": "Technical diagram: 1D to 2D transformation in 3 stages. Left: horizontal wavy line (1D signal). Middle: vertical bar chart (frequency spectrum, 5-6 bars). Right: 8x8 grid matrix with blue gradient heatmap. Large arrows connecting stages. Clean flat design, blue (#0076A8), white background, horizontal layout, no text, academic style.",
        "size": "1800x1000"
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "chapter": 3,
        "description": "端到端异常检测系统流程",
        "prompt": "Technical flowchart: Anomaly detection pipeline with 6 connected rounded rectangles. Left to right: waveform icon, filter icon, neural network layers icon, decoder icon, comparison symbol, alert icon. Horizontal arrows between blocks. Blue (#0076A8) borders, white/light blue fill, white background. Clean flat design, no text, professional academic style.",
        "size": "2000x800"
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet网络架构图",
        "chapter": 3,
        "description": "TimesNet神经网络结构详图",
        "prompt": "Technical diagram: Vertical neural network architecture. Top: input rectangle. Main module with dashed border containing: FFT block, reshape icon (1D to 2D), four parallel paths (Inception style with different sizes), merge point, normalization block. Curved skip connection arrow on side. Output rectangle at bottom. Blue (#0076A8) color scheme, white background, no text, academic style.",
        "size": "1400x2000"
    },

    "fig_voltagetimesnet_architecture": {
        "title": "VoltageTimesNet网络架构图",
        "chapter": 3,
        "description": "VoltageTimesNet架构与增强模块",
        "prompt": "Technical diagram: Side-by-side comparison of two architectures. Left column: TimesNet with input, FFT, 2D conv, output blocks in blue. Right column: VoltageTimesNet with same blocks plus two orange highlighted enhancement modules. Skip connections on both sides. Vertical dashed line separating columns. Blue (#0076A8) for base, orange for enhancements, white background, no text.",
        "size": "1800x1600"
    },

    "fig_fft_period_discovery": {
        "title": "快速傅里叶变换周期发现示意图",
        "chapter": 3,
        "description": "FFT周期检测原理图",
        "prompt": "Technical diagram: Two-panel FFT analysis. Top panel: continuous wavy signal showing periodic pattern with X-Y axes. Bottom panel: bar chart showing frequency spectrum with 2-3 tall blue bars (dominant frequencies) and shorter gray bars. Large downward arrow between panels labeled FFT. Blue (#0076A8) waveform and bars, white background, no text labels, academic style.",
        "size": "1600x1400"
    },

    "fig_2d_conv_inception": {
        "title": "二维卷积Inception模块示意图",
        "chapter": 3,
        "description": "Inception风格的多尺度卷积结构",
        "prompt": "Technical diagram: Inception module structure. Top: input rectangle. Middle: 4 parallel vertical paths - path 1: small square; path 2: small then medium square; path 3: small then large square; path 4: grid pattern then small square. All paths merge into wide rectangle. Bottom: output rectangle. Arrows showing data flow, blue (#0076A8) fill, white background, no text, academic style.",
        "size": "1600x1800"
    },

    "fig_3_3_voltage_timesnet_comparison": {
        "title": "VoltageTimesNet与TimesNet周期检测对比",
        "chapter": 3,
        "description": "两种模型周期检测机制的对比",
        "prompt": "Technical diagram: 2x2 comparison grid. Top row shows frequency spectrums: left has FFT peaks only (blue/gray), right has FFT peaks plus preset period markers (blue/orange). Bottom row shows weight bars: left has equal height bars, right has varying height bars with attention weights. Vertical dashed line in center. Blue (#0076A8) and orange accents, white background, no text, academic style.",
        "size": "1800x1400"
    },
}


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


def generate_figure(figure_key: str, output_dir: Path, models: List[str] = None) -> Dict[str, Any]:
    """
    生成单个图表
    """
    if figure_key not in THESIS_FIGURES:
        print(f"未知的图表键: {figure_key}")
        return {"success": False, "error": "Unknown figure key"}

    config = THESIS_FIGURES[figure_key]
    models = models or IMAGE_MODELS

    print(f"\n{'='*60}")
    print(f"正在生成: {config['title']}")
    print(f"章节: 第{config['chapter']}章")
    print(f"描述: {config['description']}")
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

    for model in models:
        print(f"  尝试模型: {model}")

        response = call_image_generation_api(
            config["prompt"],
            model,
            config.get("size", "1800x1200")
        )

        if response:
            # 确定输出文件
            if isinstance(response, bytes) and len(response) > 1000:
                # 可能是图像数据
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
                # 文本响应
                text_path = output_dir / f"{figure_key}_response.txt"
                if isinstance(response, bytes):
                    response = response.decode("utf-8", errors="ignore")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(f"Model: {model}\n\n{response}")
                print(f"  ℹ 模型返回文本响应: {text_path}")
                result["model_used"] = model
                result["output_path"] = str(text_path)
                result["response_type"] = "text"
                # 继续尝试其他模型

        time.sleep(3)  # 避免速率限制

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
    print(f"图像模型: {IMAGE_MODELS[0]}")
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
