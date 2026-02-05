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

# 图像生成模型优先级
IMAGE_MODELS = [
    "google/gemini-2.5-flash-image",
    "google/gemini-3-pro-image-preview",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
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
# 论文图表提示词（中文版，带标签）
# ============================================================
THESIS_FIGURES: Dict[str, Dict[str, Any]] = {

    # --------------------------------------------------------
    # 第2章：数据采集与预处理
    # --------------------------------------------------------

    "fig_2_1_data_collection_architecture": {
        "title": "数据采集分层架构图",
        "chapter": 2,
        "description": "农村低压配电网数据采集系统三层架构",
        "prompt": """
生成一张专业的三层架构示意图，用于学术论文。

【图表结构】从上到下三层：

第一层（顶层）- 平台层：
- 中央绘制一个云服务器图标
- 旁边绘制数据库圆柱图标
- 标注"数据中心"
- 标注"异常检测系统"

第二层（中层）- 通信层：
- 绘制3个数据集中器方框
- 每个方框上有无线天线图标
- 用水平线连接三个集中器
- 标注"数据集中器"
- 标注"4G/NB-IoT"

第三层（底层）- 现场层：
- 绘制一排6个简化房屋图标
- 每个房屋旁有小圆圈代表智能电表
- 用水平电力线连接所有房屋
- 中间绘制变压器符号
- 标注"智能电表"
- 标注"配电变压器"
- 标注"农户"

【连接方式】
- 各层之间用垂直箭头连接，箭头向上表示数据流
- 箭头旁标注"电压数据"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1800x1400"
    },

    "fig_2_2_voltage_anomaly_types": {
        "title": "电压异常类型示意图",
        "chapter": 2,
        "description": "四种典型电压异常波形对比",
        "prompt": """
生成一张2x2网格的电压异常波形对比图，用于学术论文。

【图表布局】四个子图，每个子图展示一种异常类型：

左上角 - 电压骤降：
- 绘制正弦波，中间部分幅值突然下降至70%
- 异常区域用浅橙色阴影标记
- 绘制水平虚线表示额定电压
- 标注"电压骤降"

右上角 - 电压骤升：
- 绘制正弦波，中间部分幅值突然上升至120%
- 异常区域用浅橙色阴影标记
- 绘制水平虚线表示额定电压
- 标注"电压骤升"

左下角 - 电压闪变：
- 绘制正弦波，幅值呈周期性波动
- 外包络线显示幅值调制效果
- 标注"电压闪变"

右下角 - 电压中断：
- 绘制正弦波，中间有一段幅值为零
- 中断区域明显可见
- 标注"电压中断"

【每个子图包含】
- X轴：时间（标注"时间/ms"）
- Y轴：电压（标注"电压/V"）
- 浅灰色网格线
- 异常区域高亮

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1800x1400"
    },

    # --------------------------------------------------------
    # 第3章：模型方法
    # --------------------------------------------------------

    "fig_3_1_sliding_window": {
        "title": "滑动窗口预测示意图",
        "chapter": 3,
        "description": "时间序列滑动窗口机制可视化",
        "prompt": """
生成一张滑动窗口机制示意图，用于学术论文。

【图表元素】

1. 时间序列信号：
- 绘制一条横向的电压波形曲线，跨越整个图片宽度
- 曲线呈现周期性波动特征
- 颜色：学术蓝

2. 滑动窗口（三个位置）：
- 绘制三个矩形框，分别位于信号的左、中、右三个位置
- 框线颜色：学术蓝
- 框内填充：半透明浅蓝色
- 三个框等距排列，显示窗口滑动过程
- 左侧框标注"历史窗口"
- 中间框标注"当前窗口"
- 右侧框标注"预测窗口"

3. 滑动方向指示：
- 在窗口下方绘制一个大的水平箭头
- 箭头从左指向右
- 标注"滑动方向"

4. 输入输出关系：
- 在当前窗口下方画括号，标注"输入序列"
- 箭头指向右侧的一个点，标注"预测输出"

5. 窗口长度标注：
- 用双向箭头标注窗口宽度
- 标注"窗口长度 L"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1800x1000"
    },

    "fig_3_2_1d_to_2d_conversion": {
        "title": "一维到二维时序转换示意图",
        "chapter": 3,
        "description": "时间序列从一维到二维的变换过程",
        "prompt": """
生成一张三阶段变换示意图，展示时间序列的维度转换过程。

【图表布局】从左到右三个阶段：

阶段一（左侧）- 一维时间序列：
- 绘制一条水平的波动曲线
- 曲线呈现周期性特征
- 下方标注"一维时间序列"
- 标注维度"(批次, 时间步, 通道)"

阶段二（中间）- 频率分析：
- 绘制垂直条形图（频谱图）
- 5-6个不同高度的条形
- 最高的2-3个条形用学术蓝色（主频率）
- 其余条形用灰色
- 下方标注"频域变换"
- 标注"主周期: p₁, p₂"

阶段三（右侧）- 二维矩阵：
- 绘制8x8的网格矩阵
- 网格填充颜色渐变（白色到蓝色）
- 显示周期性模式
- 下方标注"二维表示"
- 标注维度"(批次, 周期, 时间/周期, 通道)"

【连接箭头】
- 阶段一到阶段二：大箭头，标注"快速傅里叶变换"
- 阶段二到阶段三：大箭头，标注"按周期重塑"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1800x1000"
    },

    "fig_3_7_anomaly_detection_framework": {
        "title": "异常检测框架流程图",
        "chapter": 3,
        "description": "端到端异常检测系统流程",
        "prompt": """
生成一张水平流程图，展示异常检测系统的完整流程。

【图表布局】从左到右六个模块：

模块1 - 数据输入：
- 圆角矩形框
- 内部绘制波形图标
- 标注"原始电压数据"

模块2 - 数据预处理：
- 圆角矩形框
- 内部绘制滤波器图标
- 标注"预处理"
- 下方小字"归一化、缺失值处理"

模块3 - 特征提取：
- 圆角矩形框
- 内部绘制多层神经网络图标（三个堆叠的矩形）
- 标注"特征提取"
- 下方小字"VoltageTimesNet"

模块4 - 序列重构：
- 圆角矩形框
- 内部绘制解码器图标（扩展的层次结构）
- 标注"序列重构"

模块5 - 异常评分：
- 圆角矩形框
- 内部绘制比较/减法符号
- 标注"异常评分"
- 下方小字"重构误差计算"

模块6 - 检测输出：
- 圆角矩形框
- 内部绘制警报图标
- 标注"异常检测结果"

【连接方式】
- 模块之间用水平箭头连接
- 所有框使用学术蓝边框
- 框内白色或浅蓝色填充

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "2000x800"
    },

    "fig_timesnet_architecture": {
        "title": "TimesNet网络架构图",
        "chapter": 3,
        "description": "TimesNet神经网络结构详图",
        "prompt": """
生成一张垂直方向的神经网络架构图，展示TimesNet的结构。

【图表布局】从上到下：

输入层：
- 顶部绘制一个矩形
- 标注"输入张量"
- 标注维度"(B, T, C)"

TimesBlock模块（主体部分，详细展示）：
- 用虚线框包围整个模块
- 框顶部标注"TimesBlock × N层"

模块内部（从上到下）：
1. 小矩形，标注"快速傅里叶变换"
2. 向下箭头
3. 图标表示维度变换，标注"一维→二维重塑"
4. 向下箭头
5. Inception结构（四个并行路径）：
   - 路径1：小方块，标注"1×1卷积"
   - 路径2：小方块→中方块，标注"3×3卷积"
   - 路径3：小方块→大方块，标注"5×5卷积"
   - 路径4：网格图案→小方块，标注"池化"
   - 四个路径在底部汇合
6. 矩形，标注"特征聚合"
7. 矩形，标注"层归一化"

跳跃连接：
- 从模块输入画曲线箭头到模块输出
- 标注"残差连接"

输出层：
- 底部绘制一个矩形
- 标注"输出张量"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1400x2000"
    },

    "fig_voltagetimesnet_architecture": {
        "title": "VoltageTimesNet网络架构图",
        "chapter": 3,
        "description": "VoltageTimesNet架构与增强模块",
        "prompt": """
生成一张对比架构图，展示VoltageTimesNet相对于TimesNet的改进。

【图表布局】左右两列对比：

左列 - 标准TimesNet：
- 标题"TimesNet"
- 垂直流程：
  * 输入框
  * 快速傅里叶变换框
  * 二维卷积框
  * 输出框
- 所有框使用学术蓝色
- 侧边有跳跃连接曲线

右列 - VoltageTimesNet（本文模型）：
- 标题"VoltageTimesNet（本文）"
- 垂直流程与左侧相同
- 额外的增强模块（用橙色高亮）：
  * 在FFT后添加橙色框，标注"领域先验注入"
  * 在输出前添加橙色框，标注"周期权重增强"
- 侧边有跳跃连接曲线

【视觉区分】
- 左侧全部学术蓝
- 右侧基础部分学术蓝，增强部分橙色
- 中间用垂直虚线分隔
- 右侧稍大，表示增强

【标注说明】
- 在橙色框旁边标注"本文贡献"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1800x1600"
    },

    "fig_fft_period_discovery": {
        "title": "快速傅里叶变换周期发现示意图",
        "chapter": 3,
        "description": "FFT周期检测原理图",
        "prompt": """
生成一张上下两个面板的FFT分析示意图。

【图表布局】上下两个子图：

上方面板 - 时域信号：
- 绘制连续的波形曲线
- 波形显示明显的周期性特征（多个重复的波峰波谷）
- X轴标注"时间/采样点"
- Y轴标注"幅值/V"
- 曲线颜色：学术蓝
- 面板标题"时域波形"

下方面板 - 频域频谱：
- 绘制条形图形式的频谱
- X轴标注"频率/Hz"
- Y轴标注"幅度"
- 2-3个最高的条形用学术蓝色，标注"主频率"
- 其余条形用浅灰色
- 在主频率条形上方标注"f₁"、"f₂"
- 面板标题"频域频谱"

【连接指示】
- 两个面板之间绘制大的向下箭头
- 箭头旁标注"快速傅里叶变换"

【周期计算说明】
- 在下方面板旁边添加文字框
- 标注"检测周期: p = T/f"

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1600x1400"
    },

    "fig_2d_conv_inception": {
        "title": "二维卷积Inception模块示意图",
        "chapter": 3,
        "description": "Inception风格的多尺度卷积结构",
        "prompt": """
生成一张Inception模块的结构示意图。

【图表布局】垂直方向，展示数据流：

输入（顶部）：
- 单个矩形框
- 标注"输入特征"

四个并行分支（中间）：
从左到右排列四条并行路径：

分支1（最左）：
- 单个小方块
- 标注"1×1卷积"

分支2：
- 小方块连接中方块
- 上方小方块标注"1×1卷积"
- 下方中方块标注"3×3卷积"

分支3：
- 小方块连接大方块
- 上方小方块标注"1×1卷积"
- 下方大方块标注"5×5卷积"

分支4（最右）：
- 网格图案连接小方块
- 上方网格标注"最大池化"
- 下方小方块标注"1×1卷积"

特征拼接（汇合点）：
- 四个分支在底部汇合成一个宽矩形
- 标注"特征拼接"

输出（底部）：
- 单个矩形框
- 标注"输出特征"

【视觉效果】
- 不同方块大小表示不同感受野
- 用箭头显示数据流向
- 所有模块使用学术蓝填充

""" + THESIS_STYLE_REQUIREMENTS,
        "size": "1600x1800"
    },

    "fig_3_3_voltage_timesnet_comparison": {
        "title": "VoltageTimesNet与TimesNet周期检测对比",
        "chapter": 3,
        "description": "两种模型周期检测机制的对比",
        "prompt": """
生成一张对比图，展示两种周期检测方法的差异。

【图表布局】左右对比，上下两行：

第一行 - 周期来源：

左侧（TimesNet）：
- 标题"TimesNet周期检测"
- 绘制FFT频谱图
- 箭头指向检测到的频率峰值
- 标注"仅依赖FFT检测"
- 显示可能遗漏的小峰值（用虚线圈出）

右侧（VoltageTimesNet）：
- 标题"VoltageTimesNet周期检测"
- 绘制FFT频谱图
- 额外添加预设的电网周期标记（50Hz等）
- 标注"FFT检测 + 领域先验"
- 用橙色标记预设周期

第二行 - 周期权重：

左侧（TimesNet）：
- 绘制等高的权重条形图
- 标注"均等权重"

右侧（VoltageTimesNet）：
- 绘制不等高的权重条形图
- 通过可学习注意力分配权重
- 标注"自适应权重"
- 重要周期权重更高

【视觉区分】
- 左侧使用灰色/蓝色
- 右侧使用蓝色/橙色（强调改进）
- 中间垂直虚线分隔

""" + THESIS_STYLE_REQUIREMENTS,
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
            timeout=180
        )

        if response.status_code == 200:
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            # 检查是否返回了图像（base64 格式）
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                # 提取 base64 数据
                                base64_data = image_url.split(",")[1]
                                return base64.b64decode(base64_data)
                        elif "image" in item.get("type", ""):
                            if "data" in item:
                                return base64.b64decode(item["data"])

            # 检查是否在文本中返回了 base64 图像
            if isinstance(content, str):
                # 尝试提取 base64 图像数据
                base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                match = re.search(base64_pattern, content)
                if match:
                    return base64.b64decode(match.group(1))

            return content.encode() if content else None

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
