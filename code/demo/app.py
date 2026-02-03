"""
农村低压配电网电压异常检测 - Gradio 交互式演示
Rural Low-Voltage Distribution Network Voltage Anomaly Detection Demo

用于论文答辩演示，展示 TimesNet 周期建模原理、VoltageTimesNet 创新点、多模型性能对比
"""

import gradio as gr
import sys
from pathlib import Path

# 添加项目路径
DEMO_DIR = Path(__file__).parent
CODE_DIR = DEMO_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DEMO_DIR))

# 导入标签页
from tabs.tab1_principle import create_principle_tab
from tabs.tab2_innovation import create_innovation_tab
from tabs.tab3_arena import create_arena_tab
from tabs.tab4_detection import create_detection_tab

# 导入配置
from config import GRADIO_THEME, THESIS_COLORS


def create_header():
    """创建页面头部"""
    return gr.Markdown(
        """
        # 🔌 农村低压配电网电压异常检测系统

        **基于 TimesNet 的时间序列异常检测方法研究与应用**

        本演示系统用于论文答辩，展示研究成果和模型性能。

        ---
        """
    )


def create_footer():
    """创建页面底部"""
    return gr.Markdown(
        """
        ---

        ### 📚 系统说明

        | 标签页 | 功能 | 说明 |
        |--------|------|------|
        | 原理演示 | FFT 周期发现 | 展示 TimesNet 核心算法原理 |
        | 创新对比 | 模型改进 | VoltageTimesNet 与 TimesNet 的差异对比 |
        | 模型竞技场 | 性能对比 | 6 个模型的多维度性能对比 |
        | 自定义检测 | 实时推理 | 上传 CSV 进行异常检测 |

        **技术栈**: PyTorch + Gradio + Plotly

        **模型**: VoltageTimesNet_v2 (最优) | VoltageTimesNet | TimesNet | TPATimesNet | MTSTimesNet | DLinear

        ---

        <center>

        📧 联系作者 | 📖 [项目文档](https://github.com/sheldon123z/Rural-Low-Voltage-Detection) | 🤗 [HuggingFace](https://huggingface.co/Sheldon123z)

        </center>
        """
    )


def create_app():
    """创建 Gradio 应用"""

    # 加载自定义 CSS
    css_path = DEMO_DIR / "assets" / "custom.css"
    custom_css = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()

    # 创建应用 (Gradio 6.x API)
    with gr.Blocks(title="农村低压配电网电压异常检测系统") as app:

        # 页面头部
        create_header()

        # 标签页
        with gr.Tabs():
            # Tab 1: 原理演示
            create_principle_tab()

            # Tab 2: 创新对比
            create_innovation_tab()

            # Tab 3: 模型竞技场
            create_arena_tab()

            # Tab 4: 自定义检测
            create_detection_tab()

        # 页面底部
        create_footer()

    return app


def main():
    """主函数"""
    app = create_app()

    # 启动服务 (Gradio 6.x API: theme 和 css 移到 launch)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
