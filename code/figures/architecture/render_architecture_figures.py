#!/usr/bin/env python3
"""
将 draw.io 架构图渲染为论文所需的 PNG 格式。

直接使用已有的 HTML 文件，用 Playwright 渲染。
支持两种 HTML 格式：
1. data-mxgraph 方式 - 直接截取 SVG
2. iframe 方式 - 截取 .diagram-card 元素
"""

import asyncio
import os
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR.parent / "thesis"
HTTP_PORT = 8766


class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """安静的 HTTP 处理器，不输出日志"""
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass  # 静默


def start_http_server(directory: Path, port: int) -> HTTPServer:
    """启动 HTTP 服务器"""
    os.chdir(directory)
    handler = lambda *args, **kwargs: QuietHTTPHandler(*args, directory=str(directory), **kwargs)
    server = HTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def render_html_to_png(html_filename: str, output_path: Path, wait_time: int = 18000, use_iframe_mode: bool = False):
    """使用 Playwright 将 HTML 渲染为 PNG"""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("安装 playwright: pip install playwright && playwright install chromium")
        return False

    async with async_playwright() as p:
        # 使用系统 Chrome（如果可用）或 chromium
        try:
            browser = await p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = await p.chromium.launch(headless=True)

        page = await browser.new_page(viewport={"width": 1920, "height": 1200})

        # 通过 HTTP 服务器访问 HTML 文件
        url = f"http://localhost:{HTTP_PORT}/{html_filename}"
        print(f"  访问: {url}")
        await page.goto(url)

        # 等待 JavaScript 渲染完成
        print(f"  等待渲染... ({wait_time/1000}s)")
        await page.wait_for_timeout(wait_time)

        if use_iframe_mode:
            # iframe 模式：等待 iframe 加载，然后截取 .diagram-card 元素
            try:
                await page.wait_for_selector(".diagram-card", timeout=10000)
                print("  检测到 diagram-card 元素")
            except Exception:
                print("  警告: 未检测到 diagram-card")

            # 等待 iframe 内容加载
            await page.wait_for_timeout(5000)

            # 截取 diagram-card 元素
            try:
                card = await page.query_selector(".diagram-card")
                if card:
                    box = await card.bounding_box()
                    if box and box['width'] > 200 and box['height'] > 100:
                        await card.screenshot(path=str(output_path))
                        print(f"  保存: {output_path} ({int(box['width'])}x{int(box['height'])})")
                        await browser.close()
                        return True

                # 备用：截取整个 body
                print("  使用 body 截图...")
                await page.screenshot(path=str(output_path), full_page=True)
                print(f"  保存: {output_path} (全页)")
            except Exception as e:
                print(f"  截图失败: {e}")
                await page.screenshot(path=str(output_path), full_page=True)

        else:
            # data-mxgraph 模式：等待 SVG 元素
            try:
                await page.wait_for_selector("svg", timeout=25000)
                print("  检测到 SVG 元素")
            except Exception as e:
                print(f"  警告: 未检测到 SVG 元素: {e}")

            # 截取 SVG 元素
            try:
                svg_element = await page.query_selector("svg")
                if svg_element:
                    box = await svg_element.bounding_box()
                    if box and box['width'] > 200 and box['height'] > 100:
                        await svg_element.screenshot(path=str(output_path))
                        print(f"  保存: {output_path} ({int(box['width'])}x{int(box['height'])})")
                        await browser.close()
                        return True

                # 备用：截取 body
                body = await page.query_selector("body")
                if body:
                    await body.screenshot(path=str(output_path))
                    print(f"  保存: {output_path} (body截图)")
                    await browser.close()
                    return True

            except Exception as e:
                print(f"  截图失败: {e}")
                await page.screenshot(path=str(output_path), full_page=True)

        await browser.close()
        return True


async def render_all_figures():
    """渲染所有架构图"""

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 定义要渲染的图表
    # (HTML文件, 输出png名, 是否使用iframe模式)
    figures = [
        # data-mxgraph 方式的文件
        ("fig_2d_conv_inception.html", "fig_2d_conv_inception.png", False),
        # iframe 方式的文件
        ("fig_timesnet_overview_v3.html", "fig_timesnet_architecture.png", True),
        ("fig_voltage_timesnet.html", "fig_voltagetimesnet_architecture.png", True),
        ("fig_fft_period_discovery.html", "fig_fft_period_discovery.png", True),
    ]

    # 启动 HTTP 服务器
    print(f"\n启动 HTTP 服务器 (端口 {HTTP_PORT})...")
    server = start_http_server(SCRIPT_DIR, HTTP_PORT)
    time.sleep(1)  # 等待服务器启动

    try:
        for html_name, output_name, use_iframe in figures:
            html_path = SCRIPT_DIR / html_name
            output_path = OUTPUT_DIR / output_name

            if not html_path.exists():
                print(f"跳过: {html_name} (文件不存在)")
                continue

            print(f"\n处理: {html_name} ({'iframe模式' if use_iframe else 'data-mxgraph模式'})")

            # 渲染为 PNG
            success = await render_html_to_png(html_name, output_path, use_iframe_mode=use_iframe)
            if success:
                print(f"  成功!")
            else:
                print(f"  失败!")

    finally:
        # 关闭 HTTP 服务器
        server.shutdown()
        print("\nHTTP 服务器已关闭")


def main():
    """主函数"""
    print("=" * 60)
    print("Draw.io 架构图渲染工具")
    print("=" * 60)
    print(f"源目录: {SCRIPT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    asyncio.run(render_all_figures())

    print("\n" + "=" * 60)
    print("完成!")


if __name__ == "__main__":
    main()
