#!/usr/bin/env python3
"""
农网低电压问题资料爬取工具
支持下载PDF文件、爬取网页内容并转换为Markdown格式
"""

import os
import sys
import time
import requests
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import html2text
import json
from datetime import datetime

class ResourceScraper:
    def __init__(self, output_dir="collected_resources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.pdf_dir = self.output_dir / "pdfs"
        self.html_dir = self.output_dir / "html_content"
        self.markdown_dir = self.output_dir / "markdown"

        for dir_path in [self.pdf_dir, self.html_dir, self.markdown_dir]:
            dir_path.mkdir(exist_ok=True)

        # 请求头设置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        # HTML转Markdown转换器
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0

        # 下载记录
        self.log_file = self.output_dir / "download_log.json"
        self.download_log = self.load_log()

    def load_log(self):
        """加载下载记录"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"downloaded": [], "failed": []}

    def save_log(self):
        """保存下载记录"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.download_log, f, ensure_ascii=False, indent=2)

    def sanitize_filename(self, filename):
        """清理文件名,移除非法字符"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]  # 限制长度

    def download_pdf(self, url, custom_name=None):
        """下载PDF文件"""
        try:
            print(f"正在下载PDF: {url}")
            response = requests.get(url, headers=self.headers, timeout=30, stream=True)
            response.raise_for_status()

            # 确定文件名
            if custom_name:
                filename = custom_name if custom_name.endswith('.pdf') else f"{custom_name}.pdf"
            else:
                # 从URL或Content-Disposition头获取文件名
                if 'Content-Disposition' in response.headers:
                    filename = response.headers['Content-Disposition'].split('filename=')[-1].strip('"')
                else:
                    filename = os.path.basename(urlparse(url).path)
                    if not filename.endswith('.pdf'):
                        filename = f"{filename}.pdf"

            filename = self.sanitize_filename(filename)
            filepath = self.pdf_dir / filename

            # 下载文件
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"✓ PDF下载成功: {filename}")
            self.download_log["downloaded"].append({
                "url": url,
                "file": str(filepath),
                "type": "pdf",
                "timestamp": datetime.now().isoformat()
            })
            self.save_log()
            return True

        except Exception as e:
            print(f"✗ PDF下载失败: {url}\n  错误: {str(e)}")
            self.download_log["failed"].append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.save_log()
            return False

    def scrape_webpage(self, url, custom_name=None):
        """爬取网页内容并转换为Markdown"""
        try:
            print(f"正在爬取网页: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取标题
            title = soup.find('title')
            title_text = title.text.strip() if title else urlparse(url).path.split('/')[-1]

            # 移除脚本和样式
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()

            # 转换为Markdown
            markdown_content = self.h2t.handle(str(soup))

            # 添加元数据
            metadata = f"""---
title: {title_text}
source: {url}
scraped_date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

"""
            markdown_content = metadata + markdown_content

            # 保存文件
            if custom_name:
                filename = custom_name if custom_name.endswith('.md') else f"{custom_name}.md"
            else:
                filename = self.sanitize_filename(title_text) + ".md"

            filepath = self.markdown_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✓ 网页爬取成功: {filename}")
            self.download_log["downloaded"].append({
                "url": url,
                "file": str(filepath),
                "type": "webpage",
                "timestamp": datetime.now().isoformat()
            })
            self.save_log()
            return True

        except Exception as e:
            print(f"✗ 网页爬取失败: {url}\n  错误: {str(e)}")
            self.download_log["failed"].append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.save_log()
            return False

    def process_url(self, url, custom_name=None, delay=1):
        """自动识别URL类型并处理"""
        # 检查是否已下载
        for item in self.download_log["downloaded"]:
            if item["url"] == url:
                print(f"⊙ 已下载过: {url}")
                return True

        # 延迟以避免过快请求
        time.sleep(delay)

        # 判断类型
        if url.endswith('.pdf') or 'pdf' in url.lower():
            return self.download_pdf(url, custom_name)
        else:
            return self.scrape_webpage(url, custom_name)

    def batch_process(self, url_list):
        """批量处理URL列表"""
        print(f"\n开始批量处理 {len(url_list)} 个URL...\n")
        success_count = 0

        for i, item in enumerate(url_list, 1):
            if isinstance(item, dict):
                url = item.get('url')
                name = item.get('name')
            else:
                url = item
                name = None

            print(f"\n[{i}/{len(url_list)}] 处理: {url}")
            if self.process_url(url, name):
                success_count += 1
            time.sleep(2)  # 批量处理时增加延迟

        print(f"\n{'='*60}")
        print(f"批量处理完成!")
        print(f"成功: {success_count}/{len(url_list)}")
        print(f"失败: {len(url_list) - success_count}/{len(url_list)}")
        print(f"{'='*60}\n")

        return success_count

    def generate_index(self):
        """生成资源索引文件"""
        index_content = f"""# 农网低电压问题资料索引

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 统计信息

- 总下载资源数: {len(self.download_log['downloaded'])}
- 失败资源数: {len(self.download_log['failed'])}

## 已下载资源

### PDF文件
"""

        # PDF文件列表
        pdf_items = [item for item in self.download_log['downloaded'] if item['type'] == 'pdf']
        for i, item in enumerate(pdf_items, 1):
            filename = Path(item['file']).name
            index_content += f"{i}. [{filename}]({item['file']})\n   - 来源: {item['url']}\n   - 时间: {item['timestamp']}\n\n"

        # 网页内容列表
        index_content += "\n### 网页内容\n\n"
        webpage_items = [item for item in self.download_log['downloaded'] if item['type'] == 'webpage']
        for i, item in enumerate(webpage_items, 1):
            filename = Path(item['file']).name
            index_content += f"{i}. [{filename}]({item['file']})\n   - 来源: {item['url']}\n   - 时间: {item['timestamp']}\n\n"

        # 失败列表
        if self.download_log['failed']:
            index_content += "\n## 下载失败的资源\n\n"
            for i, item in enumerate(self.download_log['failed'], 1):
                index_content += f"{i}. {item['url']}\n   - 错误: {item['error']}\n   - 时间: {item['timestamp']}\n\n"

        # 保存索引
        index_file = self.output_dir / "INDEX.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)

        print(f"✓ 资源索引已生成: {index_file}")


def main():
    """主函数"""
    scraper = ResourceScraper()

    # 定义要爬取的资源列表
    resources = [
        # 政策文件和技术报告
        {
            'url': 'https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud.zj.gov.cn/jcms_files/jcms1/web3722/site/attach/0/b9d4ba56cf1f4be3af10f64f86447e8b.pdf',
            'name': '国网龙泉市供电公司乡村振兴五年行动方案'
        },
        {
            'url': 'http://zfxxgk.nea.gov.cn/1310783622_17225787362441n.pdf',
            'name': '2024年能源领域行业标准制定计划'
        },
        {
            'url': 'https://rmi.org.cn/wp-content/uploads/2025/12/final-1217-中国分布式光伏韧性发展路径：2026与2027年展望报告-1.pdf',
            'name': '中国分布式光伏韧性发展路径2026-2027展望报告'
        },
        {
            'url': 'https://pdf.dfcfw.com/pdf/H301_AP202501241642522180_1.pdf',
            'name': '电网专题研究报告2025'
        },

        # 学术论文
        {
            'url': 'https://academic.oup.com/ijlct/article/doi/10.1093/ijlct/ctae221/7901310',
            'name': 'High-precision_identification_low-voltage_load_smart_grids'
        },
        {
            'url': 'https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2486136',
            'name': 'Forecasting_short-term_power_load_hybrid_interpretable_deep_models'
        },
        {
            'url': 'https://arxiv.org/html/2408.16202v1',
            'name': 'Short-Term_Electricity-Load_Forecasting_Deep_Learning_Survey'
        },
        {
            'url': 'https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full',
            'name': 'Enhanced_LSTM_robotic_agent_load_forecasting'
        },
        {
            'url': 'https://arxiv.org/html/2502.08376v1',
            'name': 'Enhanced_Load_Forecasting_GAT-LSTM'
        },
        {
            'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10909240/',
            'name': 'RNN-BiLSTM-CRF_electricity_theft_detection'
        },

        # 中文资源
        {
            'url': 'https://image.hanspub.org/Html/5-2610275_48873.htm',
            'name': '基于深度学习技术的电表大数据检测系统'
        },
        {
            'url': 'https://pdf.hanspub.org/AIRR20220100000_89162339.pdf',
            'name': '人工智能与机器人研究2022'
        },

        # 新闻和行业资讯
        {
            'url': 'https://www.stcn.com/article/detail/1874069.html',
            'name': '数千亿元电网投资勾勒能源变革新版图'
        },
        {
            'url': 'https://www.sxgfw.com/xiangqing?article_id=1139',
            'name': '山西长治政企联手开展分布式光伏电压越限治理'
        },
        {
            'url': 'https://www.ndrc.gov.cn/xxgk/zcfb/ghxwj/202307/t20230714_1358371.html',
            'name': '关于实施农村电网巩固提升工程的指导意见'
        },
        {
            'url': 'https://zhuanlan.zhihu.com/p/133771041',
            'name': '六大案例解析电力行业如何应用大数据'
        },
    ]

    # 批量处理
    scraper.batch_process(resources)

    # 生成索引
    scraper.generate_index()

    print("\n所有任务完成!")
    print(f"资源保存位置: {scraper.output_dir}")


if __name__ == "__main__":
    main()
