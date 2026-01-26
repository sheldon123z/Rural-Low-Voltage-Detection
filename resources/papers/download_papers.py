#!/usr/bin/env python3
"""
Download time series papers to resources/papers/
"""

import os
import requests
import time
from pathlib import Path

# Create papers directory if it doesn't exist
PAPERS_DIR = Path(
    "/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/resources/papers"
)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Papers to download
PAPERS = {
    "TimesNet": {
        "url": "https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf",
        "authors": "Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long",
        "venue": "ICLR 2023",
        "year": 2023,
        "arxiv": "2210.02186",
    },
    "DLinear": {
        "url": "https://arxiv.org/pdf/2012.07436v2.pdf",
        "authors": "Mingze Zeng, Tuo Sheng, Kexin Yang, Weijun Chen, Ming Jin, Jiaqi Zhai",
        "venue": "arXiv",
        "year": 2022,
        "arxiv": "2012.07436",
        "note": "Are Transformers Effective for Time Series Forecasting?",
    },
    "PatchTST": {
        "url": "https://arxiv.org/pdf/2211.14730.pdf",
        "authors": "Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam",
        "venue": "ICLR 2023",
        "year": 2023,
        "arxiv": "2211.14730",
    },
    "iTransformer": {
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf",
        "authors": "Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long",
        "venue": "ICLR 2024",
        "year": 2024,
        "arxiv": "2310.06625",
    },
    "Autoformer": {
        "url": "https://ise.thss.tsinghua.edu.cn/~mlong/doc/Autoformer-nips21.pdf",
        "authors": "Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long",
        "venue": "NeurIPS 2021",
        "year": 2021,
        "arxiv": "2106.13008",
    },
    "Informer": {
        "url": "https://cdn.aaai.org/ojs/17325/17325-13-20819-1-2-20210518.pdf",
        "authors": "Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang",
        "venue": "AAAI 2021",
        "year": 2021,
        "arxiv": "2012.07436",
    },
    "Reformer": {
        "url": "https://arxiv.org/pdf/2001.04451.pdf",
        "authors": "Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya",
        "venue": "ICLR 2020",
        "year": 2020,
        "arxiv": "2001.04451",
    },
    "FEDformer": {
        "url": "https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf",
        "authors": "Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin",
        "venue": "ICML 2022",
        "year": 2022,
        "arxiv": "2201.12740",
    },
    "LightGTS": {
        "url": "https://openreview.net/attachment?id=Z5FJsp1U3Z&name=pdf",
        "authors": "Yihang Wang, Yuying Qiu, Peng Chen, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, Chenjuan Guo",
        "venue": "ICML 2025",
        "year": 2025,
        "arxiv": "2506.06005",
    },
}


def download_paper(name, info):
    """Download a paper and save it to the papers directory"""
    url = info["url"]
    authors = info["authors"]
    venue = info["venue"]
    year = info["year"]
    arxiv_id = info.get("arxiv", "")

    # Clean filename
    safe_name = name.replace(" ", "_").replace("/", "_").replace(":", "_")
    filename = f"{safe_name}_{venue}_{year}.pdf"
    filepath = PAPERS_DIR / filename

    # Skip if already exists
    if filepath.exists():
        print(f"✓ {name} already exists: {filename}")
        return True

    print(f"⬇ {name}: {filename}")
    print(f"  Authors: {authors}")
    print(f"  Venue: {venue} ({year})")
    if arxiv_id:
        print(f"  arXiv: {arxiv_id}")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Save file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = filepath.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {size_mb:.2f} MB")
        print()

        # Small delay to be respectful
        time.sleep(0.5)
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print()
        return False


def main():
    """Main function to download all papers"""
    print("=" * 80)
    print("Downloading Time Series Papers to resources/papers/")
    print("=" * 80)
    print()

    success_count = 0
    fail_count = 0

    for name, info in PAPERS.items():
        if download_paper(name, info):
            success_count += 1
        else:
            fail_count += 1

    print("=" * 80)
    print(f"Summary:")
    print(f"  ✓ Downloaded: {success_count} papers")
    print(f"  ✗ Failed: {fail_count} papers")
    print(f"  Total: {success_count + fail_count} papers")
    print(f"  Location: {PAPERS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
