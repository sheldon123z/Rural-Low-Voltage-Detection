#!/usr/bin/env python3
"""
同步图表到论文项目

将 output/ 目录中的图片同步到论文项目的 figures/ 目录

使用方法:
    python sync_to_thesis.py [--dry-run]

选项:
    --dry-run: 只显示将要复制的文件，不执行实际复制
"""

import os
import shutil
import sys

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
TARGET_DIR = '/home/zhengxiaodong/exps/Rural-Voltage-Thesis/figures'


def sync_figures(dry_run=False):
    """同步图表文件"""
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 源目录不存在: {SOURCE_DIR}")
        return

    if not os.path.exists(TARGET_DIR):
        print(f"错误: 目标目录不存在: {TARGET_DIR}")
        return

    print("=" * 60)
    print("同步图表到论文项目")
    print("=" * 60)
    print(f"源目录: {os.path.abspath(SOURCE_DIR)}")
    print(f"目标目录: {TARGET_DIR}")
    if dry_run:
        print("模式: 预览 (dry-run)")
    print("-" * 60)

    copied = 0
    skipped = 0

    for chapter_dir in sorted(os.listdir(SOURCE_DIR)):
        chapter_path = os.path.join(SOURCE_DIR, chapter_dir)
        if not os.path.isdir(chapter_path):
            continue

        target_chapter = os.path.join(TARGET_DIR, chapter_dir)

        print(f"\n{chapter_dir}/")

        for filename in sorted(os.listdir(chapter_path)):
            if not filename.endswith('.png'):
                continue

            source_file = os.path.join(chapter_path, filename)
            target_file = os.path.join(target_chapter, filename)

            # 检查是否需要更新
            need_copy = True
            if os.path.exists(target_file):
                source_mtime = os.path.getmtime(source_file)
                target_mtime = os.path.getmtime(target_file)
                if source_mtime <= target_mtime:
                    need_copy = False

            if need_copy:
                if dry_run:
                    print(f"  [将复制] {filename}")
                else:
                    os.makedirs(target_chapter, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    print(f"  [已复制] {filename}")
                copied += 1
            else:
                skipped += 1

    print("\n" + "=" * 60)
    if dry_run:
        print(f"预览完成: {copied} 个文件将被复制, {skipped} 个文件已是最新")
    else:
        print(f"同步完成: {copied} 个文件已复制, {skipped} 个文件已是最新")
    print("=" * 60)


def main():
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    sync_figures(dry_run)


if __name__ == '__main__':
    main()
