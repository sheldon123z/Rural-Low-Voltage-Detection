#!/usr/bin/env python3
"""
生成所有论文图表

使用方法:
    python generate_all_figures.py [chapter]

参数:
    chapter: 可选，指定生成特定章节的图表 (2, 3, 4)
             不指定则生成所有章节

输出目录: ../output/chapX/
"""

import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 按章节组织的脚本列表
SCRIPTS = {
    2: [
        'fig_2_1_data_collection_architecture.py',
        'fig_2_2_voltage_anomaly_types.py',
    ],
    3: [
        'fig_3_1_sliding_window.py',
        'fig_3_2_1d_to_2d_conversion.py',
        'fig_3_3_voltage_timesnet_comparison.py',
        'fig_3_7_anomaly_detection_framework.py',
    ],
    4: [
        'fig_4_1_f1_comparison.py',
        'fig_4_2_roc_pr_curves.py',
        'fig_4_3_confusion_matrices.py',
        'fig_4_4_precision_recall_tradeoff.py',
        'fig_4_5_radar_comparison.py',
        'fig_4_6_score_distribution.py',
        'fig_4_7_threshold_sensitivity.py',
        'fig_4_8_detection_visualization.py',
        'fig_4_9_training_loss.py',
        'fig_4_10_seq_len_ablation.py',
        'fig_4_11_alpha_ablation.py',
        'fig_4_12_variant_bar_comparison.py',
        'fig_4_13_variant_training_loss.py',
        'fig_4_14_phase_attention_heatmap.py',
        'fig_4_15_multiscale_contribution.py',
    ],
}


def run_script(script_name):
    """运行单个脚本"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  跳过: {script_name} (文件不存在)")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            # 提取输出中的生成文件信息
            for line in result.stdout.split('\n'):
                if '已生成' in line or '已保存' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"  错误: {script_name}")
            print(f"    {result.stderr[:200] if result.stderr else result.stdout[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  超时: {script_name}")
        return False
    except Exception as e:
        print(f"  异常: {script_name}: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("论文图表生成工具")
    print("=" * 60)

    # 解析命令行参数
    chapters = list(SCRIPTS.keys())
    if len(sys.argv) > 1:
        try:
            chapters = [int(sys.argv[1])]
        except ValueError:
            print(f"无效的章节号: {sys.argv[1]}")
            sys.exit(1)

    start_time = time.time()
    total_scripts = 0
    success_count = 0

    for chapter in chapters:
        if chapter not in SCRIPTS:
            print(f"章节 {chapter} 没有配置脚本")
            continue

        scripts = SCRIPTS[chapter]
        print(f"\n第{chapter}章 ({len(scripts)} 个脚本)")
        print("-" * 40)

        for script in scripts:
            total_scripts += 1
            if run_script(script):
                success_count += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"完成! 成功 {success_count}/{total_scripts}, 耗时 {elapsed:.1f}s")
    print(f"输出目录: {os.path.join(SCRIPT_DIR, '..', 'output')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
