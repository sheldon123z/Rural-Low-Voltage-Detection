"""
Kaggle Power Quality Dataset Preparation Script V2

改进版数据准备脚本，支持可配置的异常比例。

设计原则:
1. 训练集：只包含正常数据（单类学习）
2. 测试集：正常数据 + 部分异常数据，控制异常比例
3. 各类异常均衡采样，保持异常类型多样性

数据集信息:
- 原始数据: 11,998 条记录，128 维波形特征
- 5 类: 1=Transient, 2=Sag, 3=Normal, 4=Swell, 5=Harmonics
- 每类约 2,400 样本（均衡分布）

输出文件:
- train.csv: 训练数据（仅正常样本）
- test.csv: 测试数据（正常 + 部分异常）
- test_label.csv: 二值标签（0=正常, 1=异常）

Usage:
    python prepare_dataset_v2.py --anomaly_ratio 0.15
    python prepare_dataset_v2.py --anomaly_ratio 0.20
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_dataset_v2(
    anomaly_ratio: float = 0.15,
    train_ratio: float = 0.70,
    normal_class: int = 3,
    random_seed: int = 42,
    output_dir: str = None
):
    """
    准备适合异常检测的数据集。

    Args:
        anomaly_ratio: 测试集中的异常比例 (0.0-1.0)，默认 15%
        train_ratio: 正常数据用于训练的比例，默认 70%
        normal_class: 定义为正常的类别，默认 Class 3
        random_seed: 随机种子
        output_dir: 输出目录，默认为脚本所在目录

    Returns:
        train_df, test_df, label_df
    """
    np.random.seed(random_seed)

    # 确定输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = script_dir

    # 加载原始数据
    input_file = os.path.join(script_dir, "PowerQualityDistributionDataset1.csv")
    print(f"Loading: {input_file}")

    df = pd.read_csv(input_file, index_col=0)
    print(f"Original shape: {df.shape}")

    # 分离特征和标签
    feature_cols = [col for col in df.columns if col != 'output']
    X = df[feature_cols].values  # (11998, 128)
    y = df['output'].values      # (11998,)

    # 显示原始类别分布
    print(f"\n原始类别分布:")
    class_names = {1: 'Transient', 2: 'Sag', 3: 'Normal', 4: 'Swell', 5: 'Harmonics'}
    for cls in sorted(np.unique(y)):
        count = np.sum(y == cls)
        name = class_names.get(cls, f'Class{cls}')
        print(f"  Class {cls} ({name}): {count} ({100*count/len(y):.1f}%)")

    # 按类别分离数据
    normal_mask = (y == normal_class)
    X_normal = X[normal_mask]
    y_original = y[~normal_mask]  # 保留原始异常类别信息
    X_anomaly = X[~normal_mask]

    # 按异常类别分组
    anomaly_classes = [c for c in np.unique(y) if c != normal_class]
    X_by_class = {}
    for cls in anomaly_classes:
        mask = (y == cls)
        X_by_class[cls] = X[mask]
        print(f"  异常类 {cls}: {len(X_by_class[cls])} 样本")

    print(f"\n正常数据 (Class {normal_class}): {len(X_normal)} 样本")
    print(f"异常数据 (其他类): {len(X_anomaly)} 样本")

    # === 划分训练集和测试集 ===

    # 1. 正常数据划分
    X_normal_train, X_normal_test = train_test_split(
        X_normal,
        train_size=train_ratio,
        random_state=random_seed
    )

    n_normal_test = len(X_normal_test)
    print(f"\n训练集 (仅正常): {len(X_normal_train)} 样本")
    print(f"测试集正常部分: {n_normal_test} 样本")

    # 2. 计算需要的异常样本数量
    # anomaly_ratio = n_anomaly / (n_normal + n_anomaly)
    # n_anomaly = anomaly_ratio * n_normal / (1 - anomaly_ratio)
    n_anomaly_needed = int(anomaly_ratio * n_normal_test / (1 - anomaly_ratio))

    # 每类异常均衡采样
    n_per_class = n_anomaly_needed // len(anomaly_classes)
    remainder = n_anomaly_needed % len(anomaly_classes)

    print(f"\n目标异常比例: {anomaly_ratio*100:.1f}%")
    print(f"需要异常样本: {n_anomaly_needed} 个")
    print(f"每类采样: {n_per_class} 个 (共 {len(anomaly_classes)} 类)")

    # 3. 从每类异常中采样
    X_anomaly_test = []
    y_anomaly_class = []  # 记录原始类别（用于分析）

    for i, cls in enumerate(anomaly_classes):
        X_cls = X_by_class[cls]
        n_sample = n_per_class + (1 if i < remainder else 0)
        n_sample = min(n_sample, len(X_cls))  # 不超过可用样本数

        indices = np.random.choice(len(X_cls), size=n_sample, replace=False)
        X_anomaly_test.append(X_cls[indices])
        y_anomaly_class.extend([cls] * n_sample)

        print(f"  Class {cls} ({class_names[cls]}): 采样 {n_sample} 个")

    X_anomaly_test = np.vstack(X_anomaly_test)
    y_anomaly_class = np.array(y_anomaly_class)

    # 4. 组合测试集
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)),   # 0 = 正常
        np.ones(len(X_anomaly_test))    # 1 = 异常
    ])

    # 保存原始类别信息（用于详细分析）
    y_test_original_class = np.concatenate([
        np.full(len(X_normal_test), normal_class),
        y_anomaly_class
    ])

    # 5. 打乱测试集
    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]
    y_test_original_class = y_test_original_class[shuffle_idx]

    # === 保存数据 ===

    col_names = [f'Col{i+1}' for i in range(X_normal_train.shape[1])]

    # 训练集
    train_df = pd.DataFrame(X_normal_train, columns=col_names)
    train_file = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_file, index=False)
    print(f"\n保存: {train_file} ({train_df.shape})")

    # 测试集
    test_df = pd.DataFrame(X_test, columns=col_names)
    test_file = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_file, index=False)
    print(f"保存: {test_file} ({test_df.shape})")

    # 二值标签
    label_df = pd.DataFrame(y_test.astype(int), columns=['label'])
    label_file = os.path.join(output_dir, "test_label.csv")
    label_df.to_csv(label_file, index=False)
    print(f"保存: {label_file} ({label_df.shape})")

    # 详细类别标签（可选，用于分析）
    detail_label_df = pd.DataFrame({
        'label': y_test.astype(int),
        'original_class': y_test_original_class.astype(int)
    })
    detail_file = os.path.join(output_dir, "test_label_detailed.csv")
    detail_label_df.to_csv(detail_file, index=False)
    print(f"保存: {detail_file} (包含原始类别信息)")

    # === 统计摘要 ===

    actual_anomaly_ratio = np.sum(y_test == 1) / len(y_test)

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"训练集: {len(train_df)} 样本 (100% 正常)")
    print(f"测试集: {len(test_df)} 样本")
    print(f"  - 正常: {int(np.sum(y_test == 0))} ({100*(1-actual_anomaly_ratio):.1f}%)")
    print(f"  - 异常: {int(np.sum(y_test == 1))} ({100*actual_anomaly_ratio:.1f}%)")
    print(f"实际异常比例: {100*actual_anomaly_ratio:.2f}%")
    print(f"特征维度: {X_normal_train.shape[1]}")

    # 测试集异常类别分布
    print(f"\n测试集异常类别分布:")
    for cls in anomaly_classes:
        count = np.sum(y_test_original_class == cls)
        print(f"  Class {cls} ({class_names[cls]}): {count}")

    return train_df, test_df, label_df


def main():
    parser = argparse.ArgumentParser(
        description='准备 Kaggle Power Quality 异常检测数据集 (V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python prepare_dataset_v2.py                    # 默认 15% 异常比例
  python prepare_dataset_v2.py --anomaly_ratio 0.10   # 10% 异常比例
  python prepare_dataset_v2.py --anomaly_ratio 0.20   # 20% 异常比例
        """
    )

    parser.add_argument(
        '--anomaly_ratio', type=float, default=0.15,
        help='测试集中的异常比例 (0.0-1.0)，默认 0.15 (15%%)'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.70,
        help='正常数据用于训练的比例，默认 0.70 (70%%)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子，默认 42'
    )

    args = parser.parse_args()

    if not 0.0 < args.anomaly_ratio < 1.0:
        parser.error("anomaly_ratio 必须在 0.0 到 1.0 之间")

    if not 0.0 < args.train_ratio < 1.0:
        parser.error("train_ratio 必须在 0.0 到 1.0 之间")

    prepare_dataset_v2(
        anomaly_ratio=args.anomaly_ratio,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
