#!/usr/bin/env python
"""
快速验证Bug修复脚本
测试内容:
1. MTSTimesNet 实例级配置（避免类属性污染）
2. RuralVoltageSegLoader 标签修复（train/val返回全零标签）
3. VoltageTimesNet expand优化
"""

import sys
sys.path.insert(0, '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code')

import torch
import numpy as np
from types import SimpleNamespace

print("=" * 60)
print("Bug修复验证测试")
print("=" * 60)

# 测试1: MTSTimesNet 类属性隔离
print("\n[测试1] MTSTimesNet 实例级配置隔离")
print("-" * 40)

from models.MTSTimesNet import Model as MTSModel

# 创建两个不同seq_len的配置
configs1 = SimpleNamespace(
    task_name='anomaly_detection',
    seq_len=100,
    pred_len=0,
    label_len=0,
    enc_in=16,
    d_model=64,
    d_ff=128,
    embed='timeF',
    freq='s',
    dropout=0.1,
    e_layers=2,
    top_k=3,
    num_kernels=6,
    c_out=16,
)

configs2 = SimpleNamespace(
    task_name='anomaly_detection',
    seq_len=50,  # 不同的seq_len
    pred_len=0,
    label_len=0,
    enc_in=16,
    d_model=64,
    d_ff=128,
    embed='timeF',
    freq='s',
    dropout=0.1,
    e_layers=2,
    top_k=3,
    num_kernels=6,
    c_out=16,
)

# 创建两个模型实例
model1 = MTSModel(configs1)
model2 = MTSModel(configs2)

# 检查实例配置是否独立
print(f"Model1 seq_len=100, scale_configs: {model1.scale_configs}")
print(f"Model2 seq_len=50, scale_configs: {model2.scale_configs}")

# 验证类属性未被污染
print(f"类属性 SCALE_CONFIGS: {MTSModel.SCALE_CONFIGS}")

# 检查实例配置不同
if model1.scale_configs != model2.scale_configs:
    print("✅ 测试通过: 实例配置独立，类属性未被污染")
else:
    print("❌ 测试失败: 配置仍然共享")

# 测试2: RuralVoltageSegLoader 标签修复
print("\n[测试2] RuralVoltageSegLoader 标签修复")
print("-" * 40)

import os
data_dir = '/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/code/dataset/RuralVoltage'

if os.path.exists(os.path.join(data_dir, 'train.csv')):
    from data_provider.data_loader import RuralVoltageSegLoader

    # 创建训练集加载器
    # RuralVoltageSegLoader(args, root_path, win_size, step, flag)
    args = SimpleNamespace(
        task_name='anomaly_detection',
        root_path=data_dir,
        data='RuralVoltage'
    )
    train_dataset = RuralVoltageSegLoader(args, data_dir, win_size=100, step=10, flag='train')

    # 获取一个样本
    data, label = train_dataset[0]

    print(f"训练数据形状: {data.shape}")
    print(f"训练标签形状: {label.shape}")
    print(f"训练标签唯一值: {np.unique(label)}")

    # 验证训练标签是否为全零
    if np.all(label == 0):
        print("✅ 测试通过: 训练集标签正确为全零（表示正常数据）")
    else:
        print("❌ 测试失败: 训练集标签不为零")

    # 测试测试集
    test_dataset = RuralVoltageSegLoader(args, data_dir, win_size=100, step=10, flag='test')
    test_data, test_label = test_dataset[0]
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_label.shape}")
    print(f"测试标签唯一值: {np.unique(test_label)}")
else:
    print("⚠️ 数据集不存在，跳过此测试")

# 测试3: VoltageTimesNet expand优化
print("\n[测试3] VoltageTimesNet expand优化验证")
print("-" * 40)

from models.VoltageTimesNet import Model as VTNModel

configs_vtn = SimpleNamespace(
    task_name='anomaly_detection',
    seq_len=100,
    pred_len=0,
    label_len=0,
    enc_in=16,
    d_model=64,
    d_ff=128,
    embed='timeF',
    freq='s',
    dropout=0.1,
    e_layers=2,
    top_k=5,
    num_kernels=6,
    c_out=16,
)

model_vtn = VTNModel(configs_vtn)

# 测试前向传播
batch_size = 4
x = torch.randn(batch_size, 100, 16)

try:
    with torch.no_grad():
        output = model_vtn(x, None, None, None)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 测试通过: VoltageTimesNet前向传播正常")
except Exception as e:
    print(f"❌ 测试失败: {e}")

print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)
