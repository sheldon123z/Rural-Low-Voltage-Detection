# 论文写作材料索引

> **项目**: 农村低压配电网电压异常检测研究
> **生成时间**: 2026-02-01
> **状态**: 完整实验运行中

---

## 文档列表

| 序号 | 文档名称 | 用途 | 对应章节 |
|:----:|---------|------|:--------:|
| 1 | [实验结果汇总_论文写作材料.md](实验结果汇总_论文写作材料.md) | 实验数据、性能指标、关键发现 | 第四章 |
| 2 | [TimesNet算法描述_论文写作材料.md](TimesNet算法描述_论文写作材料.md) | 模型原理、公式推导、改进方法 | 第三章 |
| 3 | [图表资源清单_论文写作材料.md](图表资源清单_论文写作材料.md) | 已有图表、待生成图表、规范 | 全文 |
| 4 | [VoltageTimesNet_v2_results.md](VoltageTimesNet_v2_results.md) | 改进模型实验结果、召回率优化 | 第三/四章 |
| 5 | [项目状态报告_20260201.md](项目状态报告_20260201.md) | 最新项目进度、实验状态 | - |

---

## 实验进度概览

### 已完成实验

| 数据集 | 模型数量 | 最优F1 | 最优模型 | 状态 |
|:------:|:--------:|:------:|:--------:|:----:|
| PSM | 8 | 0.9735 | TimesNet | ✅ 完成 |
| MSL | 1 | 0.7636 | TimesNet | ✅ 完成 |
| SMD | 1 | 0.8246 | TimesNet | ✅ 完成 |
| RuralVoltage | 3 | 0.7113 | VoltageTimesNet_v2 | ✅ 完成 |

### 进行中实验（完整对比实验）

| 数据集 | 模型 | 状态 | 备注 |
|:------:|:----:|:----:|------|
| RuralVoltage | TimesNet | 🔄 epoch 3/10 | 完整实验进行中 |
| RuralVoltage | VoltageTimesNet, VoltageTimesNet_v2 | ⏳ 待运行 | 队列中 |
| RuralVoltage | TPATimesNet, DLinear, PatchTST | ⏳ 待运行 | 队列中 |
| KagglePQ | TimesNet, VoltageTimesNet, ... | ⏳ 待运行 | 队列中 |

### 新增实验内容

- [x] VoltageTimesNet_v2 召回率优化实验 (Recall: 71.13%)
- [x] Kaggle 电力质量数据集适配
- [ ] 完整 RuralVoltage 6模型对比 (进行中)
- [ ] 完整 KagglePQ 5模型对比 (待运行)
- [ ] 阈值敏感性分析

---

## 论文章节对应

```
第一章 绪论
    └── (无专用材料文档)

第二章 数据采集与预处理
    └── 实验结果汇总.md → 数据集说明部分
    └── 图表资源清单.md → 第二章图表

第三章 基于TimesNet的电压异常检测算法
    └── TimesNet算法描述.md (核心)
    └── 图表资源清单.md → 第三章图表

第四章 实验设计与结果分析
    └── 实验结果汇总.md (核心)
    └── 图表资源清单.md → 第四章图表

第五章 监管平台设计与实现
    └── (待补充平台设计文档)

第六章 结论与展望
    └── 实验结果汇总.md → 关键结论部分
```

---

## 快速使用指南

### 1. 查看最新实验结果

```bash
# 运行分析脚本
cd code
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_20260125_013217

# 直接读取结果 JSON
cat results/PSM_comparison_*/analysis_*/实验结果.json
```

### 2. 生成新图表

```bash
cd thesis/figures
python generate_*.py
```

### 3. 更新写作材料

当有新的实验结果时，需要更新:
1. `实验结果汇总_论文写作材料.md` - 添加新的实验数据
2. `图表资源清单_论文写作材料.md` - 添加新生成的图表

---

## 注意事项

1. **不要直接编辑论文正文** - 本目录仅提供写作材料
2. **实验结果持续更新** - 训练完成后及时更新文档
3. **图表命名规范** - 遵循 `[章节]_[内容描述].[格式]` 格式
4. **数值精度** - 性能指标保留 4 位小数

---

## 联系方式

如有问题，请查看:
- 项目 CLAUDE.md: `/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/CLAUDE.md`
- 论文目录: `/home/zhengxiaodong/exps/Rural-Low-Voltage-Detection/thesis/`

---

*索引最后更新: 2026-02-01*
