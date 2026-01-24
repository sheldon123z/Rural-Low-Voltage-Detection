# 农村低压配电网异常检测研究

基于 LSTM 和混合模型的农村低压配电网异常检测与预测研究。

## 项目结构
- `Research_Guidelines.md`：详细的研究方向和模型应用逻辑
- `code/`：模型源代码
  - `voltage_anomaly_detection/`：基于 Time-Series-Library 的异常检测框架
- `resources/`：研究资料和收集的文献
- `thesis/`：论文相关文档

## 研究背景
本项目聚焦于使用以下方法解决农村电网中的**长期依赖性**和**非线性波动**问题：
- 标准 LSTM 模型
- 混合 CNN-LSTM 模型
- 基于预测的异常检测方法
- LSTM + GNN/GIN 时空模型
- 多变量外部特征增强

## 技术栈
- **框架**：PyTorch + Time-Series-Library
- **模型**：TimesNet、VoltageTimesNet、DLinear 等 15 种深度学习模型
- **数据集**：PSM、MSL、SMAP、SMD、SWAT 及自定义农村电压数据集

## 快速开始
```bash
# 环境配置
conda create -n tslib python=3.11
conda activate tslib
pip install -r code/voltage_anomaly_detection/requirements.txt

# 训练模型
cd code/voltage_anomaly_detection
python run.py --is_training 1 --model TimesNet --data PSM
```

详细使用说明请参考 `CLAUDE.md` 和 `code/voltage_anomaly_detection/README.md`。
