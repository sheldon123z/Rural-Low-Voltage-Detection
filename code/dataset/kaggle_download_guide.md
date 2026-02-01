# Kaggle 电力质量数据集下载指南

## 1. 配置 Kaggle API

### 获取 API 密钥

1. 登录 [Kaggle](https://www.kaggle.com/)
2. 点击右上角头像 → **Settings**
3. 滚动到 **API** 部分
4. 点击 **Create New Token**
5. 下载 `kaggle.json` 文件

### 配置密钥

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 2. 推荐数据集

### 数据集 A：Power Quality Fault Detection Dataset ⭐推荐

- **链接**: https://www.kaggle.com/datasets/programmer3/power-quality-fault-detection-dataset
- **特点**: 包含多种电力质量故障类型的分类数据

```bash
# 下载命令
kaggle datasets download -d programmer3/power-quality-fault-detection-dataset \
    -p ./code/dataset/Kaggle_PowerQuality/
unzip -o ./code/dataset/Kaggle_PowerQuality/*.zip -d ./code/dataset/Kaggle_PowerQuality/
```

### 数据集 B：Power Quality Classification Dataset

- **链接**: https://www.kaggle.com/datasets/jaideepreddykotla/powerqualitydistributiondataset1
- **特点**: 电力质量扰动分类

```bash
kaggle datasets download -d jaideepreddykotla/powerqualitydistributiondataset1 \
    -p ./code/dataset/Kaggle_PowerQuality_2/
unzip -o ./code/dataset/Kaggle_PowerQuality_2/*.zip -d ./code/dataset/Kaggle_PowerQuality_2/
```

### 数据集 C：VSB Power Line Fault Detection

- **链接**: https://www.kaggle.com/c/vsb-power-line-fault-detection
- **特点**: 高频信号中检测部分放电（800,000个信号采样点）

```bash
kaggle competitions download -c vsb-power-line-fault-detection \
    -p ./code/dataset/VSB_PowerLine/
unzip -o ./code/dataset/VSB_PowerLine/*.zip -d ./code/dataset/VSB_PowerLine/
```

## 3. 手动下载方式（无需 API）

如果不想配置 API，可直接在浏览器中访问上述链接并点击 **Download** 按钮。

## 4. 数据集特征对比

| 数据集 | 类型 | 特征 | 适用场景 |
|--------|------|------|----------|
| Power Quality Fault Detection | 分类 | 电压电流波形 | 故障分类 |
| Power Quality Classification | 分类 | 扰动信号 | 扰动检测 |
| VSB Power Line | 二分类 | 高频信号 | 放电检测 |
| Swiss Smart Meter | 时序 | 电压/频率/温度 | 智能电表异常 |

## 5. 快速下载脚本

配置好 API 后，运行以下脚本一键下载所有数据集：

```bash
cd /home/zhengxiaodong/exps/Rural-Low-Voltage-Detection
bash code/dataset/download_kaggle_datasets.sh
```
