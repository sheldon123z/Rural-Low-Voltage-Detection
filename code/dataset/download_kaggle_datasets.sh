#!/bin/bash
# Kaggle 电力质量数据集批量下载脚本
# 使用前请先配置 ~/.kaggle/kaggle.json

set -e
DATASET_DIR="$(dirname "$0")"
cd "$DATASET_DIR"

echo "========================================"
echo "Kaggle Power Quality Dataset Downloader"
echo "========================================"

# 检查 kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# 检查 API 密钥
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API key not configured."
    echo "Please follow the instructions in kaggle_download_guide.md"
    exit 1
fi

# 数据集 1: Power Quality Fault Detection
echo ""
echo "[1/3] Downloading Power Quality Fault Detection Dataset..."
mkdir -p Kaggle_PowerQuality
kaggle datasets download -d programmer3/power-quality-fault-detection-dataset \
    -p ./Kaggle_PowerQuality/ --force
unzip -o ./Kaggle_PowerQuality/*.zip -d ./Kaggle_PowerQuality/ 2>/dev/null || true
rm -f ./Kaggle_PowerQuality/*.zip 2>/dev/null || true
echo "      Done: ./Kaggle_PowerQuality/"

# 数据集 2: Power Quality Classification
echo ""
echo "[2/3] Downloading Power Quality Classification Dataset..."
mkdir -p Kaggle_PowerQuality_2
kaggle datasets download -d jaideepreddykotla/powerqualitydistributiondataset1 \
    -p ./Kaggle_PowerQuality_2/ --force
unzip -o ./Kaggle_PowerQuality_2/*.zip -d ./Kaggle_PowerQuality_2/ 2>/dev/null || true
rm -f ./Kaggle_PowerQuality_2/*.zip 2>/dev/null || true
echo "      Done: ./Kaggle_PowerQuality_2/"

# 数据集 3: VSB Power Line Fault Detection (较大，约2GB)
echo ""
echo "[3/3] Downloading VSB Power Line Fault Detection Dataset..."
echo "      Warning: This dataset is ~2GB, may take a while..."
mkdir -p VSB_PowerLine
kaggle competitions download -c vsb-power-line-fault-detection \
    -p ./VSB_PowerLine/ --force
unzip -o ./VSB_PowerLine/*.zip -d ./VSB_PowerLine/ 2>/dev/null || true
rm -f ./VSB_PowerLine/*.zip 2>/dev/null || true
echo "      Done: ./VSB_PowerLine/"

echo ""
echo "========================================"
echo "All datasets downloaded successfully!"
echo "========================================"
echo ""
echo "Dataset locations:"
echo "  - Kaggle_PowerQuality/     : Power quality fault detection"
echo "  - Kaggle_PowerQuality_2/   : Power quality classification"
echo "  - VSB_PowerLine/           : Power line fault detection"
