# Rural Low Voltage Detection Project Overview

## Purpose
时序异常检测系统，用于农村低压配电网电压监控。基于 Time-Series-Library (TSLib) 框架，采用 TimesNet (ICLR 2023) 的重构方法进行异常检测。

## Tech Stack
- **Python** >= 3.9
- **PyTorch** >= 1.10.0
- **NumPy** >= 1.21.0
- **Pandas** >= 1.3.0
- **scikit-learn** >= 0.24.0
- **matplotlib** >= 3.4.0

## Project Structure
```
code/
├── run.py                          # Main entry point
├── exp/exp_anomaly_detection.py    # Training/testing logic
├── models/                         # Model implementations (20+ models)
│   ├── TimesNet.py                 # Core: FFT + 2D Conv
│   ├── VoltageTimesNet.py          # Preset periods + FFT hybrid
│   ├── TPATimesNet.py              # Three-phase attention
│   ├── MTSTimesNet.py              # Multi-scale temporal
│   ├── AdaptiveVoltageTimesNet.py  # Adaptive voltage detection
│   └── ...                         # Other baseline models
├── data_provider/                  # Data loading
│   ├── data_loader.py              # Dataset classes
│   └── data_factory.py             # DataLoader factory
├── layers/                         # Network components
│   ├── Conv_Blocks.py              # Inception blocks
│   ├── ThreePhaseAttention.py      # Voltage-specific attention
│   └── ...
├── utils/                          # Utilities
├── scripts/                        # Training scripts
└── results/                        # Experiment results
```

## Supported Datasets
- PSM (25 features)
- MSL (55 features)
- SMAP (25 features)
- SMD
- SWAT
- RuralVoltage (16 features) - Custom dataset

## Key Configuration Parameters
- `--seq_len`: Input sequence length (default: 100)
- `--d_model`: Hidden dimension (default: 64)
- `--e_layers`: Encoder layers (default: 2)
- `--top_k`: Number of periods in TimesNet (default: 5)
- `--enc_in`: Input features (PSM=25, RuralVoltage=16)
- `--preset_weight`: Weight for preset periods in VoltageTimesNet (default: 0.3)
