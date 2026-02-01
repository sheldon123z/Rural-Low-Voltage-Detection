# Suggested Commands

## Training Commands

### Train TimesNet on PSM dataset
```bash
cd code
python run.py --is_training 1 --model TimesNet --data PSM \
  --root_path ./dataset/PSM/ --seq_len 100 --d_model 64 --train_epochs 10
```

### Train VoltageTimesNet on RuralVoltage dataset
```bash
cd code
python run.py --is_training 1 --model VoltageTimesNet --data RuralVoltage \
  --root_path ./dataset/RuralVoltage/ --enc_in 16 --c_out 16 --seq_len 100
```

### Test only (no training)
```bash
cd code
python run.py --is_training 0 --model TimesNet --data PSM --root_path ./dataset/PSM/
```

## Batch Experiments

### Run comparison experiments
```bash
cd code
bash scripts/PSM/run_comparison.sh
```

### Ablation studies
```bash
cd code
bash scripts/ablation_alpha.sh      # Preset weight ablation
bash scripts/ablation_seq_len.sh    # Sequence length ablation
```

## Analysis Commands

### Analyze experiment results
```bash
cd code
python scripts/analyze_comparison_results.py --result_dir ./results/PSM_comparison_XXXXXX
```

## Development Commands

### Format code
```bash
black code/
isort code/
```

### Run linting
```bash
flake8 code/
```

### Run tests
```bash
pytest tests/
```

## Troubleshooting

### If data loader hangs
Set `--num_workers 0` to disable multiprocessing data loading.

### GPU memory issues
Reduce `--batch_size` or `--d_model`.
