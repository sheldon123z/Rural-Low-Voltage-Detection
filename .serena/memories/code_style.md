# Code Style and Conventions

## Formatting
- **Black** for code formatting (line length: 88)
- **isort** for import sorting (profile: black)
- Python version: 3.9+

## Naming Conventions
- Classes: PascalCase (e.g., `TimesNet`, `VoltageTimesBlock`)
- Functions/methods: snake_case (e.g., `FFT_for_Period`, `_build_model`)
- Constants: UPPER_CASE (e.g., `VOLTAGE_PRESET_PERIODS`)
- Variables: snake_case

## Documentation
- Docstrings for all public functions and classes
- Google-style docstrings with Args, Returns sections
- Type hints encouraged but not mandatory

## Model Implementation Pattern
All models must:
1. Be in `code/models/` directory
2. Define a `Model` class
3. Register in `model_dict` in `__init__.py`
4. Implement methods: `forecast()`, `anomaly_detection()`, `forward()`
5. Return same dimensions as input for anomaly detection

## File Organization
- One model class per file
- Layers go in `code/layers/`
- Utility functions go in `code/utils/`
- Training scripts go in `code/scripts/`

## Excluded from Linting
- checkpoints/
- dataset/
- test_results/
- __pycache__/
