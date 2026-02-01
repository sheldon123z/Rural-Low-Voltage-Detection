# Task Completion Checklist

## Before Marking Task Complete

### Code Quality
- [ ] Run `black code/` for formatting
- [ ] Run `isort code/` for import sorting
- [ ] Run `flake8 code/` for linting (no errors)
- [ ] Type hints added where appropriate

### Testing
- [ ] Run `pytest tests/` if test files exist
- [ ] Manual testing with small dataset if modifying models
- [ ] Check model can be instantiated: `python -c "from models import model_dict; print(model_dict.keys())"`

### Documentation
- [ ] Update docstrings for modified functions
- [ ] Update CLAUDE.md if adding new models or features
- [ ] Update relevant docs in `code/docs/` if applicable

### Git
- [ ] Check `git status` for untracked files
- [ ] Stage relevant changes only
- [ ] Write descriptive commit message

## Quick Validation Script
```bash
cd code
# Format
black . && isort .
# Lint
flake8 .
# Import test
python -c "from models import model_dict; from exp import Exp_Anomaly_Detection; print('OK')"
```
