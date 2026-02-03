"""冒烟测试：验证所有模型可以正确导入。"""

import sys
from pathlib import Path

# 确保 code/ 目录在 sys.path 中
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def test_model_dict_not_empty():
    """model_dict 应包含所有已注册模型。"""
    from models import model_dict

    assert len(model_dict) > 0, "model_dict 为空"
    assert "TimesNet" in model_dict
    assert "VoltageTimesNet_v2" in model_dict


def test_all_models_importable():
    """每个注册模型类应可正常导入（即类对象可调用）。"""
    from models import model_dict

    for name, cls in model_dict.items():
        assert callable(cls), f"模型 {name} 不可调用"


def test_get_model_invalid_raises():
    """传入无效模型名应抛出 ValueError。"""
    import argparse

    from models import get_model

    args = argparse.Namespace(model="NonExistentModel999")
    try:
        get_model(args)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
