from fastapi import APIRouter
from core.config import MODEL_METRICS

router = APIRouter(prefix="/models", tags=["模型"])

def success_response(data):
    return {"code": 200, "message": "success", "data": data}

@router.get("")
async def list_models():
    """获取所有模型列表及性能指标"""
    return success_response(list(MODEL_METRICS.values()))

@router.get("/current")
async def get_current_model():
    """获取当前加载的模型信息"""
    return success_response(MODEL_METRICS["VoltageTimesNet"])
