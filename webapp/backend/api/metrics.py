from pathlib import Path
from fastapi import APIRouter, Depends
from sqlmodel import Session, select, create_engine
from core.config import DATABASE_URL, BEST_MODEL_PATH
from models.database import DetectionTask

router = APIRouter(prefix="/metrics", tags=["系统指标"])
engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session

def success_response(data):
    return {"code": 200, "message": "success", "data": data}

@router.get("")
async def get_metrics(session: Session = Depends(get_session)):
    """系统运行统计"""
    tasks = session.exec(select(DetectionTask)).all()
    completed = [t for t in tasks if t.status == "completed"]

    total_detections = len(completed)
    total_anomalies = sum(t.anomaly_count or 0 for t in completed)
    avg_time = (
        sum(t.processing_time_ms or 0 for t in completed) / total_detections
        if total_detections > 0 else 0
    )

    # 检查模型权重文件是否存在
    model_ready = BEST_MODEL_PATH.exists()

    return success_response({
        "total_detections": total_detections,
        "total_anomalies_found": total_anomalies,
        "avg_processing_time_ms": round(avg_time, 1),
        "model_f1": 0.8149,
        "model_recall": 0.9110,
        "model_precision": 0.7371,
        "model_ready": model_ready,
        "model_path": str(BEST_MODEL_PATH),
    })

@router.get("/model-status")
async def get_model_status():
    """检查深度学习模型加载状态"""
    model_file_exists = BEST_MODEL_PATH.exists()
    return success_response({
        "model_file_exists": model_file_exists,
        "model_path": str(BEST_MODEL_PATH),
        "message": (
            "VoltageTimesNet 模型已就绪" if model_file_exists
            else "模型权重文件不存在，将在首次检测时尝试自动下载，或使用统计方法检测"
        ),
    })
