import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, select, create_engine
import aiofiles

from core.config import (
    DATABASE_URL, UPLOAD_DIR, RESULTS_DIR, RURAL_VOLTAGE_DIR, FEATURE_COLUMNS
)
from models.database import DetectionTask
from services.detection import get_detection_service

router = APIRouter(prefix="/detect", tags=["检测"])
engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session


def success_response(data, message="success"):
    return {"code": 200, "message": message, "data": data}


@router.post("/upload")
async def detect_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    anomaly_ratio: float = 2.085,
    session: Session = Depends(get_session),
):
    """上传CSV文件进行异常检测"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "只支持 CSV 格式文件")

    # 保存上传文件
    upload_path = UPLOAD_DIR / file.filename
    content = await file.read()
    async with aiofiles.open(upload_path, 'wb') as f:
        await f.write(content)

    # 验证格式
    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        raise HTTPException(400, f"CSV解析失败: {str(e)}")

    svc = get_detection_service()
    validation = svc.validate_csv(df)
    if not validation["valid"]:
        raise HTTPException(400, validation["error"])

    # 创建任务记录
    task = DetectionTask(
        filename=file.filename,
        anomaly_ratio=anomaly_ratio,
        status="running",
        total_samples=len(df),
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    # 执行检测（同步，小文件）
    try:
        result = svc.detect(df, anomaly_ratio=anomaly_ratio)
        result_path = RESULTS_DIR / f"{task.id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f)

        task.status = "completed"
        task.anomaly_count = result["anomaly_count"]
        task.anomaly_rate = result["anomaly_rate"]
        task.processing_time_ms = result["processing_time_ms"]
        task.result_path = str(result_path)
        task.completed_at = datetime.utcnow()
        session.add(task)
        session.commit()
        session.refresh(task)

        return success_response({
            "task_id": task.id,
            "filename": task.filename,
            "total_samples": result["total_samples"],
            "anomaly_count": result["anomaly_count"],
            "anomaly_rate": result["anomaly_rate"],
            "processing_time_ms": result["processing_time_ms"],
            "threshold": result["threshold"],
            "scores": result["scores"],
            "labels": result["labels"],
            "feature_data": result["feature_data"],
        })
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        session.add(task)
        session.commit()
        raise HTTPException(500, f"检测失败: {str(e)}")


@router.get("/sample")
async def detect_sample():
    """使用内置示例数据进行检测（演示用）"""
    import numpy as np
    sample_path = RURAL_VOLTAGE_DIR / "test.csv"
    if not sample_path.exists():
        raise HTTPException(404, "示例数据文件不存在")

    df = pd.read_csv(sample_path).head(500)  # 取前500行演示
    svc = get_detection_service()
    result = svc.detect(df, anomaly_ratio=2.085)

    return success_response({
        "filename": "示例数据 (RuralVoltage test.csv, 前500行)",
        "total_samples": result["total_samples"],
        "anomaly_count": result["anomaly_count"],
        "anomaly_rate": result["anomaly_rate"],
        "processing_time_ms": result["processing_time_ms"],
        "threshold": result["threshold"],
        "scores": result["scores"],
        "labels": result["labels"],
        "feature_data": result["feature_data"],
    })


@router.get("/history")
async def get_history(
    limit: int = 20,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    """获取检测历史"""
    tasks = session.exec(
        select(DetectionTask)
        .order_by(DetectionTask.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    total = session.exec(select(DetectionTask)).all()

    return success_response({
        "items": [
            {
                "id": t.id,
                "filename": t.filename,
                "model_name": t.model_name,
                "status": t.status,
                "total_samples": t.total_samples,
                "anomaly_count": t.anomaly_count,
                "anomaly_rate": t.anomaly_rate,
                "processing_time_ms": t.processing_time_ms,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ],
        "total": len(total),
        "limit": limit,
        "offset": offset,
    })


@router.get("/{task_id}")
async def get_task_result(task_id: str, session: Session = Depends(get_session)):
    """获取特定检测任务的完整结果"""
    task = session.get(DetectionTask, task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    result_data = None
    if task.result_path and Path(task.result_path).exists():
        with open(task.result_path) as f:
            result_data = json.load(f)

    return success_response({
        "task": {
            "id": task.id,
            "filename": task.filename,
            "status": task.status,
            "anomaly_count": task.anomaly_count,
            "anomaly_rate": task.anomaly_rate,
            "created_at": task.created_at.isoformat() if task.created_at else None,
        },
        "result": result_data,
    })
