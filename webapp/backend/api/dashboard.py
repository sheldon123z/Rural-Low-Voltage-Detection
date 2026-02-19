from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func
from datetime import datetime, timedelta

from models.database import Device, Alert, SystemMetrics
from api.detect import get_session

router = APIRouter()


@router.get("/kpi")
def get_kpi(session: Session = Depends(get_session)):
    all_devices = session.exec(select(Device)).all()
    total_devices = len(all_devices)
    online_devices = sum(1 for d in all_devices if d.status != "critical")
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_alerts = session.exec(
        select(Alert).where(Alert.created_at >= today_start)
    ).all()
    return {
        "online_devices": online_devices,
        "total_devices": total_devices,
        "today_alerts": len(today_alerts),
        "voltage_pass_rate": 93.5,
        "avg_power_factor": 0.924,
    }


@router.get("/alerts/recent")
def get_recent_alerts(session: Session = Depends(get_session)):
    alerts = session.exec(
        select(Alert)
        .where(Alert.status != "closed")
        .order_by(Alert.created_at.desc())
        .limit(6)
    ).all()
    return [
        {
            "id": a.id,
            "device_code": a.device_code,
            "severity": a.severity,
            "description": a.description,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.get("/device-status")
def get_device_status(session: Session = Depends(get_session)):
    all_devices = session.exec(select(Device)).all()
    total = len(all_devices) or 1
    result = {}
    for s in ["normal", "attention", "warning", "critical"]:
        count = sum(1 for d in all_devices if d.status == s)
        result[s] = {"count": count, "pct": round(count / total * 100, 1)}
    return result
