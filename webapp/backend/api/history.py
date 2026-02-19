from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from models.database import VoltageReading, Device
from api.detect import get_session

router = APIRouter()


@router.get("/devices")
def list_devices_for_history(session: Session = Depends(get_session)):
    devices = session.exec(select(Device).limit(10)).all()
    return [{"code": d.device_code, "name": d.name, "region": d.region} for d in devices]


@router.get("/voltage")
def get_voltage_history(
    device_code: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    if not start:
        start = (datetime.utcnow() - timedelta(days=1)).isoformat()
    if not end:
        end = datetime.utcnow().isoformat()

    readings = session.exec(
        select(VoltageReading)
        .where(VoltageReading.device_code == device_code)
        .where(VoltageReading.timestamp >= datetime.fromisoformat(start))
        .where(VoltageReading.timestamp <= datetime.fromisoformat(end))
        .order_by(VoltageReading.timestamp)
        .limit(500)
    ).all()

    return {
        "device_code": device_code,
        "count": len(readings),
        "data": [
            {
                "timestamp": r.timestamp.isoformat(),
                "va": r.va, "vb": r.vb, "vc": r.vc,
                "ia": r.ia, "ib": r.ib, "ic": r.ic,
                "power_factor": r.power_factor,
                "is_anomaly": r.is_anomaly,
            }
            for r in readings
        ],
    }
