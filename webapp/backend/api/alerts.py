from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func
from pydantic import BaseModel

from models.database import Alert
from api.detect import get_session

router = APIRouter()


class AlertStatusUpdate(BaseModel):
    status: str


@router.get("")
def list_alerts(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    severity: str = Query(""),
    status: str = Query(""),
    device_code: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    stmt = select(Alert).order_by(Alert.created_at.desc())
    if start_date:
        stmt = stmt.where(Alert.created_at >= datetime.fromisoformat(start_date))
    if end_date:
        stmt = stmt.where(Alert.created_at <= datetime.fromisoformat(end_date + "T23:59:59"))
    if severity:
        stmt = stmt.where(Alert.severity == severity)
    if status:
        stmt = stmt.where(Alert.status == status)
    if device_code:
        stmt = stmt.where(Alert.device_code.contains(device_code))

    all_items = session.exec(stmt).all()
    total = len(all_items)
    items = all_items[(page - 1) * page_size: page * page_size]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/summary")
def alert_summary(session: Session = Depends(get_session)):
    result = {}
    for sv in ["attention", "warning", "critical"]:
        items = session.exec(
            select(Alert).where(Alert.severity == sv).where(Alert.status != "closed")
        ).all()
        result[sv] = len(items)
    return result


@router.put("/{alert_id}/status")
def update_alert_status(alert_id: str, data: AlertStatusUpdate, session: Session = Depends(get_session)):
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(404, "告警不存在")
    alert.status = data.status
    if data.status == "closed":
        alert.resolved_at = datetime.utcnow()
    session.add(alert)
    session.commit()
    return {"ok": True}
