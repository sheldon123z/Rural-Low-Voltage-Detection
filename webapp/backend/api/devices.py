from typing import Optional
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func
from pydantic import BaseModel

from models.database import Device
from api.detect import get_session

router = APIRouter()


class DeviceCreate(BaseModel):
    device_code: str
    name: str
    model: str
    region: str
    status: str = "normal"
    rated_voltage: float = 220.0
    capacity: float = 100.0
    responsible: str
    address: str = ""


class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    rated_voltage: Optional[float] = None
    capacity: Optional[float] = None
    responsible: Optional[str] = None
    address: Optional[str] = None


@router.get("")
def list_devices(
    keyword: str = Query(""),
    status: str = Query(""),
    region: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    stmt = select(Device)
    if keyword:
        stmt = stmt.where(
            (Device.device_code.contains(keyword)) | (Device.name.contains(keyword))
        )
    if status:
        stmt = stmt.where(Device.status == status)
    if region:
        stmt = stmt.where(Device.region == region)

    all_items = session.exec(stmt).all()
    total = len(all_items)
    items = all_items[(page - 1) * page_size: page * page_size]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.post("")
def create_device(data: DeviceCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(Device).where(Device.device_code == data.device_code)).first()
    if existing:
        raise HTTPException(400, "设备编号已存在")
    device = Device(id=str(uuid.uuid4()), **data.model_dump())
    session.add(device)
    session.commit()
    session.refresh(device)
    return device


@router.put("/{device_id}")
def update_device(device_id: str, data: DeviceUpdate, session: Session = Depends(get_session)):
    device = session.get(Device, device_id)
    if not device:
        raise HTTPException(404, "设备不存在")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(device, k, v)
    session.add(device)
    session.commit()
    session.refresh(device)
    return device


@router.delete("/{device_id}")
def delete_device(device_id: str, session: Session = Depends(get_session)):
    device = session.get(Device, device_id)
    if not device:
        raise HTTPException(404, "设备不存在")
    session.delete(device)
    session.commit()
    return {"ok": True}
