from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid
import json


class DetectionTask(SQLModel, table=True):
    """检测任务记录"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    filename: str = Field(index=True)
    model_name: str = Field(default="VoltageTimesNet")
    status: str = Field(default="pending")  # pending/running/completed/failed
    anomaly_ratio: float = Field(default=2.085)
    total_samples: Optional[int] = None
    anomaly_count: Optional[int] = None
    anomaly_rate: Optional[float] = None
    f1_reference: Optional[float] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class SystemMetrics(SQLModel, table=True):
    """系统统计指标（单行记录）"""
    id: int = Field(default=1, primary_key=True)
    total_detections: int = Field(default=0)
    total_anomalies_found: int = Field(default=0)
    avg_processing_time_ms: float = Field(default=0.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Device(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_code: str = Field(index=True, unique=True)
    name: str
    model: str
    region: str
    status: str = "normal"  # normal/attention/warning/critical
    rated_voltage: float = 220.0
    capacity: float = 100.0
    responsible: str
    address: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_code: str = Field(index=True)
    device_name: str
    location: str
    alert_type: str  # voltage_low/voltage_high/unbalance/harmonic/frequency
    severity: str    # attention/warning/critical
    description: str
    status: str = "pending"  # pending/processing/closed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


class VoltageReading(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    device_code: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    va: float
    vb: float
    vc: float
    ia: float
    ib: float
    ic: float
    power_factor: float
    is_anomaly: bool = False
