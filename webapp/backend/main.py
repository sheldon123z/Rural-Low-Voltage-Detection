import sys
from pathlib import Path

# 确保代码路径在 sys.path 中
# main.py 在 webapp/backend/main.py
# .parent = webapp/backend/
# .parent.parent = webapp/
# .parent.parent.parent = Rural-Low-Voltage-Detection/
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent  # Rural-Low-Voltage-Detection/
CODE_DIR = PROJECT_ROOT / "code"

# 注意：backend 目录必须在最前面，否则 demo/core/ 会与 backend/core/ 冲突
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(1, str(CODE_DIR))
sys.path.insert(2, str(CODE_DIR / "demo"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, create_engine, Session

from core.config import DATABASE_URL, CORS_ORIGINS, APP_TITLE, APP_VERSION, API_V1_PREFIX
from models.database import DetectionTask, SystemMetrics, Device, Alert, VoltageReading  # noqa: F401
from api import detect as detect_router
from api import models as models_router
from api import metrics as metrics_router
from api import devices, alerts, history, statistics, dashboard

# 数据库引擎
engine = create_engine(DATABASE_URL, echo=False)

def create_db():
    SQLModel.metadata.create_all(engine)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="农村低压配电网电压异常检测平台 API",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(detect_router.router, prefix=API_V1_PREFIX)
app.include_router(models_router.router, prefix=API_V1_PREFIX)
app.include_router(metrics_router.router, prefix=API_V1_PREFIX)
app.include_router(devices.router, prefix="/api/v1/devices", tags=["设备管理"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["告警管理"])
app.include_router(history.router, prefix="/api/v1/history", tags=["历史数据"])
app.include_router(statistics.router, prefix="/api/v1/statistics", tags=["统计报表"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["监控概览"])

@app.on_event("startup")
async def startup():
    create_db()

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}
