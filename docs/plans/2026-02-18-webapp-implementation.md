# 农村低电压检测平台 Web 应用实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建完整的前后端 Web 应用，使用训练好的 VoltageTimesNet 模型进行真实的农村低电压异常检测。

**Architecture:** FastAPI 后端直接调用 PyTorch 模型（复用现有 `code/demo/core/inference.py`），React+TypeScript 前端提供现代化仪表板，SQLite 存储检测历史，Nginx/Vite 提供静态服务。系统独立在 `webapp/` 目录，不影响现有 `code/` 目录。

**Tech Stack:** FastAPI, SQLModel, Python 3.10+, React 18, TypeScript, Vite, shadcn/ui, Tailwind CSS, Apache ECharts, TanStack Query, Axios

**项目根路径:** `/Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/`

---

## 前置信息

### 关键路径
- 模型权重: `code/newest_models/best_voltagetimesnet_v2.pth`
- 模型配置: `code/newest_models/best_model_config.json`
- 推理模块: `code/demo/core/inference.py` (VoltageAnomalyDetector 类)
- 数据集: `code/dataset/RuralVoltage/realistic_v2/test.csv` (用于示例数据)
- 现有demo: `code/demo/` (Gradio 版本，仅参考)

### 模型最优参数（来自 best_model_config.json）
```json
{
  "model": "VoltageTimesNet_v2",
  "d_model": 128, "e_layers": 3, "d_ff": 256,
  "seq_len": 50, "top_k": 2, "num_kernels": 8,
  "enc_in": 16, "c_out": 16,
  "F1": 0.8149, "Recall": 0.9110, "Precision": 0.7371, "Accuracy": 0.9393
}
```

### 16维特征列名（RuralVoltage 数据集）
Va, Vb, Vc, Ia, Ib, Ic, P, Q, S, PF, THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance

---

## 任务1：项目脚手架（目录结构 + 配置文件）

**Files:**
- Create: `webapp/backend/requirements.txt`
- Create: `webapp/backend/core/config.py`
- Create: `webapp/backend/main.py`
- Create: `webapp/frontend/package.json`
- Create: `webapp/frontend/vite.config.ts`
- Create: `webapp/README.md`

**Step 1: 创建后端目录结构**
```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection
mkdir -p webapp/backend/{api,services,models,core,uploads,results}
mkdir -p webapp/frontend/src/{pages,components,api,hooks,lib,types}
mkdir -p webapp/frontend/src/components/charts
touch webapp/backend/__init__.py
touch webapp/backend/api/__init__.py
touch webapp/backend/services/__init__.py
touch webapp/backend/models/__init__.py
touch webapp/backend/core/__init__.py
```

**Step 2: 创建后端 requirements.txt**
```
# webapp/backend/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
sqlmodel==0.0.22
python-multipart==0.0.12
aiofiles==24.1.0
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
python-jose[cryptography]==3.3.0
httpx==0.27.0
```

**Step 3: 创建核心配置文件**
```python
# webapp/backend/core/config.py
from pathlib import Path
import json

# 项目根路径
WEBAPP_DIR = Path(__file__).parent.parent
BACKEND_DIR = WEBAPP_DIR
PROJECT_ROOT = WEBAPP_DIR.parent  # Rural-Low-Voltage-Detection/
CODE_DIR = PROJECT_ROOT / "code"

# 模型相关路径
MODEL_DIR = CODE_DIR / "newest_models"
BEST_MODEL_PATH = MODEL_DIR / "best_voltagetimesnet_v2.pth"
BEST_MODEL_CONFIG_PATH = MODEL_DIR / "best_model_config.json"

# 数据集路径
DATASET_DIR = CODE_DIR / "dataset"
RURAL_VOLTAGE_DIR = DATASET_DIR / "RuralVoltage" / "realistic_v2"

# 上传和结果目录
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 数据库
DATABASE_URL = f"sqlite:///{BACKEND_DIR}/detection.db"

# RuralVoltage 特征列
FEATURE_COLUMNS = [
    "Va", "Vb", "Vc", "Ia", "Ib", "Ic",
    "P", "Q", "S", "PF",
    "THD_Va", "THD_Vb", "THD_Vc",
    "Freq", "V_unbalance", "I_unbalance"
]

# 模型性能指标（预计算，用于模型对比页面）
MODEL_METRICS = {
    "VoltageTimesNet": {
        "display_name": "VoltageTimesNet (本论文)",
        "accuracy": 0.9393, "precision": 0.7371,
        "recall": 0.9110, "f1": 0.8149,
        "description": "本论文提出的核心模型，融合预设电气周期与FFT自适应发现，专为农村三相电压设计。",
        "is_primary": True,
        "model_key": "VoltageTimesNet_v2",
        "checkpoint": str(BEST_MODEL_PATH),
    },
    "TimesNet": {
        "display_name": "TimesNet (基线)",
        "accuracy": 0.8584, "precision": 0.5143,
        "recall": 0.7115, "f1": 0.5970,
        "description": "原始 TimesNet，纯 FFT 周期发现，未针对电压数据优化。",
        "is_primary": False,
        "model_key": "TimesNet",
        "checkpoint": None,
    },
    "LSTMAutoEncoder": {
        "display_name": "LSTM AutoEncoder",
        "accuracy": 0.7905, "precision": 0.3654,
        "recall": 0.5712, "f1": 0.4457,
        "description": "基于LSTM的自编码器，传统深度学习基线方法。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
    "IsolationForest": {
        "display_name": "Isolation Forest",
        "accuracy": 0.3474, "precision": 0.3474,
        "recall": 1.0000, "f1": 0.5157,
        "description": "经典无监督异常检测，召回率高但精确率低。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
    "OneClassSVM": {
        "display_name": "One-Class SVM",
        "accuracy": 0.3474, "precision": 0.3474,
        "recall": 1.0000, "f1": 0.5157,
        "description": "支持向量机单类分类，适用于小样本场景。",
        "is_primary": False,
        "model_key": None,
        "checkpoint": None,
    },
}

# API配置
API_V1_PREFIX = "/api/v1"
APP_TITLE = "农村低电压检测平台"
APP_VERSION = "1.0.0"
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
```

**Step 4: 创建数据库模型**
```python
# webapp/backend/models/database.py
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
```

**Step 5: 创建检测服务**
```python
# webapp/backend/services/detection.py
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# 添加项目代码路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Rural-Low-Voltage-Detection/
CODE_DIR = PROJECT_ROOT / "code"
DEMO_DIR = CODE_DIR / "demo"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DEMO_DIR))

from core.config import FEATURE_COLUMNS, BEST_MODEL_PATH, BEST_MODEL_CONFIG_PATH


class DetectionService:
    """异常检测服务，封装推理模块"""

    def __init__(self):
        self._detector = None
        self._loaded_model = None

    def _get_detector(self):
        """懒加载检测器"""
        if self._detector is None:
            from core.inference import VoltageAnomalyDetector
            self._detector = VoltageAnomalyDetector(
                model_name="VoltageTimesNet_v2",
                checkpoint_path=str(BEST_MODEL_PATH),
                device="cpu",
                config_path=str(BEST_MODEL_CONFIG_PATH),
            )
            self._detector.load_model()
            self._loaded_model = "VoltageTimesNet"
        return self._detector

    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证CSV文件格式"""
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            return {"valid": False, "error": f"缺少特征列: {missing}"}
        if len(df) < 50:
            return {"valid": False, "error": f"数据行数不足（最少50行，当前{len(df)}行）"}
        return {"valid": True, "rows": len(df)}

    def detect(
        self,
        df: pd.DataFrame,
        anomaly_ratio: float = 2.085,
    ) -> Dict[str, Any]:
        """执行异常检测"""
        start_time = time.time()

        detector = self._get_detector()
        seq_len = detector.seq_len  # 50

        # 提取特征列并转换为 numpy
        data = df[FEATURE_COLUMNS].values.astype(np.float32)

        # 标准化（z-score per column）
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_normalized = (data - mean) / std

        # 滑动窗口推理
        n_samples = len(data_normalized)
        scores = np.zeros(n_samples)
        counts = np.zeros(n_samples)

        for start in range(0, n_samples - seq_len + 1, seq_len // 2):
            end = start + seq_len
            window = data_normalized[start:end]
            result = detector.predict_with_percentile_threshold(
                window, anomaly_ratio=anomaly_ratio
            )
            window_scores = result["scores"]
            scores[start:end] += window_scores
            counts[start:end] += 1

        # 归一化重叠分数
        mask = counts > 0
        scores[mask] /= counts[mask]

        # 应用阈值
        threshold = np.percentile(scores, 100 - anomaly_ratio)
        labels = (scores > threshold).astype(int)

        elapsed_ms = int((time.time() - start_time) * 1000)
        anomaly_count = int(labels.sum())

        return {
            "total_samples": n_samples,
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / n_samples * 100, 2),
            "threshold": float(threshold),
            "processing_time_ms": elapsed_ms,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "timestamps": list(range(n_samples)),
            "feature_data": {
                col: df[col].tolist() for col in ["Va", "Vb", "Vc", "Freq", "V_unbalance"]
            },
        }


# 全局单例
_detection_service = None

def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
```

**Step 6: 创建 FastAPI 主应用**
```python
# webapp/backend/main.py
import sys
from pathlib import Path

# 确保代码路径在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT.parent / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "demo"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, create_engine, Session

from core.config import DATABASE_URL, CORS_ORIGINS, APP_TITLE, APP_VERSION, API_V1_PREFIX
from models.database import DetectionTask, SystemMetrics
from api import detect as detect_router
from api import models as models_router
from api import metrics as metrics_router

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

@app.on_event("startup")
async def startup():
    create_db()

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}
```

**Step 7: 创建检测 API 路由**
```python
# webapp/backend/api/detect.py
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
```

**Step 8: 创建模型和指标 API 路由**
```python
# webapp/backend/api/models.py
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
```

```python
# webapp/backend/api/metrics.py
from fastapi import APIRouter, Depends
from sqlmodel import Session, select, create_engine, func
from core.config import DATABASE_URL
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

    return success_response({
        "total_detections": total_detections,
        "total_anomalies_found": total_anomalies,
        "avg_processing_time_ms": round(avg_time, 1),
        "model_f1": 0.8149,
        "model_recall": 0.9110,
        "model_precision": 0.7371,
    })
```

**Step 9: 安装后端依赖并测试**
```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp/backend
pip install fastapi uvicorn sqlmodel python-multipart aiofiles pandas httpx
# 验证启动
python -c "from main import app; print('Backend OK')"
```

**Step 10: Commit 后端基础结构**
```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection
git add webapp/backend/ docs/plans/
git commit -m "feat: 添加 FastAPI 后端基础结构（路由、服务、数据库模型）"
```

---

## 任务2：前端项目初始化

**Files:**
- Create: `webapp/frontend/package.json`
- Create: `webapp/frontend/vite.config.ts`
- Create: `webapp/frontend/tailwind.config.js`
- Create: `webapp/frontend/src/App.tsx`
- Create: `webapp/frontend/src/main.tsx`
- Create: `webapp/frontend/index.html`

**Step 1: 初始化 React + TypeScript + Vite 项目**
```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**Step 2: 安装核心依赖**
```bash
cd webapp/frontend
npm install react-router-dom @tanstack/react-query axios
npm install echarts echarts-for-react
npm install lucide-react clsx tailwind-merge class-variance-authority
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Step 3: 安装 shadcn/ui 核心组件**
```bash
# 初始化 shadcn/ui (手动方式，不依赖CLI)
npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu
npm install @radix-ui/react-progress @radix-ui/react-slot
npm install @radix-ui/react-tabs @radix-ui/react-select
npm install @radix-ui/react-tooltip @radix-ui/react-separator
npm install framer-motion
```

**Step 4: 配置 tailwind.config.js**
```js
// webapp/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
}
```

**Step 5: 创建 vite.config.ts（含代理）**
```typescript
// webapp/frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

**Step 6: 创建全局 CSS（含 CSS 变量和电气主题色）**
```css
/* webapp/frontend/src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 210 20% 98%;
    --foreground: 215 25% 15%;
    --card: 0 0% 100%;
    --card-foreground: 215 25% 15%;
    --primary: 213 68% 48%;      /* 电气蓝 #2563eb */
    --primary-foreground: 0 0% 100%;
    --secondary: 210 20% 95%;
    --secondary-foreground: 215 25% 15%;
    --muted: 210 20% 95%;
    --muted-foreground: 215 15% 50%;
    --accent: 142 70% 45%;       /* 正常绿 */
    --accent-foreground: 0 0% 100%;
    --destructive: 0 72% 51%;    /* 告警红 */
    --destructive-foreground: 0 0% 100%;
    --border: 210 20% 88%;
    --input: 210 20% 88%;
    --ring: 213 68% 48%;
    --radius: 0.75rem;
  }
}

body {
  @apply bg-background text-foreground;
  font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

/* 滚动条美化 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { @apply bg-muted; }
::-webkit-scrollbar-thumb { @apply bg-border rounded-full; }
```

---

## 任务3：前端布局与路由

**Files:**
- Create: `webapp/frontend/src/App.tsx`
- Create: `webapp/frontend/src/components/Layout.tsx`
- Create: `webapp/frontend/src/components/Sidebar.tsx`
- Create: `webapp/frontend/src/api/client.ts`
- Create: `webapp/frontend/src/types/index.ts`

**Step 1: 创建 API 客户端**
```typescript
// webapp/frontend/src/api/client.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,  // 120秒超时（大文件检测）
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.detail || err.message || '请求失败'
    return Promise.reject(new Error(msg))
  }
)

export default api

// API 函数
export const detectApi = {
  uploadAndDetect: (file: File, anomalyRatio: number) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/detect/upload?anomaly_ratio=${anomalyRatio}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  detectSample: () => api.get('/detect/sample'),
  getHistory: (limit = 20, offset = 0) =>
    api.get(`/detect/history?limit=${limit}&offset=${offset}`),
  getTaskResult: (id: string) => api.get(`/detect/${id}`),
}

export const modelsApi = {
  listModels: () => api.get('/models'),
  getCurrentModel: () => api.get('/models/current'),
}

export const metricsApi = {
  getMetrics: () => api.get('/metrics'),
}
```

**Step 2: 创建类型定义**
```typescript
// webapp/frontend/src/types/index.ts
export interface DetectionResult {
  task_id?: string
  filename: string
  total_samples: number
  anomaly_count: number
  anomaly_rate: number
  processing_time_ms: number
  threshold: number
  scores: number[]
  labels: number[]
  feature_data: {
    Va: number[]
    Vb: number[]
    Vc: number[]
    Freq: number[]
    V_unbalance: number[]
  }
}

export interface ModelMetrics {
  display_name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  description: string
  is_primary: boolean
}

export interface DetectionTask {
  id: string
  filename: string
  model_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  total_samples?: number
  anomaly_count?: number
  anomaly_rate?: number
  processing_time_ms?: number
  created_at: string
}

export interface SystemMetrics {
  total_detections: number
  total_anomalies_found: number
  avg_processing_time_ms: number
  model_f1: number
  model_recall: number
  model_precision: number
}
```

**Step 3: 创建 Sidebar 导航组件**
```typescript
// webapp/frontend/src/components/Sidebar.tsx
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Zap, History, BarChart3, BookOpen, Activity
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: '总览仪表板' },
  { path: '/detect', icon: Zap, label: '异常检测' },
  { path: '/history', icon: History, label: '检测历史' },
  { path: '/models', icon: BarChart3, label: '模型对比' },
  { path: '/about', icon: BookOpen, label: '系统原理' },
]

export function Sidebar() {
  return (
    <aside className="w-56 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-5 border-b border-border">
        <Activity className="w-6 h-6 text-primary mr-2.5" />
        <div>
          <div className="text-sm font-semibold text-foreground leading-tight">低电压检测</div>
          <div className="text-xs text-muted-foreground">Rural Grid AI</div>
        </div>
      </div>

      {/* 导航 */}
      <nav className="flex-1 py-4 space-y-1 px-3">
        {navItems.map(({ path, icon: Icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )
            }
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* 底部状态 */}
      <div className="p-4 border-t border-border">
        <div className="text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5 mb-1">
            <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span>VoltageTimesNet 已加载</span>
          </div>
          <div className="text-muted-foreground/70">F1 = 0.8149 | Recall = 91.1%</div>
        </div>
      </div>
    </aside>
  )
}
```

**Step 4: 创建 Layout 组件**
```typescript
// webapp/frontend/src/components/Layout.tsx
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
```

**Step 5: 创建 App.tsx 路由配置**
```typescript
// webapp/frontend/src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from '@/components/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Detect } from '@/pages/Detect'
import { History } from '@/pages/History'
import { Models } from '@/pages/Models'
import { About } from '@/pages/About'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5 * 60 * 1000, retry: 1 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/detect" element={<Detect />} />
            <Route path="/history" element={<History />} />
            <Route path="/models" element={<Models />} />
            <Route path="/about" element={<About />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
```

**Step 6: 创建工具函数**
```typescript
// webapp/frontend/src/lib/utils.ts
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number, decimals = 2): string {
  return `${value.toFixed(decimals)}%`
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}

export function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleString('zh-CN', {
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  })
}
```

---

## 任务4：仪表板页面

**Files:**
- Create: `webapp/frontend/src/pages/Dashboard.tsx`
- Create: `webapp/frontend/src/components/MetricCard.tsx`

**Step 1: 创建指标卡片组件**
```typescript
// webapp/frontend/src/components/MetricCard.tsx
import { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  variant?: 'default' | 'primary' | 'success' | 'warning'
}

export function MetricCard({
  title, value, subtitle, icon: Icon, variant = 'default'
}: MetricCardProps) {
  const variantStyles = {
    default: 'bg-card border-border',
    primary: 'bg-primary/5 border-primary/20',
    success: 'bg-green-50 border-green-200',
    warning: 'bg-amber-50 border-amber-200',
  }
  const iconStyles = {
    default: 'text-muted-foreground bg-muted',
    primary: 'text-primary bg-primary/10',
    success: 'text-green-600 bg-green-100',
    warning: 'text-amber-600 bg-amber-100',
  }

  return (
    <div className={cn('rounded-xl border p-5', variantStyles[variant])}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-muted-foreground font-medium">{title}</span>
        <div className={cn('p-2 rounded-lg', iconStyles[variant])}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="text-2xl font-bold text-foreground mb-1">{value}</div>
      {subtitle && <div className="text-xs text-muted-foreground">{subtitle}</div>}
    </div>
  )
}
```

**Step 2: 创建仪表板页面**
```typescript
// webapp/frontend/src/pages/Dashboard.tsx
import { useQuery } from '@tanstack/react-query'
import { Zap, Activity, CheckCircle, TrendingUp } from 'lucide-react'
import ReactECharts from 'echarts-for-react'
import { MetricCard } from '@/components/MetricCard'
import { metricsApi, detectApi } from '@/api/client'
import { formatPercent, formatMs, formatDate } from '@/lib/utils'
import type { SystemMetrics, DetectionTask } from '@/types'

export function Dashboard() {
  const { data: metricsRes } = useQuery({
    queryKey: ['metrics'],
    queryFn: metricsApi.getMetrics,
    refetchInterval: 30000,
  })
  const { data: historyRes } = useQuery({
    queryKey: ['history', 5],
    queryFn: () => detectApi.getHistory(5, 0),
  })

  const metrics: SystemMetrics = metricsRes?.data || {
    total_detections: 0,
    total_anomalies_found: 0,
    avg_processing_time_ms: 0,
    model_f1: 0.8149,
    model_recall: 0.9110,
    model_precision: 0.7371,
  }
  const recentTasks: DetectionTask[] = historyRes?.data?.items || []

  // ECharts 模型性能雷达图配置
  const radarOption = {
    backgroundColor: 'transparent',
    radar: {
      indicator: [
        { name: '准确率', max: 1 },
        { name: '精确率', max: 1 },
        { name: '召回率', max: 1 },
        { name: 'F1分数', max: 1 },
      ],
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: '#64748b', fontSize: 11 },
      splitLine: { lineStyle: { color: ['#e2e8f0'] } },
      splitArea: { areaStyle: { color: ['rgba(37,99,235,0.02)', 'rgba(37,99,235,0.05)'] } },
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: [0.9393, 0.7371, 0.9110, 0.8149],
            name: 'VoltageTimesNet',
            areaStyle: { color: 'rgba(37,99,235,0.15)' },
            lineStyle: { color: '#2563eb', width: 2 },
            itemStyle: { color: '#2563eb' },
          },
          {
            value: [0.8584, 0.5143, 0.7115, 0.5970],
            name: 'TimesNet',
            areaStyle: { color: 'rgba(100,116,139,0.1)' },
            lineStyle: { color: '#94a3b8', width: 1.5, type: 'dashed' },
            itemStyle: { color: '#94a3b8' },
          },
        ],
        tooltip: { trigger: 'item' },
      },
    ],
    legend: {
      data: ['VoltageTimesNet', 'TimesNet'],
      bottom: 0,
      textStyle: { color: '#64748b', fontSize: 11 },
    },
    tooltip: {},
  }

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">系统总览</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          农村低压配电网电压异常检测平台 · VoltageTimesNet
        </p>
      </div>

      {/* 指标卡片 */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="累计检测次数" value={metrics.total_detections}
          subtitle="历史检测任务总数" icon={Activity} variant="primary"
        />
        <MetricCard
          title="发现异常事件" value={metrics.total_anomalies_found}
          subtitle="累计检测到的异常时间步" icon={Zap} variant="warning"
        />
        <MetricCard
          title="模型 F1 分数" value={formatPercent(metrics.model_f1 * 100, 1)}
          subtitle="VoltageTimesNet 论文实验结果" icon={TrendingUp} variant="success"
        />
        <MetricCard
          title="平均检测耗时" value={formatMs(metrics.avg_processing_time_ms || 0)}
          subtitle="每次检测平均响应时间" icon={CheckCircle}
        />
      </div>

      {/* 主内容区 */}
      <div className="grid grid-cols-3 gap-5">
        {/* 模型性能雷达图 */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold text-foreground mb-1">模型性能对比</h2>
          <p className="text-xs text-muted-foreground mb-4">RuralVoltage 数据集实验结果</p>
          <ReactECharts option={radarOption} style={{ height: 220 }} />
        </div>

        {/* 近期检测列表 */}
        <div className="col-span-2 bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold text-foreground mb-1">最近检测记录</h2>
          <p className="text-xs text-muted-foreground mb-4">最近5次检测任务</p>
          {recentTasks.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
              暂无检测记录，前往「异常检测」页面开始检测
            </div>
          ) : (
            <div className="space-y-2">
              {recentTasks.map((task) => (
                <div
                  key={task.id}
                  className="flex items-center justify-between py-2 px-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        task.status === 'completed' ? 'bg-green-500' :
                        task.status === 'failed' ? 'bg-red-500' : 'bg-amber-400 animate-pulse'
                      }`}
                    />
                    <div>
                      <div className="text-sm font-medium text-foreground">{task.filename}</div>
                      <div className="text-xs text-muted-foreground">{formatDate(task.created_at)}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-destructive">
                      {task.anomaly_rate != null ? formatPercent(task.anomaly_rate) : '-'}
                    </div>
                    <div className="text-xs text-muted-foreground">异常率</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 项目信息卡 */}
      <div className="bg-primary/5 border border-primary/20 rounded-xl p-5">
        <div className="flex items-start gap-4">
          <div className="p-2.5 bg-primary/10 rounded-lg flex-shrink-0">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground mb-1">VoltageTimesNet · 论文主模型</h3>
            <p className="text-sm text-muted-foreground">
              融合预设电气周期（工频50Hz的谐波周期）与FFT自适应周期发现，专为农村三相低压配电网设计。
              在 RuralVoltage 数据集上 F1=0.8149，相比 TimesNet 基线提升 36.5%，
              召回率达 91.1%，有效捕获低电压异常事件。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
```

---

## 任务5：异常检测页面（核心功能）

**Files:**
- Create: `webapp/frontend/src/pages/Detect.tsx`
- Create: `webapp/frontend/src/components/charts/TimeSeriesChart.tsx`
- Create: `webapp/frontend/src/components/charts/AnomalyScoreChart.tsx`
- Create: `webapp/frontend/src/components/UploadPanel.tsx`

**Step 1: 创建时序图表组件**
```typescript
// webapp/frontend/src/components/charts/TimeSeriesChart.tsx
import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'

interface TimeSeriesChartProps {
  data: number[]
  labels: number[]
  seriesName: string
  color?: string
  height?: number
}

export function TimeSeriesChart({
  data, labels, seriesName, color = '#2563eb', height = 200
}: TimeSeriesChartProps) {
  const option = useMemo(() => {
    // 找出异常区间（连续的 label=1）
    const markAreas: [{ xAxis: number }, { xAxis: number }][] = []
    let start = -1
    for (let i = 0; i <= labels.length; i++) {
      if (labels[i] === 1 && start === -1) {
        start = i
      } else if ((labels[i] !== 1 || i === labels.length) && start !== -1) {
        markAreas.push([{ xAxis: start }, { xAxis: i - 1 }])
        start = -1
      }
    }

    return {
      backgroundColor: 'transparent',
      grid: { top: 10, right: 15, bottom: 25, left: 50, containLabel: false },
      xAxis: {
        type: 'category',
        data: Array.from({ length: data.length }, (_, i) => i),
        axisLine: { lineStyle: { color: '#e2e8f0' } },
        axisTick: { show: false },
        axisLabel: { color: '#94a3b8', fontSize: 10, interval: Math.floor(data.length / 5) },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      series: [{
        name: seriesName,
        type: 'line',
        data: data,
        lineStyle: { color, width: 1.5 },
        itemStyle: { opacity: 0 },
        emphasis: { itemStyle: { opacity: 1, color } },
        smooth: false,
        markArea: {
          itemStyle: { color: 'rgba(239,68,68,0.12)' },
          data: markAreas,
        },
      }],
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.95)',
        borderColor: '#e2e8f0',
        borderWidth: 1,
        textStyle: { color: '#1e293b', fontSize: 12 },
        formatter: (params: any[]) => {
          const p = params[0]
          const isAnomaly = labels[p.dataIndex] === 1
          return `<div>
            <div style="color:#64748b;margin-bottom:4px">时间步 ${p.dataIndex}</div>
            <div>${seriesName}: <b>${typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</b></div>
            <div style="color:${isAnomaly ? '#ef4444' : '#22c55e'};margin-top:4px">
              ${isAnomaly ? '⚠️ 异常' : '✓ 正常'}
            </div>
          </div>`
        },
      },
    }
  }, [data, labels, seriesName, color])

  return <ReactECharts option={option} style={{ height }} />
}
```

**Step 2: 创建异常分数图表**
```typescript
// webapp/frontend/src/components/charts/AnomalyScoreChart.tsx
import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'

interface AnomalyScoreChartProps {
  scores: number[]
  threshold: number
  height?: number
}

export function AnomalyScoreChart({ scores, threshold, height = 160 }: AnomalyScoreChartProps) {
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    grid: { top: 15, right: 15, bottom: 25, left: 55, containLabel: false },
    xAxis: {
      type: 'category',
      data: Array.from({ length: scores.length }, (_, i) => i),
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: { color: '#94a3b8', fontSize: 10, interval: Math.floor(scores.length / 5) },
    },
    yAxis: {
      type: 'value',
      name: '重构误差',
      nameTextStyle: { color: '#94a3b8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
      axisLabel: { color: '#94a3b8', fontSize: 10 },
    },
    series: [
      {
        type: 'bar',
        data: scores.map((s, i) => ({
          value: s,
          itemStyle: { color: s > threshold ? '#ef4444' : '#2563eb', opacity: 0.75 },
        })),
        barMaxWidth: 3,
      },
      {
        type: 'line',
        data: new Array(scores.length).fill(threshold),
        lineStyle: { color: '#f59e0b', width: 1.5, type: 'dashed' },
        itemStyle: { opacity: 0 },
        name: '检测阈值',
      },
    ],
    tooltip: {
      trigger: 'axis',
      formatter: (params: any[]) => {
        const s = params[0].value
        return `时间步 ${params[0].dataIndex}<br/>异常分数: ${s.toFixed(6)}<br/>${s > threshold ? '⚠️ 超过阈值' : '✓ 正常范围'}`
      },
    },
    legend: {
      data: ['检测阈值'],
      right: 0, top: 0,
      textStyle: { color: '#64748b', fontSize: 10 },
    },
  }), [scores, threshold])

  return <ReactECharts option={option} style={{ height }} />
}
```

**Step 3: 创建检测页面**
```typescript
// webapp/frontend/src/pages/Detect.tsx
import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Upload, Play, FileText, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { AnomalyScoreChart } from '@/components/charts/AnomalyScoreChart'
import { detectApi } from '@/api/client'
import { formatPercent, formatMs } from '@/lib/utils'
import type { DetectionResult } from '@/types'

const FEATURE_COLORS: Record<string, string> = {
  Va: '#2563eb', Vb: '#7c3aed', Vc: '#db2777',
}

export function Detect() {
  const [file, setFile] = useState<File | null>(null)
  const [anomalyRatio, setAnomalyRatio] = useState(2.085)
  const [selectedFeature, setSelectedFeature] = useState<'Va' | 'Vb' | 'Vc'>('Va')
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const detectMutation = useMutation({
    mutationFn: ({ file, ratio }: { file: File; ratio: number }) =>
      detectApi.uploadAndDetect(file, ratio),
    onSuccess: (res: any) => setResult(res.data),
  })

  const sampleMutation = useMutation({
    mutationFn: detectApi.detectSample,
    onSuccess: (res: any) => setResult(res.data),
  })

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.csv')) setFile(f)
  }, [])

  const handleDetect = () => {
    if (!file) return
    detectMutation.mutate({ file, ratio: anomalyRatio })
  }

  const isLoading = detectMutation.isPending || sampleMutation.isPending

  return (
    <div className="p-6 space-y-5">
      <div>
        <h1 className="text-xl font-semibold">异常检测</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          上传电压数据 CSV 或使用内置示例数据，VoltageTimesNet 实时检测异常事件
        </p>
      </div>

      <div className="grid grid-cols-5 gap-5">
        {/* 左侧：控制面板 */}
        <div className="col-span-2 space-y-4">
          {/* 文件上传区 */}
          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Upload className="w-4 h-4" /> 数据上传
            </h2>

            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onClick={() => document.getElementById('csv-input')?.click()}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50 hover:bg-muted/30'
              }`}
            >
              <FileText className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
              {file ? (
                <>
                  <div className="text-sm font-medium text-foreground">{file.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {(file.size / 1024).toFixed(1)} KB
                  </div>
                </>
              ) : (
                <>
                  <div className="text-sm text-muted-foreground">拖放 CSV 或点击选择</div>
                  <div className="text-xs text-muted-foreground mt-1">需含 16 维电压特征列</div>
                </>
              )}
              <input
                id="csv-input" type="file" accept=".csv"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </div>

            {/* 参数设置 */}
            <div className="mt-4 space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-muted-foreground">异常比例阈值</span>
                  <span className="font-medium text-foreground">{anomalyRatio.toFixed(1)}%</span>
                </div>
                <input
                  type="range" min="0.5" max="10" step="0.5"
                  value={anomalyRatio}
                  onChange={(e) => setAnomalyRatio(parseFloat(e.target.value))}
                  className="w-full accent-primary"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.5% (严格)</span><span>5.0% (宽松)</span>
                </div>
              </div>
            </div>

            {/* 操作按钮 */}
            <div className="mt-4 space-y-2">
              <button
                onClick={handleDetect}
                disabled={!file || isLoading}
                className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="w-4 h-4" />
                {detectMutation.isPending ? '检测中...' : '开始检测'}
              </button>
              <button
                onClick={() => sampleMutation.mutate()}
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 py-2 border border-border text-muted-foreground rounded-lg text-sm hover:bg-muted/50 disabled:opacity-50 transition-colors"
              >
                {sampleMutation.isPending ? '加载中...' : '使用内置示例数据'}
              </button>
            </div>

            {/* 错误信息 */}
            {(detectMutation.error || sampleMutation.error) && (
              <div className="mt-3 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
                <span className="text-xs text-destructive">
                  {((detectMutation.error || sampleMutation.error) as Error)?.message}
                </span>
              </div>
            )}
          </div>

          {/* 数据格式说明 */}
          <div className="bg-muted/50 border border-border rounded-xl p-4">
            <h3 className="text-xs font-semibold text-foreground mb-2 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" /> 所需特征列（16维）
            </h3>
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
              {['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic', 'P', 'Q', 'S', 'PF',
                'THD_Va', 'THD_Vb', 'THD_Vc', 'Freq', 'V_unbalance', 'I_unbalance'
              ].map(col => (
                <div key={col} className="text-xs text-muted-foreground font-mono">{col}</div>
              ))}
            </div>
          </div>
        </div>

        {/* 右侧：结果展示 */}
        <div className="col-span-3 space-y-4">
          {!result && !isLoading && (
            <div className="bg-card border border-border rounded-xl flex items-center justify-center h-80">
              <div className="text-center text-muted-foreground">
                <Play className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <div className="text-sm">上传数据后点击「开始检测」</div>
                <div className="text-xs mt-1">或点击「使用内置示例数据」快速体验</div>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="bg-card border border-border rounded-xl flex items-center justify-center h-80">
              <div className="text-center text-muted-foreground">
                <div className="w-10 h-10 border-3 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <div className="text-sm">VoltageTimesNet 推理中...</div>
                <div className="text-xs mt-1">正在检测电压异常</div>
              </div>
            </div>
          )}

          {result && !isLoading && (
            <>
              {/* 检测摘要卡片 */}
              <div className="bg-card border border-border rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" /> 检测完成
                  </h2>
                  <span className="text-xs text-muted-foreground bg-muted px-2.5 py-1 rounded-full">
                    {formatMs(result.processing_time_ms)}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-foreground">{result.total_samples.toLocaleString()}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">总时间步</div>
                  </div>
                  <div className="text-center p-3 bg-destructive/5 rounded-lg">
                    <div className="text-2xl font-bold text-destructive">{result.anomaly_count.toLocaleString()}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">异常时间步</div>
                  </div>
                  <div className="text-center p-3 bg-amber-50 rounded-lg">
                    <div className="text-2xl font-bold text-amber-600">{formatPercent(result.anomaly_rate)}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">异常率</div>
                  </div>
                </div>
              </div>

              {/* 三相电压时序图 */}
              <div className="bg-card border border-border rounded-xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-semibold">电压时序图（红色区域为检测到的异常）</h2>
                  <div className="flex gap-1">
                    {(['Va', 'Vb', 'Vc'] as const).map(f => (
                      <button
                        key={f}
                        onClick={() => setSelectedFeature(f)}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                          selectedFeature === f
                            ? 'text-white'
                            : 'bg-muted text-muted-foreground hover:bg-muted/70'
                        }`}
                        style={selectedFeature === f ? { backgroundColor: FEATURE_COLORS[f] } : {}}
                      >
                        {f}
                      </button>
                    ))}
                  </div>
                </div>
                <TimeSeriesChart
                  data={result.feature_data[selectedFeature]}
                  labels={result.labels}
                  seriesName={selectedFeature}
                  color={FEATURE_COLORS[selectedFeature]}
                  height={180}
                />
              </div>

              {/* 异常分数图 */}
              <div className="bg-card border border-border rounded-xl p-5">
                <h2 className="text-sm font-semibold mb-1">重构误差（异常分数）分布</h2>
                <p className="text-xs text-muted-foreground mb-3">
                  蓝色：正常 · 红色：超过阈值（异常）· 黄色虚线：检测阈值 {result.threshold.toFixed(6)}
                </p>
                <AnomalyScoreChart scores={result.scores} threshold={result.threshold} height={140} />
              </div>

              {/* 频率和不平衡度 */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-card border border-border rounded-xl p-4">
                  <h3 className="text-xs font-semibold mb-2 text-muted-foreground">系统频率 (Freq)</h3>
                  <TimeSeriesChart
                    data={result.feature_data.Freq}
                    labels={result.labels}
                    seriesName="频率/Hz"
                    color="#0891b2"
                    height={120}
                  />
                </div>
                <div className="bg-card border border-border rounded-xl p-4">
                  <h3 className="text-xs font-semibold mb-2 text-muted-foreground">电压不平衡度 (V_unbalance)</h3>
                  <TimeSeriesChart
                    data={result.feature_data.V_unbalance}
                    labels={result.labels}
                    seriesName="不平衡度"
                    color="#d97706"
                    height={120}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
```

---

## 任务6：历史记录页面

**Files:**
- Create: `webapp/frontend/src/pages/History.tsx`

**Step 1: 创建历史记录页面**
```typescript
// webapp/frontend/src/pages/History.tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { History as HistoryIcon, CheckCircle, XCircle, Clock, Search } from 'lucide-react'
import { detectApi } from '@/api/client'
import { formatPercent, formatMs, formatDate } from '@/lib/utils'
import type { DetectionTask } from '@/types'

const STATUS_CONFIG = {
  completed: { label: '完成', icon: CheckCircle, color: 'text-green-600 bg-green-50' },
  failed: { label: '失败', icon: XCircle, color: 'text-red-600 bg-red-50' },
  running: { label: '运行中', icon: Clock, color: 'text-amber-600 bg-amber-50' },
  pending: { label: '等待', icon: Clock, color: 'text-gray-600 bg-gray-50' },
}

export function History() {
  const [page, setPage] = useState(0)
  const pageSize = 20

  const { data, isLoading } = useQuery({
    queryKey: ['history', page],
    queryFn: () => detectApi.getHistory(pageSize, page * pageSize),
  })

  const tasks: DetectionTask[] = data?.data?.items || []
  const total = data?.data?.total || 0

  return (
    <div className="p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <HistoryIcon className="w-5 h-5" /> 检测历史
          </h1>
          <p className="text-sm text-muted-foreground mt-0.5">共 {total} 条检测记录</p>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-muted-foreground text-sm">加载中...</div>
        ) : tasks.length === 0 ? (
          <div className="p-12 text-center text-muted-foreground text-sm">
            暂无检测历史，前往「异常检测」页面进行第一次检测
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-muted/50 border-b border-border">
              <tr>
                {['文件名', '模型', '时间步数', '异常数', '异常率', '耗时', '检测时间', '状态'].map(h => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {tasks.map((task) => {
                const status = STATUS_CONFIG[task.status] || STATUS_CONFIG.pending
                const StatusIcon = status.icon
                return (
                  <tr key={task.id} className="hover:bg-muted/30 transition-colors">
                    <td className="px-4 py-3 font-medium text-foreground max-w-[180px] truncate">
                      {task.filename}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                      {task.model_name}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {task.total_samples?.toLocaleString() || '-'}
                    </td>
                    <td className="px-4 py-3 text-destructive font-medium">
                      {task.anomaly_count?.toLocaleString() || '-'}
                    </td>
                    <td className="px-4 py-3">
                      {task.anomaly_rate != null ? (
                        <span className={`font-semibold ${task.anomaly_rate > 5 ? 'text-destructive' : 'text-amber-600'}`}>
                          {formatPercent(task.anomaly_rate)}
                        </span>
                      ) : '-'}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {task.processing_time_ms ? formatMs(task.processing_time_ms) : '-'}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground text-xs">
                      {formatDate(task.created_at)}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${status.color}`}>
                        <StatusIcon className="w-3 h-3" />
                        {status.label}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* 分页 */}
      {total > pageSize && (
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-50 hover:bg-muted transition-colors"
          >
            上一页
          </button>
          <span className="text-sm text-muted-foreground">
            第 {page + 1} / {Math.ceil(total / pageSize)} 页
          </span>
          <button
            onClick={() => setPage(p => p + 1)}
            disabled={(page + 1) * pageSize >= total}
            className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-50 hover:bg-muted transition-colors"
          >
            下一页
          </button>
        </div>
      )}
    </div>
  )
}
```

---

## 任务7：模型对比页面

**Files:**
- Create: `webapp/frontend/src/pages/Models.tsx`
- Create: `webapp/frontend/src/components/charts/ModelRadarChart.tsx`

**Step 1: 创建雷达图组件**
```typescript
// webapp/frontend/src/components/charts/ModelRadarChart.tsx
import ReactECharts from 'echarts-for-react'

interface ModelData {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  is_primary: boolean
}

interface ModelRadarChartProps {
  models: ModelData[]
  height?: number
}

const MODEL_COLORS = [
  '#2563eb', '#94a3b8', '#f59e0b', '#10b981', '#ef4444'
]

export function ModelRadarChart({ models, height = 350 }: ModelRadarChartProps) {
  const option = {
    backgroundColor: 'transparent',
    legend: {
      data: models.map(m => m.name),
      bottom: 0,
      textStyle: { color: '#64748b', fontSize: 11 },
    },
    radar: {
      indicator: [
        { name: '准确率', max: 1, min: 0 },
        { name: '精确率', max: 1, min: 0 },
        { name: '召回率', max: 1, min: 0 },
        { name: 'F1分数', max: 1, min: 0 },
      ],
      shape: 'polygon',
      splitNumber: 5,
      center: ['50%', '48%'],
      radius: '60%',
      axisName: {
        color: '#475569',
        fontSize: 12,
        fontFamily: 'system-ui',
      },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      splitArea: {
        areaStyle: {
          color: ['rgba(248,250,252,0.5)', 'rgba(241,245,249,0.5)'],
        },
      },
      axisLine: { lineStyle: { color: '#e2e8f0' } },
    },
    series: [{
      type: 'radar',
      data: models.map((m, i) => ({
        name: m.name,
        value: [m.accuracy, m.precision, m.recall, m.f1],
        areaStyle: {
          color: `${MODEL_COLORS[i] || '#64748b'}${m.is_primary ? '25' : '10'}`,
        },
        lineStyle: {
          color: MODEL_COLORS[i] || '#64748b',
          width: m.is_primary ? 2.5 : 1.5,
          type: m.is_primary ? 'solid' : 'dashed',
        },
        itemStyle: { color: MODEL_COLORS[i] || '#64748b' },
        symbol: m.is_primary ? 'circle' : 'emptyCircle',
        symbolSize: m.is_primary ? 6 : 4,
      })),
    }],
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const [acc, prec, rec, f1] = params.value
        return `<div style="font-family:system-ui;padding:4px">
          <div style="font-weight:600;margin-bottom:6px">${params.name}</div>
          <div>准确率: <b>${(acc*100).toFixed(1)}%</b></div>
          <div>精确率: <b>${(prec*100).toFixed(1)}%</b></div>
          <div>召回率: <b>${(rec*100).toFixed(1)}%</b></div>
          <div>F1分数: <b>${(f1*100).toFixed(1)}%</b></div>
        </div>`
      },
    },
  }

  return <ReactECharts option={option} style={{ height }} />
}
```

**Step 2: 创建模型对比页面**
```typescript
// webapp/frontend/src/pages/Models.tsx
import { useQuery } from '@tanstack/react-query'
import { Star, TrendingUp, Target, Activity } from 'lucide-react'
import { ModelRadarChart } from '@/components/charts/ModelRadarChart'
import { modelsApi } from '@/api/client'
import { formatPercent } from '@/lib/utils'
import type { ModelMetrics } from '@/types'
import ReactECharts from 'echarts-for-react'

const METRIC_COLORS = ['#2563eb', '#94a3b8', '#f59e0b', '#10b981', '#ef4444']

export function Models() {
  const { data } = useQuery({
    queryKey: ['models'],
    queryFn: modelsApi.listModels,
  })
  const models: ModelMetrics[] = data?.data || []

  // F1分数柱状图
  const barOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 20, bottom: 80, left: 20, containLabel: true },
    xAxis: {
      type: 'category',
      data: models.map(m => m.display_name.split('(')[0].trim()),
      axisLabel: { color: '#64748b', fontSize: 11, interval: 0, rotate: 20 },
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      min: 0, max: 1,
      axisLabel: { color: '#64748b', fontSize: 11, formatter: (v: number) => `${(v*100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
    },
    series: [
      {
        type: 'bar',
        data: models.map((m, i) => ({
          value: m.f1,
          itemStyle: {
            color: METRIC_COLORS[i] || '#64748b',
            borderRadius: [6, 6, 0, 0],
            opacity: m.is_primary ? 1 : 0.7,
          },
          label: {
            show: true,
            position: 'top',
            formatter: `{c}`,
            color: '#475569',
            fontSize: 11,
            fontWeight: m.is_primary ? 700 : 400,
          },
        })),
        barMaxWidth: 48,
        name: 'F1 分数',
      },
    ],
    tooltip: {
      trigger: 'axis',
      formatter: (params: any[]) => {
        const m = models[params[0].dataIndex]
        return m ? `<b>${m.display_name}</b><br/>F1: ${(m.f1*100).toFixed(1)}%<br/>精确率: ${(m.precision*100).toFixed(1)}%<br/>召回率: ${(m.recall*100).toFixed(1)}%` : ''
      },
    },
  }

  return (
    <div className="p-6 space-y-5">
      <div>
        <h1 className="text-xl font-semibold">模型性能对比</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          RuralVoltage 数据集实验结果，14.6% 异常率
        </p>
      </div>

      {/* 模型卡片 */}
      <div className="grid grid-cols-5 gap-3">
        {models.map((model, i) => (
          <div
            key={model.display_name}
            className={`bg-card rounded-xl border p-4 ${
              model.is_primary ? 'border-primary/30 ring-1 ring-primary/20' : 'border-border'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: METRIC_COLORS[i] }}
              />
              {model.is_primary && (
                <span className="text-xs text-primary font-medium flex items-center gap-0.5">
                  <Star className="w-3 h-3" /> 论文主模型
                </span>
              )}
            </div>
            <div className="text-xs font-semibold text-foreground mb-2 leading-tight">
              {model.display_name}
            </div>
            <div className="space-y-1.5">
              {[
                { label: 'F1', value: model.f1, highlight: model.is_primary },
                { label: '精确率', value: model.precision },
                { label: '召回率', value: model.recall },
                { label: '准确率', value: model.accuracy },
              ].map(({ label, value, highlight }) => (
                <div key={label} className="flex justify-between items-center">
                  <span className="text-xs text-muted-foreground">{label}</span>
                  <span className={`text-xs font-semibold ${highlight ? 'text-primary' : 'text-foreground'}`}>
                    {formatPercent(value * 100, 1)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 图表区 */}
      <div className="grid grid-cols-2 gap-5">
        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold mb-0.5">F1 分数对比</h2>
          <p className="text-xs text-muted-foreground mb-4">VoltageTimesNet 领先基线模型 36.5%</p>
          <ReactECharts option={barOption} style={{ height: 220 }} />
        </div>
        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold mb-0.5">多维度雷达图</h2>
          <p className="text-xs text-muted-foreground mb-2">准确率、精确率、召回率、F1 四维对比</p>
          <ModelRadarChart
            models={models.map(m => ({
              name: m.display_name.split('(')[0].trim(),
              accuracy: m.accuracy,
              precision: m.precision,
              recall: m.recall,
              f1: m.f1,
              is_primary: m.is_primary,
            }))}
            height={280}
          />
        </div>
      </div>

      {/* 模型说明 */}
      <div className="space-y-3">
        {models.map((model, i) => (
          <div
            key={model.display_name}
            className={`bg-card border rounded-xl p-4 flex items-start gap-3 ${
              model.is_primary ? 'border-primary/20 bg-primary/2' : 'border-border'
            }`}
          >
            <span
              className="w-2.5 h-2.5 rounded-full mt-1.5 flex-shrink-0"
              style={{ backgroundColor: METRIC_COLORS[i] }}
            />
            <div>
              <div className="text-sm font-semibold text-foreground">{model.display_name}</div>
              <div className="text-sm text-muted-foreground mt-0.5">{model.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

---

## 任务8：系统原理页面

**Files:**
- Create: `webapp/frontend/src/pages/About.tsx`

**Step 1: 创建系统原理页面（含FFT可视化）**
```typescript
// webapp/frontend/src/pages/About.tsx
import { useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import { BookOpen, Cpu, Database, GitBranch } from 'lucide-react'

// 生成示例FFT数据
function generateFFTData() {
  const N = 256
  const t = Array.from({ length: N }, (_, i) => i)
  // 模拟三相电压信号（50Hz基波 + 谐波）
  const signal = t.map(i => (
    Math.cos(2 * Math.PI * 50 / 256 * i) +          // 50Hz基波
    0.3 * Math.cos(2 * Math.PI * 150 / 256 * i) +   // 3次谐波
    0.15 * Math.cos(2 * Math.PI * 250 / 256 * i) +  // 5次谐波
    0.05 * (Math.random() - 0.5)                     // 噪声
  ))
  // FFT频谱（简化）
  const freqs = Array.from({ length: N / 2 }, (_, i) => i)
  const spectrum = freqs.map(i => {
    if (Math.abs(i - 50) < 2) return 1.0
    if (Math.abs(i - 150) < 2) return 0.3
    if (Math.abs(i - 250) < 2) return 0.15
    return Math.random() * 0.02
  })
  return { t, signal, freqs, spectrum }
}

export function About() {
  const [fftData] = useState(generateFFTData)

  const signalOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 10, bottom: 30, left: 40 },
    xAxis: { type: 'category', data: fftData.t, axisLabel: { color: '#94a3b8', fontSize: 10 }, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisTick: { show: false } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{ type: 'line', data: fftData.signal, lineStyle: { color: '#2563eb', width: 1 }, itemStyle: { opacity: 0 }, smooth: false }],
    tooltip: { show: false },
  }

  const fftOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 10, bottom: 30, left: 50 },
    xAxis: { type: 'value', name: '频率 (Hz)', nameTextStyle: { color: '#94a3b8', fontSize: 10 }, max: 300, axisLabel: { color: '#94a3b8', fontSize: 10 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
    yAxis: { type: 'value', name: '幅度', nameTextStyle: { color: '#94a3b8', fontSize: 10 }, axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'bar',
      data: fftData.freqs.map((f, i) => [f, fftData.spectrum[i]]),
      itemStyle: {
        color: (params: any) => {
          const freq = params.value[0]
          if (Math.abs(freq - 50) < 2) return '#2563eb'
          if (Math.abs(freq - 150) < 2) return '#7c3aed'
          if (Math.abs(freq - 250) < 2) return '#db2777'
          return '#e2e8f0'
        },
      },
      barMaxWidth: 3,
    }],
    tooltip: { formatter: (p: any) => `${p.value[0].toFixed(0)}Hz: ${p.value[1].toFixed(3)}` },
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <BookOpen className="w-5 h-5" /> 系统原理
        </h1>
        <p className="text-sm text-muted-foreground mt-0.5">VoltageTimesNet 算法原理与数据集说明</p>
      </div>

      {/* FFT 周期发现可视化 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-1 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" /> TimesNet 核心：FFT 周期发现
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          TimesNet 通过 FFT 发现时间序列中的主要周期，将 1D 时序转化为 2D 时间-周期结构，
          再用 2D 卷积提取跨周期特征，最后加权融合重构原始序列。重构误差超过阈值即判定异常。
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">原始电压信号（含谐波）</div>
            <ReactECharts option={signalOption} style={{ height: 140 }} />
          </div>
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">
              FFT 频谱（蓝:50Hz, 紫:150Hz, 粉:250Hz）
            </div>
            <ReactECharts option={fftOption} style={{ height: 140 }} />
          </div>
        </div>
      </div>

      {/* VoltageTimesNet 创新点 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-primary" /> VoltageTimesNet 核心创新
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {[
            {
              title: '预设电气周期',
              desc: '融入电力系统领域知识：工频50Hz基波周期及其整数倍谐波周期，避免FFT在噪声环境中误判主周期。',
              color: 'bg-blue-50 border-blue-200',
            },
            {
              title: '可学习周期权重',
              desc: '自动学习预设周期（30%）与FFT发现周期（70%）的最优混合比例，适应不同农村配电网场景。',
              color: 'bg-purple-50 border-purple-200',
            },
            {
              title: '异常放大器',
              desc: '在重构误差层面增加异常放大机制，提升对低幅度电压异常（如轻微低电压）的检测灵敏度。',
              color: 'bg-pink-50 border-pink-200',
            },
          ].map(({ title, desc, color }) => (
            <div key={title} className={`rounded-lg border p-4 ${color}`}>
              <div className="text-sm font-semibold text-foreground mb-2">{title}</div>
              <div className="text-xs text-muted-foreground leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 数据集说明 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Database className="w-4 h-4 text-primary" /> RuralVoltage 数据集
        </h2>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <table className="w-full text-xs">
              <tbody className="divide-y divide-border">
                {[
                  ['数据集名称', 'RuralVoltage (realistic_v2)'],
                  ['总样本数', '60,000 (训练50k + 测试10k)'],
                  ['特征维度', '16维三相电气量'],
                  ['异常率', '14.6%（实际农村配电网仿真）'],
                  ['采样频率', '每15分钟一个时间步'],
                  ['模型序列长度', '50个时间步（约12.5小时）'],
                ].map(([label, value]) => (
                  <tr key={label}>
                    <td className="py-2 pr-4 text-muted-foreground font-medium">{label}</td>
                    <td className="py-2 text-foreground">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <div className="text-xs font-semibold text-muted-foreground mb-2">16维特征说明</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              {[
                ['Va, Vb, Vc', 'A/B/C相电压'],
                ['Ia, Ib, Ic', 'A/B/C相电流'],
                ['P, Q, S', '有功/无功/视在功率'],
                ['PF', '功率因数'],
                ['THD_Va/Vb/Vc', '三相总谐波畸变率'],
                ['Freq', '系统频率'],
                ['V_unbalance', '电压不平衡度'],
                ['I_unbalance', '电流不平衡度'],
              ].map(([name, desc]) => (
                <div key={name}>
                  <span className="font-mono text-foreground">{name}</span>
                  <span className="text-muted-foreground ml-1">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 项目信息 */}
      <div className="bg-muted/50 border border-border rounded-xl p-4 text-xs text-muted-foreground">
        <div className="font-semibold text-foreground mb-1">项目说明</div>
        本系统基于研究生论文《基于 TimesNet 的农村低压配电网电压异常检测方法研究》开发。
        核心模型 VoltageTimesNet 经过 Optuna 30次超参数优化，在 RuralVoltage 数据集上
        实现 F1=0.8149，召回率 91.1%，有效检测欠压、过压、三相不平衡等农村电网常见故障。
      </div>
    </div>
  )
}
```

---

## 任务9：启动脚本 + README

**Files:**
- Create: `webapp/start.sh`
- Create: `webapp/README.md`
- Create: `webapp/frontend/src/main.tsx`（入口文件）
- Create: `webapp/frontend/index.html`

**Step 1: 创建 main.tsx**
```typescript
// webapp/frontend/src/main.tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

**Step 2: 创建 index.html**
```html
<!-- webapp/frontend/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>农村低电压检测平台</title>
    <meta name="description" content="基于VoltageTimesNet的农村低压配电网电压异常检测系统" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**Step 3: 创建启动脚本**
```bash
#!/bin/bash
# webapp/start.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 农村低电压检测平台启动 ==="
echo "后端: http://localhost:8000"
echo "前端: http://localhost:5173"
echo ""

# 启动后端
echo "[1/2] 启动 FastAPI 后端..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# 等待后端就绪
sleep 3

# 启动前端
echo "[2/2] 启动 React 前端..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 系统启动完成"
echo "   前端界面: http://localhost:5173"
echo "   API文档:  http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务..."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '已停止'" INT
wait
```

**Step 4: 创建 README**
```markdown
# 农村低电压检测平台 - Web 应用

## 快速启动

### 方式一：一键启动
```bash
cd webapp
chmod +x start.sh
./start.sh
```

### 方式二：分别启动
```bash
# 后端（终端1）
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 前端（终端2）
cd webapp/frontend
npm install
npm run dev
```

## 访问地址
- 前端界面: http://localhost:5173
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health
```

**Step 5: 最终测试**
```bash
# 测试后端启动
cd webapp/backend
python -c "
import sys
sys.path.insert(0, '../../../code')
sys.path.insert(0, '../../../code/demo')
from main import app
print('Backend imports OK')
"

# 测试前端构建
cd webapp/frontend
npm run build
echo "Frontend build OK"
```

**Step 6: 最终 Commit**
```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection
git add webapp/
git commit -m "feat: 完整的低电压检测 Web 平台（FastAPI + React）"
```

---

## 总结

### 功能清单
- [x] 仪表板：指标卡片、雷达图、最近检测记录
- [x] 异常检测：CSV上传、示例数据、结果时序图、异常分数图
- [x] 历史记录：分页表格、状态显示
- [x] 模型对比：5模型对比卡片、F1柱状图、雷达图
- [x] 系统原理：FFT可视化、创新点说明、数据集信息

### 技术验证点
- VoltageAnomalyDetector 推理模块复用 ✓
- SQLite 检测历史持久化 ✓
- ECharts 时序图 + 异常区间标注 ✓
- shadcn/ui + Tailwind CSS 现代 UI ✓
- React Query 服务端状态管理 ✓
