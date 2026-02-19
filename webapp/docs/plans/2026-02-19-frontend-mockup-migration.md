# 前端 Mockup 迁移实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将论文 Mockup（Vue 3 + Element Plus）的全部页面设计迁移到真实系统（React + FastAPI）中，同时保留 ML 推理功能。

**Architecture:** 前端使用 React 19 + Ant Design 5.x 替换 Tailwind/Radix UI，后端 FastAPI 新增设备/告警/历史/统计 API，新增 SQLite 表结构并用种子数据填充，所有新页面通过 TanStack Query 与后端对接。

**Tech Stack:** React 19, TypeScript, Ant Design 5.x, ECharts, FastAPI, SQLModel, SQLite

---

## 关键路径速览

```
Task 1: 安装 Ant Design，配置主题
Task 2: 新建主布局（深色侧边栏）
Task 3: 后端 - 新增数据库表和种子数据
Task 4: 后端 - 设备管理 API
Task 5: 后端 - 告警管理 API
Task 6: 后端 - 历史查询 API
Task 7: 后端 - 统计/概览 API
Task 8: 前端 - 登录页
Task 9: 前端 - 监控概览（Dashboard）
Task 10: 前端 - 设备管理页
Task 11: 前端 - 告警管理页
Task 12: 前端 - 历史查询页
Task 13: 前端 - 统计报表页
Task 14: 前端 - 异常检测页（样式迁移）
Task 15: 前端 - 模型对比页（样式迁移）
Task 16: API 客户端扩展 + 整体联调
```

---

## Task 1: 安装 Ant Design，配置主题

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/main.tsx`
- Create: `frontend/src/styles/theme.ts`
- Create: `frontend/src/styles/global.css`
- Modify: `frontend/vite.config.ts`

**Step 1: 安装 Ant Design 依赖**

```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp/frontend
npm install antd @ant-design/icons @ant-design/cssinjs
npm uninstall tailwindcss @tailwindcss/vite autoprefixer
```

**Step 2: 创建主题配置文件 `frontend/src/styles/theme.ts`**

```typescript
import type { ThemeConfig } from 'antd'

export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: '#1677ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#ff4d4f',
    borderRadius: 8,
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif",
  },
  components: {
    Layout: {
      siderBg: '#001529',
      headerBg: '#ffffff',
    },
    Menu: {
      darkItemBg: '#001529',
      darkSubMenuItemBg: '#000c17',
    },
  },
}
```

**Step 3: 创建全局样式 `frontend/src/styles/global.css`**

```css
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #f0f2f5;
  -webkit-font-smoothing: antialiased;
}

/* 卡片通用样式 */
.stat-card {
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
  padding: 20px 24px;
  transition: all 0.25s ease;
}

.stat-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

/* KPI 卡片图标区 */
.kpi-icon-wrap {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

/* 告警摘要卡片色边框 */
.alert-summary-card {
  border-radius: 6px;
  border-left: 4px solid;
  background: #fff;
  padding: 20px 24px;
  transition: all 0.25s ease;
}

.alert-summary-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

/* 表格通用 */
.ant-table-thead > tr > th {
  background: #f5f7fa !important;
  font-weight: 600;
  font-size: 13px;
}

.ant-table-tbody > tr:hover > td {
  background: #ecf5ff !important;
}

/* 图表容器 */
.chart-container {
  border-radius: 8px;
  background: #fff;
  padding: 16px 20px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}

.chart-title {
  font-size: 15px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f5f5f5;
}
```

**Step 4: 修改 `frontend/src/main.tsx`，引入 Ant Design**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router'
import App from './App'
import { antdTheme } from './styles/theme'
import './styles/global.css'
import 'antd/dist/reset.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5 * 60 * 1000, retry: 1 },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ConfigProvider theme={antdTheme} locale={zhCN}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ConfigProvider>
    </QueryClientProvider>
  </React.StrictMode>
)
```

**Step 5: 修改 `frontend/vite.config.ts`，移除 tailwindcss**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
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

**Step 6: 验证启动无报错**

```bash
cd frontend && npm run dev
```

Expected: 浏览器打开 http://localhost:5173，无 Tailwind 报错

**Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: install antd and configure theme, remove tailwind"
```

---

## Task 2: 新建主布局（深色侧边栏）

**Files:**
- Create: `frontend/src/components/MainLayout.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: 创建 `frontend/src/components/MainLayout.tsx`**

```tsx
import { useState, useEffect } from 'react'
import { Layout, Menu, Dropdown, Badge, Avatar, Breadcrumb } from 'antd'
import {
  DashboardOutlined, AlertOutlined, HistoryOutlined,
  BarChartOutlined, ExperimentOutlined, LineChartOutlined,
  ThunderboltOutlined, MenuFoldOutlined, MenuUnfoldOutlined,
  BellOutlined, UserOutlined, LogoutOutlined, AppstoreOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation, Outlet } from 'react-router'

const { Sider, Header, Content } = Layout

const menuItems = [
  {
    key: '/', icon: <DashboardOutlined />, label: '监控概览',
  },
  {
    key: 'devices-group', icon: <AppstoreOutlined />, label: '设备管理',
    children: [{ key: '/devices', label: '设备列表' }],
  },
  {
    key: 'anomaly-group', icon: <AlertOutlined />, label: '异常分析',
    children: [
      { key: '/alerts', label: '告警管理' },
      { key: '/detect', label: '异常检测' },
    ],
  },
  {
    key: 'data-group', icon: <HistoryOutlined />, label: '数据中心',
    children: [
      { key: '/history', label: '历史查询' },
      { key: '/statistics', label: '统计报表' },
    ],
  },
  {
    key: 'model-group', icon: <LineChartOutlined />, label: '模型中心',
    children: [{ key: '/models', label: '模型对比' }],
  },
]

const breadcrumbMap: Record<string, string> = {
  '/': '监控概览',
  '/devices': '设备管理 / 设备列表',
  '/alerts': '异常分析 / 告警管理',
  '/detect': '异常分析 / 异常检测',
  '/history': '数据中心 / 历史查询',
  '/statistics': '数据中心 / 统计报表',
  '/models': '模型中心 / 模型对比',
}

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [time, setTime] = useState(new Date())
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const currentPath = location.pathname
  const breadcrumb = breadcrumbMap[currentPath] || currentPath

  const userMenu = {
    items: [
      { key: 'profile', icon: <UserOutlined />, label: '个人信息' },
      { type: 'divider' as const },
      {
        key: 'logout', icon: <LogoutOutlined />, label: '退出登录',
        onClick: () => { localStorage.removeItem('token'); navigate('/login') },
      },
    ],
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        trigger={null}
        width={220}
        style={{ background: '#001529', position: 'fixed', left: 0, top: 0, bottom: 0, zIndex: 100 }}
      >
        <div style={{
          height: 56, display: 'flex', alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start',
          padding: collapsed ? 0 : '0 20px', gap: 10,
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          <ThunderboltOutlined style={{ color: '#1677ff', fontSize: 22 }} />
          {!collapsed && (
            <span style={{ color: '#fff', fontWeight: 700, fontSize: 14, whiteSpace: 'nowrap' }}>
              低电压监管平台
            </span>
          )}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[currentPath]}
          defaultOpenKeys={['devices-group', 'anomaly-group', 'data-group', 'model-group']}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ background: '#001529', borderRight: 'none', marginTop: 8 }}
        />
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 220, transition: 'margin-left 0.2s' }}>
        <Header style={{
          background: '#fff', height: 56, padding: '0 16px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          boxShadow: '0 1px 4px rgba(0,0,0,0.08)', position: 'sticky', top: 0, zIndex: 99,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 18, cursor: 'pointer', color: '#6b7280' }}
            >
              {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </span>
            <Breadcrumb items={breadcrumb.split(' / ').map(b => ({ title: b }))} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <span style={{ fontSize: 13, color: '#6b7280' }}>
              {time.toLocaleString('zh-CN')}
            </span>
            <Badge count={5} size="small">
              <BellOutlined style={{ fontSize: 18, color: '#6b7280', cursor: 'pointer' }} />
            </Badge>
            <Dropdown menu={userMenu} placement="bottomRight">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <Avatar size={32} style={{ background: '#1677ff' }} icon={<UserOutlined />} />
                <span style={{ fontSize: 13, color: '#1f2937' }}>管理员</span>
              </div>
            </Dropdown>
          </div>
        </Header>

        <Content style={{ padding: 16, minHeight: 'calc(100vh - 56px)' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
```

**Step 2: 修改 `frontend/src/App.tsx`**

```tsx
import { Routes, Route, Navigate } from 'react-router'
import MainLayout from './components/MainLayout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Devices from './pages/Devices'
import Alerts from './pages/Alerts'
import Detect from './pages/Detect'
import History from './pages/History'
import Statistics from './pages/Statistics'
import Models from './pages/Models'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('token')
  return token ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<PrivateRoute><MainLayout /></PrivateRoute>}>
        <Route index element={<Dashboard />} />
        <Route path="devices" element={<Devices />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="detect" element={<Detect />} />
        <Route path="history" element={<History />} />
        <Route path="statistics" element={<Statistics />} />
        <Route path="models" element={<Models />} />
      </Route>
    </Routes>
  )
}
```

**Step 3: 验证布局渲染**

访问 http://localhost:5173，登录后应看到深色侧边栏和顶部 Header

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add main layout with dark sidebar and header"
```

---

## Task 3: 后端 - 新增数据库表和种子数据

**Files:**
- Modify: `backend/models/database.py`
- Create: `backend/scripts/seed_data.py`
- Modify: `backend/main.py`

**Step 1: 修改 `backend/models/database.py`，新增三个表**

```python
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class DetectionTask(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    model_name: str = "VoltageTimesNet"
    status: str = "pending"
    anomaly_ratio: float = 2.085
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
    id: int = Field(default=1, primary_key=True)
    total_detections: int = 0
    total_anomalies_found: int = 0
    avg_processing_time_ms: float = 0.0
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
```

**Step 2: 创建种子数据脚本 `backend/scripts/seed_data.py`**

```python
"""生成测试用的设备、告警、电压历史数据"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import random
from datetime import datetime, timedelta
from sqlmodel import Session, create_engine, SQLModel

from models.database import Device, Alert, VoltageReading

DATABASE_URL = "sqlite:///./detection.db"
engine = create_engine(DATABASE_URL)

REGIONS = ["延庆区", "怀柔区", "密云区", "平谷区", "门头沟区"]
DEVICE_MODELS = ["DDZY866", "DTZY866", "DDS666", "DDN866"]
RESPONSIBLES = ["张伟", "李强", "王磊", "刘洋", "陈杰"]
STATUSES = ["normal", "normal", "normal", "attention", "warning", "critical"]

ALERT_TYPES = ["voltage_low", "voltage_high", "unbalance", "harmonic", "frequency"]
SEVERITIES = ["attention", "warning", "critical"]
ALERT_DESCS = {
    "voltage_low": "A相电压低于198V，触发欠压告警",
    "voltage_high": "B相电压超过242V，触发过压告警",
    "unbalance": "三相电压不平衡度超过5%",
    "harmonic": "总谐波畸变率THD超过5%",
    "frequency": "系统频率偏离50Hz±0.2Hz",
}


def seed_devices(session: Session) -> list[str]:
    codes = []
    for i in range(1, 31):
        code = f"DEV{i:04d}"
        codes.append(code)
        region = REGIONS[i % len(REGIONS)]
        d = Device(
            id=str(uuid.uuid4()),
            device_code=code,
            name=f"{region}#{i:02d}监测点",
            model=DEVICE_MODELS[i % len(DEVICE_MODELS)],
            region=region,
            status=STATUSES[i % len(STATUSES)],
            rated_voltage=220.0,
            capacity=random.choice([50.0, 100.0, 200.0, 315.0]),
            responsible=RESPONSIBLES[i % len(RESPONSIBLES)],
            address=f"{region}某乡镇第{i}监测点",
        )
        session.add(d)
    session.commit()
    print(f"✅ 已生成 {len(codes)} 个设备")
    return codes


def seed_alerts(session: Session, codes: list[str]):
    count = 0
    for i in range(80):
        code = random.choice(codes)
        atype = random.choice(ALERT_TYPES)
        severity = SEVERITIES[i % len(SEVERITIES)]
        created = datetime.utcnow() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
        )
        status = random.choice(["pending", "pending", "processing", "closed"])
        resolved_at = created + timedelta(hours=random.randint(1, 12)) if status == "closed" else None
        a = Alert(
            id=str(uuid.uuid4()),
            device_code=code,
            device_name=f"监测点{code}",
            location=f"{random.choice(REGIONS)}某乡镇",
            alert_type=atype,
            severity=severity,
            description=ALERT_DESCS[atype],
            status=status,
            created_at=created,
            resolved_at=resolved_at,
        )
        session.add(a)
        count += 1
    session.commit()
    print(f"✅ 已生成 {count} 条告警")


def seed_voltage_readings(session: Session, codes: list[str]):
    base_time = datetime.utcnow() - timedelta(days=30)
    # 每个设备每15分钟一条，共30天
    # 为节省时间，只生成前5个设备的历史数据，每设备24h × 4条/h = 96条
    count = 0
    for code in codes[:5]:
        for i in range(96 * 30):  # 30天
            ts = base_time + timedelta(minutes=15 * i)
            anomaly = random.random() < 0.05
            va = random.gauss(220, 3)
            if anomaly:
                va = random.choice([
                    random.uniform(185, 197),  # 欠压
                    random.uniform(243, 260),  # 过压
                ])
            r = VoltageReading(
                device_code=code,
                timestamp=ts,
                va=round(va, 2),
                vb=round(random.gauss(220, 3), 2),
                vc=round(random.gauss(220, 3), 2),
                ia=round(random.gauss(10, 1), 2),
                ib=round(random.gauss(10, 1), 2),
                ic=round(random.gauss(10, 1), 2),
                power_factor=round(random.uniform(0.88, 0.98), 3),
                is_anomaly=anomaly,
            )
            session.add(r)
            count += 1
        session.commit()
    print(f"✅ 已生成 {count} 条电压历史记录")


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # 检查是否已有数据
        from sqlmodel import select
        existing = session.exec(select(Device)).first()
        if existing:
            print("⚠️  数据库已有设备数据，跳过种子生成")
        else:
            codes = seed_devices(session)
            seed_alerts(session, codes)
            seed_voltage_readings(session, codes)
    print("✅ 种子数据生成完成")
```

**Step 3: 修改 `backend/main.py`，确保新表被创建并注册新路由**

在现有 `main.py` 中，找到 `SQLModel.metadata.create_all(engine)` 这行，确保 `models/database.py` 中的所有 SQLModel 类都被导入（触发表创建）。在 import 区域补充：

```python
# 在已有导入后追加（确保所有表被注册）
from models.database import Device, Alert, VoltageReading  # noqa: F401
```

并在路由注册区域新增：

```python
from api import devices, alerts, history, statistics, dashboard

app.include_router(devices.router, prefix="/api/v1/devices", tags=["设备管理"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["告警管理"])
app.include_router(history.router, prefix="/api/v1/history", tags=["历史数据"])
app.include_router(statistics.router, prefix="/api/v1/statistics", tags=["统计报表"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["监控概览"])
```

**Step 4: 运行种子脚本**

```bash
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp/backend
python scripts/seed_data.py
```

Expected: 看到 `✅ 已生成 30 个设备` / `80 条告警` / `N 条电压历史记录`

**Step 5: Commit**

```bash
git add backend/
git commit -m "feat: add Device/Alert/VoltageReading tables and seed data"
```

---

## Task 4: 后端 - 设备管理 API

**Files:**
- Create: `backend/api/devices.py`

**Step 1: 创建 `backend/api/devices.py`**

```python
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
    keyword: str = Query("", description="搜索设备编号或名称"),
    status: str = Query("", description="状态过滤"),
    region: str = Query("", description="区域过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    query = select(Device)
    if keyword:
        query = query.where(
            (Device.device_code.contains(keyword)) | (Device.name.contains(keyword))
        )
    if status:
        query = query.where(Device.status == status)
    if region:
        query = query.where(Device.region == region)

    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    items = session.exec(query.offset((page - 1) * page_size).limit(page_size)).all()
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
```

**Step 2: 验证 API**

```bash
cd backend && uvicorn main:app --reload --port 8000
curl http://localhost:8000/api/v1/devices
```

Expected: 返回含 `total` 和 `items` 的 JSON

**Step 3: Commit**

```bash
git add backend/api/devices.py backend/main.py
git commit -m "feat: add device CRUD API"
```

---

## Task 5: 后端 - 告警管理 API

**Files:**
- Create: `backend/api/alerts.py`

**Step 1: 创建 `backend/api/alerts.py`**

```python
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func
from pydantic import BaseModel

from models.database import Alert
from api.detect import get_session

router = APIRouter()


class AlertStatusUpdate(BaseModel):
    status: str  # pending/processing/closed


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
    query = select(Alert).order_by(Alert.created_at.desc())
    if start_date:
        query = query.where(Alert.created_at >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.where(Alert.created_at <= datetime.fromisoformat(end_date + "T23:59:59"))
    if severity:
        query = query.where(Alert.severity == severity)
    if status:
        query = query.where(Alert.status == status)
    if device_code:
        query = query.where(Alert.device_code.contains(device_code))

    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    items = session.exec(query.offset((page - 1) * page_size).limit(page_size)).all()
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/summary")
def alert_summary(session: Session = Depends(get_session)):
    for severity in ["attention", "warning", "critical"]:
        pass
    result = {}
    for sv in ["attention", "warning", "critical"]:
        count = session.exec(
            select(func.count()).where(Alert.severity == sv).where(Alert.status != "closed")
        ).one()
        result[sv] = count
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
```

**Step 2: 验证**

```bash
curl "http://localhost:8000/api/v1/alerts/summary"
```

Expected: `{"attention": N, "warning": N, "critical": N}`

**Step 3: Commit**

```bash
git add backend/api/alerts.py
git commit -m "feat: add alert management API with summary"
```

---

## Task 6: 后端 - 历史查询 + 统计 + 概览 API

**Files:**
- Create: `backend/api/history.py`
- Create: `backend/api/statistics.py`
- Create: `backend/api/dashboard.py`

**Step 1: 创建 `backend/api/history.py`**

```python
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select
from typing import Optional

from models.database import VoltageReading, Device
from api.detect import get_session

router = APIRouter()


@router.get("/devices")
def list_devices_for_history(session: Session = Depends(get_session)):
    """返回有历史数据的设备列表（供下拉选择）"""
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
```

**Step 2: 创建 `backend/api/statistics.py`**

```python
import random
from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_statistics():
    """返回统计报表所需数据（部分基于种子数据模拟）"""
    months = ["1月", "2月", "3月", "4月", "5月", "6月",
              "7月", "8月", "9月", "10月", "11月", "12月"]
    anomaly_trend = [random.randint(30, 120) for _ in months]
    anomaly_rate = [round(v / 1000 * 100, 2) for v in anomaly_trend]

    return {
        "anomaly_type_dist": [
            {"name": "欠压", "value": 42},
            {"name": "过压", "value": 18},
            {"name": "三相不平衡", "value": 25},
            {"name": "谐波畸变", "value": 10},
            {"name": "频率异常", "value": 5},
        ],
        "monthly_trend": {
            "months": months,
            "anomaly_count": anomaly_trend,
            "anomaly_rate": anomaly_rate,
        },
        "region_ranking": [
            {"region": "延庆区", "count": 35},
            {"region": "怀柔区", "count": 28},
            {"region": "密云区", "count": 22},
            {"region": "平谷区", "count": 15},
            {"region": "门头沟区", "count": 10},
        ],
        "voltage_quality": {
            "labels": ["电压合格率", "三相平衡度", "频率合格率", "功率因数", "谐波合规率", "供电可靠率"],
            "values": [93.5, 87.2, 98.1, 91.4, 95.6, 99.2],
        },
    }
```

**Step 3: 创建 `backend/api/dashboard.py`**

```python
from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func
from datetime import datetime, timedelta

from models.database import Device, Alert, SystemMetrics
from api.detect import get_session

router = APIRouter()


@router.get("/kpi")
def get_kpi(session: Session = Depends(get_session)):
    total_devices = session.exec(select(func.count()).select_from(Device)).one()
    online_devices = session.exec(
        select(func.count()).where(Device.status != "critical")
    ).one()
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
    today_alerts = session.exec(
        select(func.count()).where(Alert.created_at >= today_start)
    ).one()
    metrics = session.get(SystemMetrics, 1)

    return {
        "online_devices": online_devices,
        "total_devices": total_devices,
        "today_alerts": today_alerts,
        "voltage_pass_rate": 93.5,
        "avg_power_factor": 0.924,
        "model_f1": metrics.avg_processing_time_ms if metrics else 0,
    }


@router.get("/alerts/recent")
def get_recent_alerts(session: Session = Depends(get_session)):
    alerts = session.exec(
        select(Alert)
        .where(Alert.status != "closed")
        .order_by(Alert.created_at.desc())
        .limit(6)
    ).all()
    return alerts


@router.get("/device-status")
def get_device_status(session: Session = Depends(get_session)):
    result = {}
    for status in ["normal", "attention", "warning", "critical"]:
        count = session.exec(
            select(func.count()).where(Device.status == status)
        ).one()
        result[status] = count
    total = sum(result.values()) or 1
    return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in result.items()}
```

**Step 4: 验证所有新 API**

```bash
curl http://localhost:8000/api/v1/dashboard/kpi
curl http://localhost:8000/api/v1/statistics
curl "http://localhost:8000/api/v1/history/voltage?device_code=DEV0001"
```

**Step 5: Commit**

```bash
git add backend/api/
git commit -m "feat: add history/statistics/dashboard APIs"
```

---

## Task 7: 前端 - API 客户端扩展

**Files:**
- Modify: `frontend/src/api/client.ts`

**Step 1: 扩展 `frontend/src/api/client.ts`**

```typescript
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,
})

// ---- 已有 API（保留） ----
export const detectApi = {
  uploadAndDetect: (file: File, anomalyRatio: number) => {
    const form = new FormData()
    form.append('file', file)
    form.append('anomaly_ratio', anomalyRatio.toString())
    return api.post('/detect/upload', form).then(r => r.data)
  },
  detectSample: () => api.get('/detect/sample').then(r => r.data),
  getHistory: (limit = 20, offset = 0) =>
    api.get('/detect/history', { params: { limit, offset } }).then(r => r.data),
  getTaskResult: (id: string) => api.get(`/detect/${id}`).then(r => r.data),
}

export const modelsApi = {
  listModels: () => api.get('/models').then(r => r.data),
  getCurrentModel: () => api.get('/models/current').then(r => r.data),
}

export const metricsApi = {
  getMetrics: () => api.get('/metrics').then(r => r.data),
}

// ---- 新增 API ----
export const devicesApi = {
  list: (params?: Record<string, unknown>) =>
    api.get('/devices', { params }).then(r => r.data),
  create: (data: Record<string, unknown>) =>
    api.post('/devices', data).then(r => r.data),
  update: (id: string, data: Record<string, unknown>) =>
    api.put(`/devices/${id}`, data).then(r => r.data),
  delete: (id: string) => api.delete(`/devices/${id}`).then(r => r.data),
}

export const alertsApi = {
  list: (params?: Record<string, unknown>) =>
    api.get('/alerts', { params }).then(r => r.data),
  summary: () => api.get('/alerts/summary').then(r => r.data),
  updateStatus: (id: string, status: string) =>
    api.put(`/alerts/${id}/status`, { status }).then(r => r.data),
}

export const historyApi = {
  listDevices: () => api.get('/history/devices').then(r => r.data),
  getVoltage: (params: Record<string, unknown>) =>
    api.get('/history/voltage', { params }).then(r => r.data),
}

export const statisticsApi = {
  get: () => api.get('/statistics').then(r => r.data),
}

export const dashboardApi = {
  getKpi: () => api.get('/dashboard/kpi').then(r => r.data),
  getRecentAlerts: () => api.get('/dashboard/alerts/recent').then(r => r.data),
  getDeviceStatus: () => api.get('/dashboard/device-status').then(r => r.data),
}
```

**Step 2: Commit**

```bash
git add frontend/src/api/
git commit -m "feat: extend API client for new endpoints"
```

---

## Task 8: 前端 - 登录页

**Files:**
- Create: `frontend/src/pages/Login.tsx`

**Step 1: 创建 `frontend/src/pages/Login.tsx`**（仿照 Mockup 左右分屏设计）

```tsx
import { useState } from 'react'
import { Form, Input, Button, Checkbox, message } from 'antd'
import { UserOutlined, LockOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router'

export default function Login() {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const onFinish = async (values: { username: string; password: string }) => {
    setLoading(true)
    await new Promise(r => setTimeout(r, 800))
    if (values.username === 'admin' && values.password === 'admin123') {
      localStorage.setItem('token', 'mock-token-admin')
      message.success('登录成功')
      navigate('/')
    } else {
      message.error('用户名或密码错误')
    }
    setLoading(false)
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* 左侧品牌区 */}
      <div style={{
        flex: '0 0 55%', background: 'linear-gradient(135deg, #001529 0%, #003a8c 60%, #1677ff 100%)',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        padding: '60px 80px', position: 'relative', overflow: 'hidden',
      }}>
        {/* 背景装饰球 */}
        {[...Array(8)].map((_, i) => (
          <div key={i} style={{
            position: 'absolute',
            width: `${30 + i * 15}px`, height: `${30 + i * 15}px`,
            borderRadius: '50%',
            background: `rgba(22, 119, 255, ${0.05 + i * 0.02})`,
            top: `${10 + i * 10}%`, left: `${5 + i * 12}%`,
            animation: `float ${8 + i * 2}s linear infinite`,
          }} />
        ))}
        <style>{`
          @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0); }
            50% { transform: translateY(-20px) rotate(180deg); }
          }
          @keyframes pulse-line {
            0% { stroke-dashoffset: 100; }
            100% { stroke-dashoffset: 0; }
          }
        `}</style>

        {/* SVG 电力系统示意 */}
        <svg width="380" height="200" viewBox="0 0 380 200" style={{ marginBottom: 40 }}>
          {/* 传输线路 */}
          <line x1="60" y1="80" x2="320" y2="80" stroke="rgba(100,160,255,0.4)" strokeWidth="2"
            strokeDasharray="8 4" />
          <line x1="60" y1="80" x2="60" y2="160" stroke="rgba(100,160,255,0.4)" strokeWidth="2" />
          <line x1="190" y1="80" x2="190" y2="160" stroke="rgba(100,160,255,0.4)" strokeWidth="2" />
          <line x1="320" y1="80" x2="320" y2="160" stroke="rgba(100,160,255,0.4)" strokeWidth="2" />

          {/* 变压器塔 */}
          {[60, 190, 320].map((x, i) => (
            <g key={i}>
              <polygon points={`${x},${50 + i * 5} ${x - 20},${90 + i * 5} ${x + 20},${90 + i * 5}`}
                fill="none" stroke="rgba(100,160,255,0.7)" strokeWidth="1.5" />
              <line x1={x} y1={90 + i * 5} x2={x} y2={160} stroke="rgba(100,160,255,0.7)" strokeWidth="1.5" />
              <circle cx={x} cy={160} r="8" fill="rgba(22,119,255,0.6)" />
            </g>
          ))}

          {/* 脉冲动画 */}
          <circle r="4" fill="#1677ff" opacity="0.9">
            <animateMotion dur="3s" repeatCount="indefinite"
              path="M60,80 L190,80 L320,80" />
          </circle>
          <circle r="3" fill="#52c41a" opacity="0.8">
            <animateMotion dur="4s" repeatCount="indefinite" begin="1s"
              path="M60,80 L190,80 L320,80" />
          </circle>
        </svg>

        <h1 style={{ color: '#fff', fontSize: 28, fontWeight: 700, marginBottom: 12, textAlign: 'center' }}>
          农村电网低电压监管平台
        </h1>
        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14, marginBottom: 40 }}>
          Rural Grid Low-Voltage Monitoring Platform
        </p>

        <div style={{ display: 'flex', gap: 40 }}>
          {[
            { value: '24/7', label: '全天候监控' },
            { value: 'AI', label: '智能检测' },
            { value: '99.9%', label: '服务可用性' },
          ].map(item => (
            <div key={item.label} style={{ textAlign: 'center' }}>
              <div style={{ color: '#1677ff', fontSize: 24, fontWeight: 700 }}>{item.value}</div>
              <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, marginTop: 4 }}>{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 右侧登录区 */}
      <div style={{
        flex: 1, background: '#fff', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', padding: '60px 80px',
      }}>
        <div style={{ width: '100%', maxWidth: 360 }}>
          <div style={{ textAlign: 'center', marginBottom: 40 }}>
            <div style={{
              width: 56, height: 56, borderRadius: 16,
              background: 'linear-gradient(135deg, #1677ff, #003a8c)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 16px',
            }}>
              <ThunderboltOutlined style={{ color: '#fff', fontSize: 28 }} />
            </div>
            <h2 style={{ fontSize: 24, fontWeight: 700, color: '#1f2937', marginBottom: 6 }}>
              欢迎登录
            </h2>
            <p style={{ color: '#6b7280', fontSize: 14 }}>请输入您的账号和密码</p>
          </div>

          <Form onFinish={onFinish} size="large" initialValues={{ remember: true }}>
            <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
              <Input prefix={<UserOutlined style={{ color: '#9ca3af' }} />} placeholder="用户名" />
            </Form.Item>
            <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password prefix={<LockOutlined style={{ color: '#9ca3af' }} />} placeholder="密码" />
            </Form.Item>
            <Form.Item>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Form.Item name="remember" valuePropName="checked" noStyle>
                  <Checkbox>记住密码</Checkbox>
                </Form.Item>
                <a style={{ color: '#1677ff' }}>忘记密码？</a>
              </div>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading}
                style={{ width: '100%', height: 44, fontSize: 16,
                  background: 'linear-gradient(135deg, #1677ff, #003a8c)' }}>
                登 录
              </Button>
            </Form.Item>
          </Form>

          <p style={{ textAlign: 'center', color: '#9ca3af', fontSize: 12, marginTop: 24 }}>
            © 2024 农村电网低电压监管平台. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  )
}
```

**Step 2: 访问 http://localhost:5173/login 验证**

Expected: 左右分屏登录页，用 admin/admin123 可以登录

**Step 3: Commit**

```bash
git add frontend/src/pages/Login.tsx
git commit -m "feat: add login page with split-screen design"
```

---

## Task 9: 前端 - 监控概览（Dashboard）

**Files:**
- Modify: `frontend/src/pages/Dashboard.tsx`

**Step 1: 重写 `frontend/src/pages/Dashboard.tsx`**

```tsx
import { Row, Col, Card, Tag, Timeline, Progress } from 'antd'
import {
  AppstoreOutlined, AlertOutlined, CheckCircleOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { dashboardApi, alertsApi } from '@/api/client'

const STATUS_COLOR = {
  normal: '#52c41a', attention: '#1677ff', warning: '#faad14', critical: '#ff4d4f',
}
const STATUS_LABEL = {
  normal: '正常', attention: '注意', warning: '警告', critical: '严重',
}
const SEVERITY_COLOR: Record<string, string> = {
  attention: 'blue', warning: 'orange', critical: 'red',
}

function KpiCard({ title, value, unit, icon, color }: {
  title: string; value: string | number; unit?: string; icon: React.ReactNode; color: string
}) {
  return (
    <div className="stat-card" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
      <div className="kpi-icon-wrap" style={{ background: `${color}1a`, color }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: 30, fontWeight: 700, color: '#1f2937', lineHeight: 1 }}>
          {value}
          {unit && <span style={{ fontSize: 14, fontWeight: 400, marginLeft: 4, color: '#6b7280' }}>{unit}</span>}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { data: kpi } = useQuery({ queryKey: ['dashboard-kpi'], queryFn: dashboardApi.getKpi, refetchInterval: 30000 })
  const { data: recentAlerts } = useQuery({ queryKey: ['recent-alerts'], queryFn: dashboardApi.getRecentAlerts })
  const { data: deviceStatus } = useQuery({ queryKey: ['device-status'], queryFn: dashboardApi.getDeviceStatus })

  // 模拟24h三相电压趋势数据
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`)
  const mockVoltage = (base: number) => hours.map(() => +(base + (Math.random() - 0.5) * 10).toFixed(1))

  const voltageChartOption = {
    grid: { top: 40, right: 20, bottom: 60, left: 50 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['A相', 'B相', 'C相'], top: 0 },
    dataZoom: [{ type: 'slider', bottom: 0, height: 20 }],
    xAxis: { type: 'category', data: hours, axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', name: '电压(V)', min: 180, max: 250 },
    series: [
      { name: 'A相', type: 'line', data: mockVoltage(220), smooth: true,
        lineStyle: { color: '#1677ff' }, symbol: 'none',
        markLine: { silent: true, data: [{ yAxis: 198, lineStyle: { color: '#ff4d4f', type: 'dashed' } },
          { yAxis: 242, lineStyle: { color: '#ff4d4f', type: 'dashed' } }] } },
      { name: 'B相', type: 'line', data: mockVoltage(219), smooth: true,
        lineStyle: { color: '#52c41a' }, symbol: 'none' },
      { name: 'C相', type: 'line', data: mockVoltage(221), smooth: true,
        lineStyle: { color: '#faad14' }, symbol: 'none' },
    ],
  }

  const anomalyPieOption = {
    tooltip: { trigger: 'item' },
    legend: { orient: 'vertical', right: 10, top: 'center' },
    series: [{
      type: 'pie', radius: ['45%', '70%'], center: ['38%', '50%'],
      label: { show: false },
      data: [
        { name: '欠压', value: 42, itemStyle: { color: '#1677ff' } },
        { name: '过压', value: 18, itemStyle: { color: '#ff4d4f' } },
        { name: '三相不平衡', value: 25, itemStyle: { color: '#faad14' } },
        { name: '谐波畸变', value: 10, itemStyle: { color: '#52c41a' } },
        { name: '频率异常', value: 5, itemStyle: { color: '#722ed1' } },
      ],
    }],
  }

  return (
    <div>
      {/* KPI 卡片行 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <KpiCard title="在线设备" value={kpi?.online_devices ?? '--'}
            unit={`/ ${kpi?.total_devices ?? '--'} 台`}
            icon={<AppstoreOutlined />} color="#1677ff" />
        </Col>
        <Col span={6}>
          <KpiCard title="今日告警" value={kpi?.today_alerts ?? '--'}
            unit="条" icon={<AlertOutlined />} color="#ff4d4f" />
        </Col>
        <Col span={6}>
          <KpiCard title="电压合格率" value={kpi?.voltage_pass_rate ?? '--'}
            unit="%" icon={<CheckCircleOutlined />} color="#52c41a" />
        </Col>
        <Col span={6}>
          <KpiCard title="平均功率因数" value={kpi?.avg_power_factor ?? '--'}
            icon={<ThunderboltOutlined />} color="#faad14" />
        </Col>
      </Row>

      {/* 电压趋势 + 告警时间线 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={15}>
          <div className="chart-container">
            <div className="chart-title">24小时三相电压趋势</div>
            <ReactECharts option={voltageChartOption} style={{ height: 280 }} />
          </div>
        </Col>
        <Col span={9}>
          <div className="chart-container" style={{ height: '100%' }}>
            <div className="chart-title">最新告警</div>
            <Timeline style={{ marginTop: 8, maxHeight: 260, overflowY: 'auto' }}
              items={(recentAlerts || []).map((a: Record<string, string>) => ({
                color: STATUS_COLOR[a.severity as keyof typeof STATUS_COLOR] || '#999',
                children: (
                  <div>
                    <Tag color={SEVERITY_COLOR[a.severity]}>{a.severity === 'critical' ? '严重' : a.severity === 'warning' ? '警告' : '注意'}</Tag>
                    <span style={{ fontSize: 12, color: '#6b7280', marginLeft: 4 }}>{a.device_code}</span>
                    <div style={{ fontSize: 13, color: '#374151', marginTop: 2 }}>{a.description}</div>
                  </div>
                ),
              }))}
            />
          </div>
        </Col>
      </Row>

      {/* 异常分布 + 设备状态 */}
      <Row gutter={16}>
        <Col span={9}>
          <div className="chart-container">
            <div className="chart-title">异常类型分布</div>
            <ReactECharts option={anomalyPieOption} style={{ height: 220 }} />
          </div>
        </Col>
        <Col span={15}>
          <div className="chart-container" style={{ height: '100%' }}>
            <div className="chart-title">设备状态总览</div>
            <div style={{ padding: '8px 0' }}>
              {(['normal', 'attention', 'warning', 'critical'] as const).map(s => {
                const d = deviceStatus?.[s] || { count: 0, pct: 0 }
                return (
                  <div key={s} style={{ marginBottom: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span style={{ fontSize: 13, color: '#374151' }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                          background: STATUS_COLOR[s], marginRight: 6 }} />
                        {STATUS_LABEL[s]}
                      </span>
                      <span style={{ fontSize: 13, color: '#6b7280' }}>{d.count} 台 ({d.pct}%)</span>
                    </div>
                    <Progress percent={d.pct} showInfo={false}
                      strokeColor={STATUS_COLOR[s]} trailColor="#f0f0f0" strokeWidth={10} />
                  </div>
                )
              })}
            </div>
          </div>
        </Col>
      </Row>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/src/pages/Dashboard.tsx
git commit -m "feat: rewrite dashboard with KPI cards, charts, and alert timeline"
```

---

## Task 10: 前端 - 设备管理页

**Files:**
- Create: `frontend/src/pages/Devices.tsx`

**Step 1: 创建 `frontend/src/pages/Devices.tsx`**

```tsx
import { useState } from 'react'
import { Table, Button, Input, Select, Row, Col, Tag, Modal, Form, Space, Popconfirm, message } from 'antd'
import { PlusOutlined, SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { devicesApi } from '@/api/client'

const STATUS_MAP: Record<string, { color: string; label: string }> = {
  normal: { color: 'success', label: '正常' },
  attention: { color: 'processing', label: '注意' },
  warning: { color: 'warning', label: '警告' },
  critical: { color: 'error', label: '严重' },
}
const REGIONS = ['延庆区', '怀柔区', '密云区', '平谷区', '门头沟区']

export default function Devices() {
  const [filters, setFilters] = useState({ keyword: '', status: '', region: '', page: 1 })
  const [modalOpen, setModalOpen] = useState(false)
  const [detailDevice, setDetailDevice] = useState<Record<string, unknown> | null>(null)
  const [editDevice, setEditDevice] = useState<Record<string, unknown> | null>(null)
  const [form] = Form.useForm()
  const qc = useQueryClient()

  const { data, isFetching } = useQuery({
    queryKey: ['devices', filters],
    queryFn: () => devicesApi.list(filters),
  })

  const createMut = useMutation({
    mutationFn: devicesApi.create,
    onSuccess: () => { message.success('设备创建成功'); qc.invalidateQueries({ queryKey: ['devices'] }); setModalOpen(false) },
  })

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) => devicesApi.update(id, data),
    onSuccess: () => { message.success('更新成功'); qc.invalidateQueries({ queryKey: ['devices'] }); setEditDevice(null) },
  })

  const deleteMut = useMutation({
    mutationFn: devicesApi.delete,
    onSuccess: () => { message.success('删除成功'); qc.invalidateQueries({ queryKey: ['devices'] }) },
  })

  const columns = [
    { title: '设备编号', dataIndex: 'device_code', width: 100 },
    { title: '设备名称', dataIndex: 'name' },
    { title: '型号', dataIndex: 'model', width: 100 },
    { title: '所属区域', dataIndex: 'region', width: 90 },
    { title: '状态', dataIndex: 'status', width: 80,
      render: (v: string) => <Tag color={STATUS_MAP[v]?.color}>{STATUS_MAP[v]?.label}</Tag> },
    { title: '额定电压(V)', dataIndex: 'rated_voltage', width: 110 },
    { title: '容量(kVA)', dataIndex: 'capacity', width: 90 },
    { title: '负责人', dataIndex: 'responsible', width: 80 },
    {
      title: '操作', width: 160,
      render: (_: unknown, r: Record<string, unknown>) => (
        <Space>
          <Button type="link" size="small" onClick={() => setDetailDevice(r)}>详情</Button>
          <Button type="link" size="small" onClick={() => { setEditDevice(r); form.setFieldsValue(r) }}>编辑</Button>
          <Popconfirm title="确认删除该设备？" onConfirm={() => deleteMut.mutate(r.id as string)}>
            <Button type="link" size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  const DeviceForm = () => (
    <Form form={form} layout="vertical" labelCol={{ span: 6 }}>
      <Row gutter={16}>
        <Col span={12}>
          <Form.Item label="设备编号" name="device_code" rules={[{ required: true }]}>
            <Input disabled={!!editDevice} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="设备名称" name="name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="型号" name="model" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="所属区域" name="region" rules={[{ required: true }]}>
            <Select options={REGIONS.map(r => ({ label: r, value: r }))} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="状态" name="status">
            <Select options={Object.entries(STATUS_MAP).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="负责人" name="responsible">
            <Input />
          </Form.Item>
        </Col>
      </Row>
    </Form>
  )

  return (
    <div>
      {/* 筛选栏 */}
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle">
          <Col><Input prefix={<SearchOutlined />} placeholder="设备编号/名称"
            value={filters.keyword} onChange={e => setFilters(f => ({ ...f, keyword: e.target.value }))}
            style={{ width: 200 }} /></Col>
          <Col><Select placeholder="状态" allowClear style={{ width: 120 }}
            onChange={v => setFilters(f => ({ ...f, status: v || '' }))}
            options={Object.entries(STATUS_MAP).map(([k, v]) => ({ label: v.label, value: k }))} /></Col>
          <Col><Select placeholder="区域" allowClear style={{ width: 120 }}
            onChange={v => setFilters(f => ({ ...f, region: v || '' }))}
            options={REGIONS.map(r => ({ label: r, value: r }))} /></Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />}
              onClick={() => setFilters(f => ({ ...f, page: 1 }))}>查询</Button>
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />}
              onClick={() => setFilters({ keyword: '', status: '', region: '', page: 1 })}>重置</Button>
          </Col>
          <Col flex="auto" style={{ textAlign: 'right' }}>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => { form.resetFields(); setModalOpen(true) }}>
              添加设备
            </Button>
          </Col>
        </Row>
      </div>

      {/* 数据表格 */}
      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={data?.items || []}
          rowKey="id"
          loading={isFetching}
          size="middle"
          pagination={{
            total: data?.total,
            current: filters.page,
            pageSize: 10,
            showSizeChanger: true,
            showTotal: t => `共 ${t} 条`,
            onChange: page => setFilters(f => ({ ...f, page })),
          }}
          scroll={{ x: 900 }}
        />
      </div>

      {/* 新增弹窗 */}
      <Modal title="添加设备" open={modalOpen} width={600}
        onOk={() => form.validateFields().then(v => createMut.mutate(v))}
        onCancel={() => setModalOpen(false)} okText="保存" cancelText="取消">
        <DeviceForm />
      </Modal>

      {/* 编辑弹窗 */}
      <Modal title="编辑设备" open={!!editDevice} width={600}
        onOk={() => form.validateFields().then(v => updateMut.mutate({ id: editDevice!.id as string, data: v }))}
        onCancel={() => setEditDevice(null)} okText="保存" cancelText="取消">
        <DeviceForm />
      </Modal>

      {/* 详情弹窗 */}
      <Modal title="设备详情" open={!!detailDevice} footer={null}
        onCancel={() => setDetailDevice(null)}>
        {detailDevice && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 24px', padding: '8px 0' }}>
            {Object.entries({
              '设备编号': detailDevice.device_code, '设备名称': detailDevice.name,
              '型号': detailDevice.model, '所属区域': detailDevice.region,
              '额定电压': `${detailDevice.rated_voltage}V`, '容量': `${detailDevice.capacity}kVA`,
              '负责人': detailDevice.responsible, '状态': STATUS_MAP[detailDevice.status as string]?.label,
            }).map(([k, v]) => (
              <div key={k}>
                <div style={{ color: '#9ca3af', fontSize: 12 }}>{k}</div>
                <div style={{ color: '#1f2937', fontWeight: 500 }}>{v as string}</div>
              </div>
            ))}
          </div>
        )}
      </Modal>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/src/pages/Devices.tsx
git commit -m "feat: add device management page with CRUD"
```

---

## Task 11: 前端 - 告警管理页

**Files:**
- Create: `frontend/src/pages/Alerts.tsx`

**Step 1: 创建 `frontend/src/pages/Alerts.tsx`**

```tsx
import { useState } from 'react'
import { Table, Button, Select, Row, Col, Tag, DatePicker, Input, Space, message } from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { alertsApi } from '@/api/client'
import dayjs from 'dayjs'

const { RangePicker } = DatePicker

const SEVERITY = {
  attention: { color: 'blue', label: '注意' },
  warning: { color: 'orange', label: '警告' },
  critical: { color: 'red', label: '严重' },
}
const STATUS = {
  pending: { color: 'default', label: '未处理' },
  processing: { color: 'processing', label: '处理中' },
  closed: { color: 'success', label: '已关闭' },
}
const ALERT_TYPE_LABEL: Record<string, string> = {
  voltage_low: '欠压', voltage_high: '过压',
  unbalance: '三相不平衡', harmonic: '谐波畸变', frequency: '频率异常',
}

export default function Alerts() {
  const [filters, setFilters] = useState({ severity: '', status: '', device_code: '', page: 1, dateRange: null as null | [string, string] })
  const qc = useQueryClient()

  const { data: summary } = useQuery({ queryKey: ['alert-summary'], queryFn: alertsApi.summary })
  const { data, isFetching } = useQuery({
    queryKey: ['alerts', filters],
    queryFn: () => alertsApi.list({
      severity: filters.severity, status: filters.status,
      device_code: filters.device_code, page: filters.page,
      ...(filters.dateRange ? { start_date: filters.dateRange[0], end_date: filters.dateRange[1] } : {}),
    }),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => alertsApi.updateStatus(id, status),
    onSuccess: () => { message.success('状态更新成功'); qc.invalidateQueries({ queryKey: ['alerts'] }); qc.invalidateQueries({ queryKey: ['alert-summary'] }) },
  })

  const summaryCards = [
    { key: 'attention', label: '注意告警', color: '#1677ff', borderColor: '#1677ff' },
    { key: 'warning', label: '警告告警', color: '#faad14', borderColor: '#faad14' },
    { key: 'critical', label: '严重告警', color: '#ff4d4f', borderColor: '#ff4d4f' },
  ]

  const columns = [
    { title: '时间', dataIndex: 'created_at', width: 150,
      render: (v: string) => dayjs(v).format('MM-DD HH:mm') },
    { title: '设备编号', dataIndex: 'device_code', width: 100 },
    { title: '位置', dataIndex: 'location' },
    { title: '类型', dataIndex: 'alert_type', width: 100,
      render: (v: string) => ALERT_TYPE_LABEL[v] || v },
    { title: '严重程度', dataIndex: 'severity', width: 90,
      render: (v: keyof typeof SEVERITY) => <Tag color={SEVERITY[v]?.color}>{SEVERITY[v]?.label}</Tag> },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'status', width: 90,
      render: (v: keyof typeof STATUS) => <Tag color={STATUS[v]?.color}>{STATUS[v]?.label}</Tag> },
    {
      title: '操作', width: 160,
      render: (_: unknown, r: Record<string, unknown>) => (
        <Space>
          {r.status === 'pending' && (
            <Button type="link" size="small"
              onClick={() => updateMut.mutate({ id: r.id as string, status: 'processing' })}>
              开始处理
            </Button>
          )}
          {r.status === 'processing' && (
            <Button type="link" size="small"
              onClick={() => updateMut.mutate({ id: r.id as string, status: 'closed' })}>
              关闭
            </Button>
          )}
          {r.status === 'closed' && <span style={{ color: '#9ca3af', fontSize: 13 }}>已关闭</span>}
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* 摘要卡片 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        {summaryCards.map(card => (
          <Col span={8} key={card.key}>
            <div className="alert-summary-card" style={{ borderLeftColor: card.borderColor }}>
              <div style={{ color: '#6b7280', fontSize: 13, marginBottom: 8 }}>{card.label}</div>
              <div style={{ fontSize: 36, fontWeight: 700, color: card.color, lineHeight: 1 }}>
                {summary?.[card.key] ?? '--'}
              </div>
              <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>条未处理</div>
            </div>
          </Col>
        ))}
      </Row>

      {/* 筛选栏 */}
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle" wrap>
          <Col>
            <RangePicker onChange={(_, s) => setFilters(f => ({ ...f, dateRange: s[0] ? [s[0], s[1]] : null }))} />
          </Col>
          <Col>
            <Select placeholder="严重程度" allowClear style={{ width: 120 }}
              onChange={v => setFilters(f => ({ ...f, severity: v || '' }))}
              options={Object.entries(SEVERITY).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Col>
          <Col>
            <Select placeholder="处理状态" allowClear style={{ width: 120 }}
              onChange={v => setFilters(f => ({ ...f, status: v || '' }))}
              options={Object.entries(STATUS).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Col>
          <Col>
            <Input prefix={<SearchOutlined />} placeholder="设备编号" style={{ width: 160 }}
              onChange={e => setFilters(f => ({ ...f, device_code: e.target.value }))} />
          </Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />}
              onClick={() => setFilters(f => ({ ...f, page: 1 }))}>查询</Button>
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />}
              onClick={() => setFilters({ severity: '', status: '', device_code: '', page: 1, dateRange: null })}>
              重置
            </Button>
          </Col>
        </Row>
      </div>

      {/* 数据表格 */}
      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={data?.items || []}
          rowKey="id"
          loading={isFetching}
          size="middle"
          pagination={{
            total: data?.total,
            current: filters.page,
            pageSize: 10,
            showTotal: t => `共 ${t} 条`,
            onChange: page => setFilters(f => ({ ...f, page })),
          }}
          scroll={{ x: 900 }}
        />
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/src/pages/Alerts.tsx
git commit -m "feat: add alert management page with filters and status update"
```

---

## Task 12: 前端 - 历史查询页

**Files:**
- Modify: `frontend/src/pages/History.tsx`

**Step 1: 重写 `frontend/src/pages/History.tsx`**

```tsx
import { useState } from 'react'
import { Row, Col, Select, Button, DatePicker, Checkbox, Table, Tag, message } from 'antd'
import { SearchOutlined, DownloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { historyApi } from '@/api/client'
import dayjs from 'dayjs'

const { RangePicker } = DatePicker

export default function History() {
  const [deviceCode, setDeviceCode] = useState('')
  const [dateRange, setDateRange] = useState<[string, string]>([
    dayjs().subtract(1, 'day').toISOString(), dayjs().toISOString(),
  ])
  const [dataTypes, setDataTypes] = useState(['voltage'])
  const [queried, setQueried] = useState(false)

  const { data: devices } = useQuery({ queryKey: ['history-devices'], queryFn: historyApi.listDevices })
  const { data: voltageData, refetch, isFetching } = useQuery({
    queryKey: ['history-voltage', deviceCode, dateRange],
    queryFn: () => historyApi.getVoltage({ device_code: deviceCode, start: dateRange[0], end: dateRange[1] }),
    enabled: queried && !!deviceCode,
  })

  const rows: Record<string, unknown>[] = voltageData?.data || []
  const timestamps = rows.map((r) => dayjs(r.timestamp as string).format('MM-DD HH:mm'))

  const chartOption = {
    grid: { top: 40, right: 20, bottom: 60, left: 50 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['A相电压', 'B相电压', 'C相电压'], top: 0 },
    dataZoom: [{ type: 'slider', bottom: 0, height: 20 }],
    xAxis: { type: 'category', data: timestamps, axisLabel: { rotate: 30, fontSize: 11 } },
    yAxis: { type: 'value', name: '电压(V)', min: 180, max: 260 },
    series: [
      { name: 'A相电压', type: 'line', data: rows.map(r => r.va), smooth: true,
        lineStyle: { color: '#1677ff' }, symbol: 'none',
        markLine: { silent: true, data: [
          { yAxis: 198, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '下限198V' } },
          { yAxis: 242, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '上限242V' } },
        ] } },
      { name: 'B相电压', type: 'line', data: rows.map(r => r.vb), smooth: true,
        lineStyle: { color: '#52c41a' }, symbol: 'none' },
      { name: 'C相电压', type: 'line', data: rows.map(r => r.vc), smooth: true,
        lineStyle: { color: '#faad14' }, symbol: 'none' },
    ],
  }

  const columns = [
    { title: '时间', dataIndex: 'timestamp', width: 150,
      render: (v: string) => dayjs(v).format('MM-DD HH:mm') },
    { title: 'A相电压(V)', dataIndex: 'va',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined, fontWeight: (v < 198 || v > 242) ? 600 : undefined }}>{v}</span> },
    { title: 'B相电压(V)', dataIndex: 'vb',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined }}>{v}</span> },
    { title: 'C相电压(V)', dataIndex: 'vc',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined }}>{v}</span> },
    { title: 'A相电流(A)', dataIndex: 'ia' },
    { title: 'B相电流(A)', dataIndex: 'ib' },
    { title: 'C相电流(A)', dataIndex: 'ic' },
    { title: '功率因数', dataIndex: 'power_factor',
      render: (v: number) => <span style={{ color: v < 0.85 ? '#faad14' : undefined }}>{v}</span> },
    { title: '异常', dataIndex: 'is_anomaly', width: 70,
      render: (v: boolean) => v ? <Tag color="red">异常</Tag> : null },
  ]

  const exportCsv = () => {
    if (!rows.length) { message.warning('没有数据可导出'); return }
    const header = '时间,A相电压,B相电压,C相电压,A相电流,B相电流,C相电流,功率因数\n'
    const content = rows.map(r =>
      `${r.timestamp},${r.va},${r.vb},${r.vc},${r.ia},${r.ib},${r.ic},${r.power_factor}`
    ).join('\n')
    const blob = new Blob([header + content], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `voltage_${deviceCode}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      {/* 查询区 */}
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle" style={{ marginBottom: 12 }}>
          <Col>
            <Select placeholder="选择监测设备" style={{ width: 220 }} value={deviceCode || undefined}
              onChange={setDeviceCode}
              options={(devices || []).map((d: Record<string, string>) => ({ label: `${d.code} ${d.name}`, value: d.code }))} />
          </Col>
          <Col>
            <RangePicker showTime onChange={(_, s) => s[0] && setDateRange([s[0], s[1]])} />
          </Col>
          <Col>
            {[{ label: '24小时', hours: 24 }, { label: '7天', hours: 168 }, { label: '30天', hours: 720 }].map(q => (
              <Button key={q.label} size="small" style={{ marginRight: 4 }}
                onClick={() => setDateRange([dayjs().subtract(q.hours, 'hour').toISOString(), dayjs().toISOString()])}>
                {q.label}
              </Button>
            ))}
          </Col>
        </Row>
        <Row gutter={12} align="middle">
          <Col>
            <Checkbox.Group value={dataTypes} onChange={v => setDataTypes(v as string[])}
              options={[
                { label: '电压', value: 'voltage' },
                { label: '电流', value: 'current' },
                { label: '功率因数', value: 'pf' },
              ]} />
          </Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />} loading={isFetching}
              onClick={() => { if (!deviceCode) { message.warning('请选择设备'); return }; setQueried(true); refetch() }}>
              查询
            </Button>
          </Col>
          <Col>
            <Button icon={<DownloadOutlined />} onClick={exportCsv}>导出CSV</Button>
          </Col>
        </Row>
      </div>

      {/* 图表 */}
      {rows.length > 0 && (
        <div className="chart-container" style={{ marginBottom: 16 }}>
          <div className="chart-title">电压历史趋势</div>
          <ReactECharts option={chartOption} style={{ height: 300 }} />
        </div>
      )}

      {/* 数据表 */}
      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={rows}
          rowKey="timestamp"
          size="middle"
          loading={isFetching}
          pagination={{ pageSize: 20, showTotal: t => `共 ${t} 条` }}
          scroll={{ x: 900 }}
          rowClassName={(r) => (r.is_anomaly ? 'anomaly-row' : '')}
        />
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/src/pages/History.tsx
git commit -m "feat: rewrite history query page with voltage chart and CSV export"
```

---

## Task 13: 前端 - 统计报表页

**Files:**
- Modify: `frontend/src/pages/Statistics.tsx` (or create if needed)

**Step 1: 重写统计报表页**

```tsx
import { Row, Col, Button } from 'antd'
import { DownloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { statisticsApi } from '@/api/client'

export default function Statistics() {
  const { data } = useQuery({ queryKey: ['statistics'], queryFn: statisticsApi.get })

  if (!data) return <div>加载中...</div>

  const pieOption = {
    tooltip: { trigger: 'item' },
    legend: { orient: 'vertical', right: 10, top: 'center' },
    series: [{
      type: 'pie', radius: ['45%', '70%'], center: ['38%', '50%'],
      label: { show: false },
      data: (data.anomaly_type_dist || []).map((d: Record<string, unknown>, i: number) => ({
        ...d,
        itemStyle: { color: ['#1677ff', '#ff4d4f', '#faad14', '#52c41a', '#722ed1'][i] },
      })),
    }],
  }

  const barLineOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['异常次数', '异常率(%)'] },
    xAxis: { type: 'category', data: data.monthly_trend?.months },
    yAxis: [
      { type: 'value', name: '次数' },
      { type: 'value', name: '异常率(%)', position: 'right' },
    ],
    series: [
      {
        name: '异常次数', type: 'bar', data: data.monthly_trend?.anomaly_count,
        itemStyle: {
          color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: '#1677ff' }, { offset: 1, color: '#69b1ff' }] },
        },
      },
      {
        name: '异常率(%)', type: 'line', yAxisIndex: 1, data: data.monthly_trend?.anomaly_rate,
        lineStyle: { color: '#faad14' }, itemStyle: { color: '#faad14' }, smooth: true, symbol: 'none',
      },
    ],
  }

  const hbarOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 80, right: 60 },
    xAxis: { type: 'value', name: '异常次数' },
    yAxis: { type: 'category', data: (data.region_ranking || []).map((r: Record<string, unknown>) => r.region) },
    series: [{
      type: 'bar',
      data: (data.region_ranking || []).map((r: Record<string, unknown>, i: number) => ({
        value: r.count,
        itemStyle: { color: i < 3 ? { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#1677ff' }, { offset: 1, color: '#69b1ff' }] } : '#adc6ff' },
      })),
      label: { show: true, position: 'right' },
    }],
  }

  const radarOption = {
    tooltip: {},
    radar: {
      indicator: (data.voltage_quality?.labels || []).map((l: string) => ({ name: l, max: 100 })),
    },
    series: [{
      type: 'radar',
      data: [{
        value: data.voltage_quality?.values,
        name: '电压质量',
        areaStyle: { color: 'rgba(22, 119, 255, 0.2)' },
        lineStyle: { color: '#1677ff' },
        itemStyle: { color: '#1677ff' },
      }],
    }],
  }

  const charts = [
    { title: '异常类型分布', option: pieOption },
    { title: '月度异常趋势', option: barLineOption },
    { title: '区域异常排行', option: hbarOption },
    { title: '电压质量指标', option: radarOption },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
        <Button icon={<DownloadOutlined />} type="primary">导出报告</Button>
      </div>
      <Row gutter={16}>
        {charts.map((c, i) => (
          <Col span={12} key={i} style={{ marginBottom: 16 }}>
            <div className="chart-container">
              <div className="chart-title">{c.title}</div>
              <ReactECharts option={c.option} style={{ height: 280 }} />
            </div>
          </Col>
        ))}
      </Row>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/src/pages/Statistics.tsx
git commit -m "feat: rewrite statistics page with 4-chart 2x2 grid"
```

---

## Task 14 & 15: 保留 Detect/Models 页，应用 Ant Design 基础样式

**Files:**
- Modify: `frontend/src/pages/Detect.tsx`
- Modify: `frontend/src/pages/Models.tsx`

这两个页面**保留所有逻辑**，仅将 Tailwind className 替换为 Ant Design 的 `Card`、`Button`、`Tag`、`Upload` 等组件，并使用 `className="stat-card"` 或 `antd` 组件来维持视觉一致性。主要工作量是替换 HTML 标签和 Tailwind class，不改动任何业务逻辑。

关键替换规则：
- `className="bg-white rounded-lg shadow p-6"` → `className="stat-card"`
- `<button>` → `<Button type="primary">`
- 状态徽章 → `<Tag color="...">`
- `className="text-gray-500"` → `style={{ color: '#6b7280' }}`

**Step: Commit 后完成**

```bash
git add frontend/src/pages/
git commit -m "feat: apply antd styles to detect and models pages"
```

---

## Task 16: 整体联调 + 验证

**Step 1: 启动完整系统**

```bash
# 终端1: 启动后端
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp/backend
uvicorn main:app --reload --port 8000

# 终端2: 启动前端
cd /Users/xiaodongzheng/paper/Low-Voltage-Detection/Rural-Low-Voltage-Detection/webapp/frontend
npm run dev
```

**Step 2: 验证清单**

- [ ] http://localhost:5173/login → 左右分屏登录页，admin/admin123 可登录
- [ ] http://localhost:5173/ → 监控概览，KPI 卡片 + 图表 + 告警时间线均显示
- [ ] http://localhost:5173/devices → 设备列表，可筛选、可增删改查
- [ ] http://localhost:5173/alerts → 告警管理，摘要卡片 + 筛选 + 状态更新
- [ ] http://localhost:5173/history → 选择设备后查询，图表 + 表格 + CSV 导出
- [ ] http://localhost:5173/statistics → 4 个图表正确渲染
- [ ] http://localhost:5173/detect → ML 检测功能正常
- [ ] http://localhost:5173/models → 模型对比正常

**Step 3: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete frontend mockup migration - all pages implemented with antd"
```

---

## 成功标准

1. ✅ 登录页左右分屏，视觉与 Mockup 高度一致
2. ✅ 深色侧边栏（#001529）+ 白色顶部 Header
3. ✅ 监控概览 KPI + 图表 + 告警时间线
4. ✅ 设备管理支持完整 CRUD
5. ✅ 告警管理分级筛选 + 状态流转
6. ✅ 历史查询含电压图表 + 数据表 + CSV 导出
7. ✅ 统计报表 2x2 四图表
8. ✅ 原有 ML 检测功能不受影响
