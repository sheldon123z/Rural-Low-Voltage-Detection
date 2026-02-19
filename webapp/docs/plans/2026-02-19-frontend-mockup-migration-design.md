# 前端 Mockup 迁移设计文档

**日期**: 2026-02-19
**项目**: 农村电网低电压监管平台 - 前端美化迁移

---

## 1. 项目背景

当前真实系统前端（React + Tailwind）页面设计较为简陋，论文 Mockup（Vue 3 + Element Plus）具有专业的企业级设计。本次工作目标是将 Mockup 的全部页面设计迁移到真实系统中，同时保留现有的 ML 推理功能。

---

## 2. 技术栈决策

| 层 | 现状 | 迁移后 |
|----|------|--------|
| 前端框架 | React 19 + TypeScript | React 19 + TypeScript（保留） |
| UI 组件库 | Tailwind CSS + Radix UI | **Ant Design (antd 5.x)** |
| 图表库 | ECharts + echarts-for-react | ECharts + echarts-for-react（保留） |
| 路由 | React Router 7 | React Router 7（保留） |
| 状态管理 | TanStack Query | TanStack Query（保留） |
| HTTP 客户端 | Axios | Axios（保留） |

**选择 Ant Design 原因**：
- 与 Mockup 使用的 Element Plus 设计语言高度相似
- 完整的企业级组件（Table、Form、Menu、Modal 等）
- 深色侧边栏支持
- 无需重建底层组件

---

## 3. 新系统页面架构

### 3.1 路由结构

```
/login                    登录页（左右分屏 + SVG动画）
/                         监控概览（KPI卡片 + 多图表）
/devices                  设备管理（表格 + CRUD弹窗）
/alerts                   告警管理（分级筛选 + 状态更新）
/history                  历史数据查询（时序图 + 数据表）
/statistics               统计报表（2x2四图表网格）
/detect                   异常检测（保留现有 ML 功能）
/models                   模型对比（保留现有功能）
```

### 3.2 布局结构

```
MainLayout
├── Sider（220px / 折叠 64px）
│   ├── Logo区
│   └── Menu（深色主题 #001529）
│       ├── 监控概览
│       ├── 设备管理组
│       │   └── 设备列表
│       ├── 异常分析组
│       │   ├── 告警管理
│       │   └── 异常检测
│       ├── 数据中心组
│       │   ├── 历史查询
│       │   └── 统计报表
│       └── 模型中心组
│           └── 模型对比
└── Content
    ├── Header（56px, 折叠按钮+面包屑+时间+用户下拉）
    └── 主内容区（路由渲染）
```

---

## 4. 设计系统

### 4.1 颜色变量

```css
--primary-color: #1677ff;
--success-color: #52c41a;
--warning-color: #faad14;
--danger-color: #ff4d4f;
--attention-color: #1677ff;
--text-primary: #1f2937;
--text-secondary: #6b7280;
--border-color: #e5e7eb;
--bg-gray: #f0f2f5;
--sidebar-bg: #001529;
--header-height: 56px;
--sidebar-width: 220px;
```

### 4.2 组件规范

**卡片**：`border-radius: 8px`，`box-shadow: 0 1px 4px rgba(0,0,0,0.06)`

**KPI卡片**：左侧彩色图标区（52x52px），悬停上浮2px

**表格**：条纹斑马纹，头部 `#f5f7fa`，悬停 `#ecf5ff`

**告警等级标签**：注意（蓝）/ 警告（橙）/ 严重（红）

---

## 5. 后端扩展

### 5.1 新增数据库表

```python
# Device 表
class Device(SQLModel):
    id: str  # UUID
    device_code: str       # 设备编号
    name: str              # 设备名称
    model: str             # 型号
    region: str            # 所属区域
    status: str            # normal/attention/warning/critical
    rated_voltage: float   # 额定电压 (V)
    capacity: float        # 容量 (kVA)
    responsible: str       # 负责人
    created_at: datetime

# Alert 表
class Alert(SQLModel):
    id: str
    device_code: str
    device_name: str
    location: str
    alert_type: str        # voltage_low/voltage_high/unbalance/harmonic
    severity: str          # attention/warning/critical
    description: str
    status: str            # pending/processing/closed
    created_at: datetime
    resolved_at: Optional[datetime]

# VoltageReading 表（历史数据）
class VoltageReading(SQLModel):
    id: int
    device_code: str
    timestamp: datetime
    va: float  # A相电压
    vb: float  # B相电压
    vc: float  # C相电压
    ia: float  # A相电流
    ib: float
    ic: float
    power_factor: float
```

### 5.2 新增 API 路由

```
设备管理:
  GET    /api/v1/devices                  # 列表（支持分页、搜索、状态过滤）
  POST   /api/v1/devices                  # 新增设备
  PUT    /api/v1/devices/{id}             # 编辑设备
  DELETE /api/v1/devices/{id}             # 删除设备

告警管理:
  GET    /api/v1/alerts                   # 列表（支持时间/级别/状态过滤）
  PUT    /api/v1/alerts/{id}/status       # 更新告警状态
  GET    /api/v1/alerts/summary           # 告警摘要（注意/警告/严重数量）

历史数据:
  GET    /api/v1/history/voltage          # 指定设备的电压时序（按时间范围）
  GET    /api/v1/history/export           # 导出 CSV

统计报表:
  GET    /api/v1/statistics               # 月度统计、区域分布、异常类型分布

监控概览:
  GET    /api/v1/dashboard/kpi            # KPI指标汇总
  GET    /api/v1/dashboard/alerts/recent  # 最新告警列表
```

---

## 6. 各页面详细设计

### 6.1 登录页 (Login)

- 左右分屏（55% / 45%）
- 左侧：品牌介绍 + 电力线路 SVG 动画 + 统计数字
- 右侧：登录表单（用户名/密码/记住我）
- JWT Token 认证（简化版：硬编码 admin/admin123）
- 响应式：768px 以下变单栏

### 6.2 监控概览 (Dashboard)

- **行1**：4 个 KPI 卡片（在线设备/今日告警/电压合格率/平均功率因数）
- **行2**：24小时三相电压趋势图（60%）+ 最新告警时间线（40%）
- **行3**：异常类型分布甜甜圈（35%）+ 设备状态进度条（65%）

### 6.3 设备管理 (Devices)

- 顶部筛选栏（编号搜索/状态/区域/重置/添加）
- 分页数据表格（9列 + 操作）
- 操作：查看详情弹窗 / 编辑弹窗 / 删除确认

### 6.4 告警管理 (Alerts)

- 顶部摘要卡片（注意/警告/严重数量，左色边框）
- 筛选栏（日期/级别/状态/设备编号/查询/重置）
- 分页表格（告警编号/时间/设备/位置/类型/严重程度/描述/状态/操作）

### 6.5 历史查询 (History)

- 查询条件：设备选择 + 时间范围 + 快捷按钮（24h/7d/30d）
- 数据类型复选框（电压/电流/功率/功率因数）
- ECharts 三相电压折线图 + 上下限标注线
- 数据表格（异常值自动标红）
- CSV 导出功能

### 6.6 统计报表 (Statistics)

- 2x2 四图表布局：
  - 左上：异常类型甜甜圈
  - 右上：月度异常趋势（柱+折线混合）
  - 左下：区域异常排行（水平柱状）
  - 右下：电压质量雷达图

### 6.7 异常检测 (Detect) - 保留增强

- 保留现有 ML 检测功能
- 应用 Ant Design 样式重构 UI
- 上传区 + 结果展示保持不变

### 6.8 模型对比 (Models) - 保留增强

- 保留现有模型性能展示
- 应用 Ant Design 样式重构 UI

---

## 7. 开发计划

### Phase 1：环境准备（1-2小时）
- 安装 Ant Design、替换 Tailwind
- 配置 antd 主题（自定义颜色变量）
- 搭建新的 MainLayout（深色侧边栏）

### Phase 2：后端扩展（2-3小时）
- 新增数据库表（Device/Alert/VoltageReading）
- 实现新增 API 路由
- 生成 mock 种子数据

### Phase 3：前端页面开发（4-6小时）
- 登录页
- 监控概览
- 设备管理
- 告警管理
- 历史查询
- 统计报表

### Phase 4：集成与优化（1-2小时）
- API 对接
- 细节样式调整
- 响应式适配

---

## 8. 成功标准

1. ✅ 所有 Mockup 页面在真实系统中可见且功能可用
2. ✅ 视觉与 Mockup 95%+ 一致
3. ✅ 现有 ML 检测功能正常运行
4. ✅ 后端新增 API 正常返回数据
5. ✅ 登录页面设计精美，可用于论文截图
