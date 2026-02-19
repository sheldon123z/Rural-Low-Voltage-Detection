# 农村低电压检测平台 Web 应用设计文档

**日期**: 2026-02-18
**版本**: V1.0
**项目路径**: `Rural-Low-Voltage-Detection/webapp/`

---

## 一、项目背景

本系统基于已训练完成的 VoltageTimesNet 模型（F1=0.8149, Recall=0.9110）构建完整的
前后端 Web 应用，实现农村低压配电网电压异常的实时检测与可视化展示。

## 二、技术栈选型

### 后端
- **框架**: FastAPI (Python 3.10+)
- **ML集成**: 直接复用 `code/demo/core/inference.py` 中的 `VoltageAnomalyDetector`
- **数据库**: SQLite（检测历史）
- **ORM**: SQLModel (SQLAlchemy + Pydantic)
- **依赖**: `fastapi`, `uvicorn`, `sqlmodel`, `python-multipart`, `numpy`, `torch`

### 前端
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **UI 组件**: shadcn/ui + Tailwind CSS
- **图表**: Apache ECharts (echarts-for-react)
- **路由**: React Router v6
- **状态管理**: TanStack Query (React Query)
- **HTTP客户端**: Axios

## 三、目录结构

```
webapp/
├── backend/
│   ├── main.py              # FastAPI 应用入口
│   ├── api/
│   │   ├── detect.py        # 检测路由
│   │   ├── models.py        # 模型管理路由
│   │   └── metrics.py       # 系统指标路由
│   ├── services/
│   │   ├── detection.py     # 检测服务（调用推理模块）
│   │   └── model_manager.py # 模型加载管理
│   ├── models/
│   │   └── database.py      # SQLModel 数据模型
│   ├── core/
│   │   └── config.py        # 配置（模型路径、数据集路径）
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx    # 仪表板
│   │   │   ├── Detect.tsx       # 检测页面
│   │   │   ├── History.tsx      # 历史记录
│   │   │   ├── Models.tsx       # 模型对比
│   │   │   └── About.tsx        # 系统说明
│   │   ├── components/
│   │   │   ├── TimeSeriesChart.tsx  # 时序图组件
│   │   │   ├── AnomalyHeatmap.tsx   # 异常热力图
│   │   │   ├── ModelRadarChart.tsx  # 雷达图
│   │   │   └── UploadPanel.tsx      # 上传面板
│   │   ├── api/
│   │   │   └── client.ts        # API 请求封装
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 四、API 接口设计

### 检测接口
```
POST   /api/v1/detect          # 上传CSV文件进行异常检测
GET    /api/v1/detect/{id}     # 获取检测结果
GET    /api/v1/detect/history  # 检测历史列表（分页）
GET    /api/v1/sample-data     # 获取示例数据（内置测试数据）
```

### 模型接口
```
GET    /api/v1/models              # 所有模型列表 + 性能指标
POST   /api/v1/models/{name}/load  # 切换当前使用模型
GET    /api/v1/models/current      # 当前加载的模型信息
```

### 系统接口
```
GET    /api/v1/health          # 健康检查
GET    /api/v1/metrics         # 系统统计（检测次数、平均响应时间等）
```

### 统一响应格式
```json
{
  "code": 200,
  "message": "success",
  "data": { ... },
  "timestamp": "2026-02-18T10:00:00Z"
}
```

## 五、前端页面详细设计

### 1. 仪表板 (/)
- 顶部统计卡片：总检测次数、今日检测次数、平均F1分数、异常发现率
- 中部：最近7天检测趋势折线图（ECharts）
- 右侧：实时模型状态卡片（当前加载模型、内存用量）
- 底部：最近5条检测记录快速预览

### 2. 实时检测 (/detect)
- 左侧面板：
  - CSV文件上传区（drag & drop）
  - 模型选择下拉框（VoltageTimesNet / TimesNet / LSTMAutoEncoder等）
  - 异常比例滑块（anomaly_ratio: 0.5-5.0）
  - "开始检测"按钮
- 右侧结果区：
  - 检测进度条（大文件时显示）
  - 16维特征时序图（可选择展示哪些特征）
  - 异常区间高亮标注（红色背景）
  - 异常分数分布直方图
  - 检测摘要（总时间步、异常数量、异常率、F1参考值）

### 3. 历史记录 (/history)
- 可搜索/筛选的检测历史表格
- 每行：文件名、检测时间、模型、异常率、耗时、操作(查看/下载)
- 点击展开查看完整检测结果图表

### 4. 模型对比 (/models)
- 5个模型的性能指标卡片（VoltageTimesNet、TimesNet、LSTMAutoEncoder、IForest、OC-SVM）
- 多维雷达图：F1、精确率、召回率、准确率
- ROC/PR曲线对比图
- 模型特点文字说明（主模型 VoltageTimesNet 重点展示）

### 5. 系统说明 (/about)
- VoltageTimesNet 架构原理图（SVG动画）
- FFT周期发现原理可视化（动态图）
- 数据集说明（RuralVoltage数据集特征描述）
- 项目论文引用信息

## 六、数据库设计

### detection_task 表
```sql
id           TEXT PRIMARY KEY  -- UUID
filename     TEXT              -- 上传文件名
model_name   TEXT              -- 使用的模型
status       TEXT              -- pending/running/completed/failed
anomaly_ratio REAL             -- 检测参数
created_at   DATETIME          -- 创建时间
completed_at DATETIME          -- 完成时间
result_path  TEXT              -- 结果JSON文件路径
summary      TEXT              -- JSON摘要（异常数、异常率等）
```

## 七、关键实现细节

### 模型集成
- 在 FastAPI 启动时加载 VoltageTimesNet_v2 模型（异步）
- 模型路径：`../../code/newest_models/best_voltagetimesnet_v2.pth`
- 大文件使用 `BackgroundTasks` 异步处理

### 数据处理
- 前端上传 CSV → 后端验证列（需包含16个特征列）
- 数据标准化（使用训练集统计量或在线标准化）
- 滑动窗口推理（seq_len=50）

### 前端性能
- 大数据集时序图虚拟化（只渲染可视区域）
- 图表 debounce 防抖渲染
- React Query 缓存 5分钟内的检测结果

## 八、部署方案

开发模式：
- 后端: `uvicorn main:app --reload --port 8000`
- 前端: `npm run dev`（端口 5173，代理 /api 到 8000）

生产模式：
- 后端: `uvicorn main:app --workers 2 --port 8000`
- 前端: `npm run build` → Nginx 静态服务
- 或使用 Docker Compose 一键启动
