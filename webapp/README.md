# 农村低电压检测平台

基于 VoltageTimesNet 的农村低压配电网电压异常检测 Web 系统。

## 功能

- **总览仪表板** - 系统指标、模型性能雷达图、最近检测记录
- **异常检测** - CSV 文件上传 / 示例数据，实时检测，时序图可视化
- **检测历史** - 分页浏览所有检测记录
- **模型对比** - 5 个模型性能对比（F1柱状图、雷达图）
- **系统原理** - FFT 周期发现可视化、VoltageTimesNet 创新点说明

## 快速启动

```bash
# 安装后端依赖（首次）
cd backend
pip install -r requirements.txt
cd ..

# 安装前端依赖（首次）
cd frontend
npm install
cd ..

# 启动系统
./start.sh
```

## 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:5173 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |

## 技术栈

- **后端**: Python FastAPI + SQLModel + SQLite
- **前端**: React 18 + TypeScript + Vite + shadcn/ui + ECharts
- **模型**: VoltageTimesNet (F1=0.8149, Recall=91.1%)
