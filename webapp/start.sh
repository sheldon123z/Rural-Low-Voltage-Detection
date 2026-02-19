#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  农村低电压检测平台"
echo "  Rural Low-Voltage Detection Platform"
echo "=========================================="
echo ""
echo "后端 API:  http://localhost:8000"
echo "前端界面:  http://localhost:5173"
echo "API 文档:  http://localhost:8000/docs"
echo ""

# 启动后端
echo "[1/2] 启动 FastAPI 后端..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

echo "      等待后端就绪..."
sleep 4

# 检查后端健康
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "      后端已就绪"
else
    echo "      后端未响应，请检查日志"
fi

# 启动前端
echo "[2/2] 启动 React 前端..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "系统启动完成！"
echo ""
echo "  打开浏览器访问: http://localhost:5173"
echo ""
echo "按 Ctrl+C 停止所有服务..."
trap "echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '已停止'" INT TERM
wait
