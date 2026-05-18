#!/bin/bash

echo "========================================"
echo "启动工业大数据可视化大屏服务"
echo "========================================"

echo ""
echo "正在启动后端服务..."
gnome-terminal -- bash -c "source venv/bin/activate && python start_backend.py; exec bash" 2>/dev/null || \
xterm -e "source venv/bin/activate && python start_backend.py" 2>/dev/null || \
echo "请手动启动后端服务: source venv/bin/activate && python start_backend.py"

echo ""
echo "等待后端服务启动..."
sleep 3

echo ""
echo "正在启动前端服务..."
gnome-terminal -- bash -c "cd big-screen-vue-datav && npm run serve; exec bash" 2>/dev/null || \
xterm -e "cd big-screen-vue-datav && npm run serve" 2>/dev/null || \
echo "请手动启动前端服务: cd big-screen-vue-datav && npm run serve"

echo ""
echo "========================================"
echo "服务启动完成！"
echo "========================================"
echo ""
echo "访问地址："
echo "- 前端界面: http://localhost:8080"
echo "- 后端API: http://localhost:5000"
echo "- 运维平台: http://localhost:8080/maintenance"
echo ""

