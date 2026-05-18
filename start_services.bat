@echo off
chcp 65001
echo ========================================
echo 启动工业大数据可视化大屏服务
echo ========================================

echo.
echo 正在启动后端服务...
start "后端服务" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && python start_backend.py"

echo.
echo 等待后端服务启动...
timeout /t 3 /nobreak >nul

echo.
echo 正在启动前端服务...
start "前端服务" cmd /k "cd /d %~dp0\big-screen-vue-datav && npm run serve"

echo.
echo ========================================
echo 服务启动完成！
echo ========================================
echo.
echo 访问地址：
echo - 前端界面: http://localhost:8080
echo - 后端API: http://localhost:5000
echo - 运维平台: http://localhost:8080/maintenance
echo.
echo 按任意键关闭此窗口...
pause >nul
