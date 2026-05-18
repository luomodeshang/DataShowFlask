@echo off
chcp 65001
echo ========================================
echo 工业大数据可视化大屏 - Windows部署脚本
echo ========================================

echo.
echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo 正在检查Node.js环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Node.js，请先安装Node.js 14+
    pause
    exit /b 1
)

echo.
echo 正在创建Python虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    if errorlevel 1 (
        echo 错误: 创建虚拟环境失败
        pause
        exit /b 1
    )
)

echo.
echo 正在激活虚拟环境并安装Python依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 安装Python依赖失败
    pause
    exit /b 1
)

echo.
echo 正在安装前端依赖...
cd big-screen-vue-datav
if exist node_modules (
    echo 前端依赖已存在，跳过安装
) else (
    npm install
    if errorlevel 1 (
        echo 错误: 安装前端依赖失败
        pause
        exit /b 1
    )
)

echo.
echo 正在构建前端项目...
npm run build
if errorlevel 1 (
    echo 错误: 构建前端项目失败
    pause
    exit /b 1
)

cd ..

echo.
echo ========================================
echo 部署完成！
echo ========================================
echo.
echo 启动服务：
echo 1. 后端服务: python start_backend.py
echo 2. 前端服务: cd big-screen-vue-datav && npm run serve
echo.
echo 访问地址：
echo - 前端界面: http://localhost:8080
echo - 后端API: http://localhost:5000
echo - 运维平台: http://localhost:8080/maintenance
echo.
pause

