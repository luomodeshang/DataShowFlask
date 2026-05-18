#!/bin/bash

echo "========================================"
echo "工业大数据可视化大屏 - Linux部署脚本"
echo "========================================"

# 检查Python环境
echo ""
echo "正在检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查Node.js环境
echo ""
echo "正在检查Node.js环境..."
if ! command -v node &> /dev/null; then
    echo "错误: 未找到Node.js，请先安装Node.js 14+"
    exit 1
fi

# 创建Python虚拟环境
echo ""
echo "正在创建Python虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败"
        exit 1
    fi
fi

# 激活虚拟环境并安装Python依赖
echo ""
echo "正在激活虚拟环境并安装Python依赖..."
source venv/bin/activate
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: 安装Python依赖失败"
    exit 1
fi

# 安装前端依赖
echo ""
echo "正在安装前端依赖..."
cd big-screen-vue-datav
if [ -d "node_modules" ]; then
    echo "前端依赖已存在，跳过安装"
else
    npm install
    if [ $? -ne 0 ]; then
        echo "错误: 安装前端依赖失败"
        exit 1
    fi
fi

# 构建前端项目
echo ""
echo "正在构建前端项目..."
npm run build
if [ $? -ne 0 ]; then
    echo "错误: 构建前端项目失败"
    exit 1
fi

cd ..

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "启动服务："
echo "1. 后端服务: python start_backend.py"
echo "2. 前端服务: cd big-screen-vue-datav && npm run serve"
echo ""
echo "访问地址："
echo "- 前端界面: http://localhost:8080"
echo "- 后端API: http://localhost:5000"
echo "- 运维平台: http://localhost:8080/maintenance"
echo ""

