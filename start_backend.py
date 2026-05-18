#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业大数据可视化大屏后端启动脚本
"""

import sys
import io
import os

# 设置标准输出编码为UTF-8（解决Windows中文乱码问题）
if sys.platform == 'win32':
    # Windows系统设置控制台编码
    try:
        # 尝试设置控制台代码页为UTF-8
        os.system('chcp 65001 >nul 2>&1')
        # 设置标准输出编码
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("=" * 50)
    print("工业大数据可视化大屏后端服务")
    print("=" * 50)
    print("服务地址: http://localhost:5000")
    print("API文档: http://localhost:5000/api/")
    print("=" * 50)
    print("可用的API接口:")
    print("- /api/industrial_pie_data - 工业大数据饼图数据")
    print("- /api/vibration - 振动监测数据")
    print("- /api/realtime_vibration - 实时振动监测数据")
    print("- /api/device_status - 设备状态数据")
    print("- /api/production - 产量数据")
    print("- /api/equipment - 设备数据")
    print("- /api/ranking - 生产线排行榜")
    print("- /api/monthly - 月度数据")
    print("- /api/quality_analysis - 质量分析数据")
    print("- /api/production_daily - 每日产量数据")
    print("- /api/realtime - 实时综合数据")
    print("=" * 50)
    print("启动中...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

