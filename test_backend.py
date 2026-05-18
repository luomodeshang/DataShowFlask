#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试后端服务是否能正常启动
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

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("=" * 50)
    print("测试后端服务启动")
    print("=" * 50)
    
    # 测试导入
    print("\n1. 测试模块导入...")
    from app import app
    print("   ✓ 模块导入成功")
    
    # 测试模型加载
    print("\n2. 检查模型文件...")
    model_files = [
        'trained_models/model_x.pkl',
        'trained_models/model_y.pkl',
        'trained_models/features.txt'
    ]
    for file in model_files:
        if os.path.exists(file):
            print(f"   ✓ {file} 存在")
        else:
            print(f"   ⚠ {file} 不存在（将使用模拟数据）")
    
    # 测试配置文件
    print("\n3. 检查配置文件...")
    config_paths = [
        '新建文本文档.ini',
        'YieldModel/新建文本文档.ini'
    ]
    found = False
    for path in config_paths:
        if os.path.exists(path):
            print(f"   ✓ 找到配置文件: {path}")
            found = True
            break
    if not found:
        print("   ⚠ 未找到配置文件（将使用模拟数据）")
    
    # 测试Flask应用
    print("\n4. 测试Flask应用...")
    with app.test_client() as client:
        # 测试一个简单的API
        response = client.get('/api/production')
        if response.status_code == 200:
            print("   ✓ API测试成功")
        else:
            print(f"   ⚠ API返回状态码: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("测试完成！后端服务应该可以正常启动")
    print("=" * 50)
    print("\n启动命令: python app.py")
    print("服务地址: http://localhost:5000")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

