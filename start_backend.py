#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后端启动脚本（简化版 — 等同于 python app.py）"""
from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
