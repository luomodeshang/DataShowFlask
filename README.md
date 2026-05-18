# DataShowFlask

工业大数据可视化大屏 + 数据预处理平台

## 快速启动

```bash
pip install -r requirements.txt
python app.py
```

访问 http://localhost:5000

## 项目结构

```
DataShowFlask/
├── app.py                 # Flask 后端（API + 路由）
├── data_fetcher.py        # 数据库数据获取模块
├── db_config.py           # 数据库配置管理
├── db_config.json         # 数据库连接配置
├── data/                  # 测试数据集（20个）
├── templates/
│   ├── index.html         # 首页
│   └── preprocess.html    # 数据预处理平台
├── static/
│   └── css/style.css
├── start_backend.py       # 启动脚本（备选）
├── deploy_linux.sh        # Linux 部署脚本
├── deploy_windows.bat     # Windows 部署脚本
├── start_services.sh      # Linux 一键启动（后端+前端）
├── start_services.bat     # Windows 一键启动（后端+前端）
└── requirements.txt
```

## API 接口

| 路由 | 说明 |
|------|------|
| `/` | 首页 |
| `/preprocess` | 数据预处理平台 |
| `/api/production` | 产量数据 |
| `/api/equipment` | 设备数据 |
| `/api/ranking` | 生产线排行榜 |
| `/api/monthly` | 月度数据 |
| `/api/vibration` | 振动监测数据 |
| `/api/realtime_vibration` | 实时振动监测数据 |
| `/api/device_status` | 设备状态 |
| `/api/device_efficiency` | 设备效率 |
| `/api/quality_analysis` | 质量分析 |
| `/api/production_daily` | 每日产量 |
| `/api/industrial_pie_data` | 工业大数据饼图 |
| `/api/maintenance/realtime` | 运维实时数据 |
| `/api/maintenance/predict` | 预测预警 |
| `/api/maintenance/diagnosis` | 故障诊断 |
