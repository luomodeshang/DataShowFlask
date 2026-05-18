from flask import Flask, render_template, send_from_directory
import json, os

app = Flask(__name__)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 数据清洗平台页
@app.route('/preprocess')
def preprocess():
    return render_template('preprocess.html')

# 数据集 JSON 接口
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
