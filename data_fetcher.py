# -*- coding: utf-8 -*-
"""
数据获取模块 - 从本地数据库获取真实数据
移植自 YieldModel/main_yield.py
"""

from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
import pyodbc
import os
import sys
import math
from db_config import build_conn_str, load_db_config

VIBRATION_THRESHOLD = 16.0
NOISE_THRESHOLD = 90.0
TEMPERATURE_THRESHOLD = 5.0

def normalize_ratio(value, threshold):
    if threshold <= 0:
        return 0.0
    return round(max(0.0, value / threshold), 3)

def init_database_index():
    """初始化数据库字段索引"""
    columns = [
        "Speed_X",  # 0
        "Speed_Y",  # 1
        "Speed_Z",  # 2
        "Speed_R",  # 3
        "positioing_error_X",  # 4
        "positioing_error_Y",  # 5
        "acceleration_X",  # 6
        "acceleration_Y",  # 7
        "acceleration_Z",  # 8
        "acceleration_R",  # 9
        "visual_error_X",  # 10
        "visual_error_Y",  # 11
        "fit_error_X",  # 12
        "fit_error_Y",  # 13
        "vibration_X",  # 14
        "vibration_Z",  # 15
        "noise_X",  # 16
        "PLC_step",  # 17
        "absolute_position_X",  # 18
        "absolute_position_Y",  # 19
        "offset_X",  # 20
        "offset_Y",  # 21
        "offset_R",  # 22
        "Grating_feedback_X",  # 23
        "Grating_feedback_Y",  # 24
        "setting_red_circle_X",  # 25
        "setting_red_circle_Y",  # 26
        "setting_blue_circle_X",  # 27
        "setting_blue_circle_Y",  # 28
        "setting_yellow_square_X",  # 29
        "setting_yellow_square_Y",  # 30
        "setting_blue_square_X",  # 31
        "setting_blue_square_Y",  # 32
        "visual_scanning_pixel_coordinate_X",  # 33
        "visual_scanning_pixel_coordinate_Y",  # 34
        "visual_setting_pixel_coordinate_X",  # 35
        "visual_setting_pixel_coordinate_Y",  # 36
        "visual_scanning_world_coordinate_X",  # 37
        "visual_scanning_world_coordinate_Y"  # 38
    ]
    return columns

def get_naZero_data(List, Column_item, top, bottom):
    """找到二位列表中固定列某一行段中非零值"""
    for i in range(bottom - top + 1):
        if List[i + top][Column_item] != 0:
            return List[i + top][Column_item]
    return -1

def Get_NaZero_data_From_especific_PLC_step(List, Column_item, PLC_step):
    """从分割好的单个工件步骤List里获取对应的PLC_step范围内首个非零值"""
    for i in range(len(List)):
        if List[i][17] == PLC_step and List[i][Column_item] != 0:
            return List[i][Column_item]
    return -1

def get_Most_data(List):
    """获取列表中出现次数最多的值"""
    if not List:
        return None
    count = Counter(List)
    max_count = max(count.values())
    modes = [num for num, cnt in count.items() if cnt == max_count]
    return modes if len(modes) > 1 else modes[0]

def init_hash_for_PLC_step():
    """为每个工件建立哈希表,存储完整的PLC步骤"""
    PLC_SETP_LEN = 12
    Hash_List_for_PLC = []
    for i in range(PLC_SETP_LEN):
        Hash_List_for_PLC.append([])
    return Hash_List_for_PLC

def Connect_SQL(begin_time=None, end_time=None, table_name='BK_DataModel_Immediate'):
    """
    连接SQL Server数据库并获取数据
    支持从 BK_DataModel 或 BK_DataModel_Immediate 表获取数据
    """
    try:
        config = load_db_config(os.path.dirname(__file__))
        database = config.get('database', 'DataAnalysis')
        conn_str = build_conn_str(config, database_key='database')
        
        # 连接到数据库
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # 构建SQL查询
        sql = f'''SELECT 
              Speed_X,
              Speed_Y,
              Speed_Z,
              Speed_R,
              positioing_error_X,
              positioing_error_Y,
              acceleration_X,
              acceleration_Y,
              acceleration_Z,
              acceleration_R,
              visual_error_X,
              visual_error_Y,
              fit_error_X,
              fit_error_Y,
              vibration_X,
              vibration_Z,
              noise_X,
              PLC_step,
              absolute_position_X,
              absolute_position_Y,
              offset_X,
              offset_Y,
              offset_R,
              Grating_feedback_X,
              Grating_feedback_Y,
              setting_red_circle_X,
              setting_red_circle_Y,
              setting_blue_circle_X,
              setting_blue_circle_Y,
              setting_yellow_square_X,
              setting_yellow_square_Y,
              setting_blue_square_X,
              setting_blue_square_Y,
              visual_scanning_pixel_coordinate_X,
              visual_scanning_pixel_coordinate_Y,
              visual_setting_pixel_coordinate_X,
              visual_setting_pixel_coordinate_Y,
              visual_scanning_world_coordinate_X,
              visual_scanning_world_coordinate_Y,
              creation_date
              FROM {database}.DBO.{table_name}'''
        
        # 添加时间范围条件
        params = []
        if begin_time and end_time:
            sql += " WHERE creation_date BETWEEN ? AND ?"
            params.extend([begin_time, end_time])
        
        # 执行查询
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        
        rows = cursor.fetchall()
        conn.close()
        return rows
        
    except Exception as e:
        print(f"数据库连接错误: {type(e)} {e}")
        return None

Yield_columns = [
    'all定位误差X-1', 'all定位误差Y-1', 'all定位误差X-2', 'all定位误差Y-2',
    'all定位误差X-3', 'all定位误差Y-3', 'all定位误差X-4', 'all定位误差Y-4',
    'all物件坐标X-1', 'all物件坐标Y-1', 'all物理坐标X-1', 'all物理坐标Y-1',
    'catch定位误差X(移动到拍照位)', 'catch定位误差Y(移动到拍照位)',
    'catch光栅反馈X(移动到拍照位)', 'catch光栅反馈Y(移动到拍照位)',
    '视觉误差X', '视觉误差Y',
    'catch复拍物件坐标X', 'catch复拍物件坐标Y', 'catch复拍物理坐标X', 'catch复拍物理坐标Y',
    'catchPLC坐标X', 'catchPLC坐标Y',
    'catch定位误差X', 'catch定位误差Y',
    'catch光栅反馈X', 'catch光栅反馈Y',
    'releasePLC坐标X', 'releasePLC坐标Y', 'release定位误差X', 'release定位误差Y',
    'release光栅反馈X', 'release光栅反馈Y', '贴合误差X', '贴合误差Y',
    'release视觉坐标X(像素)', 'release视觉坐标Y(像素)'
]

Yield_index = {
    'all定位误差X-1': 0, 'all定位误差Y-1': 1, 'all定位误差X-2': 2, 'all定位误差Y-2': 3,
    'all定位误差X-3': 4, 'all定位误差Y-3': 5, 'all定位误差X-4': 6, 'all定位误差Y-4': 7,
    'all物件坐标X-1': 8, 'all物件坐标Y-1': 9, 'all物理坐标X-1': 10, 'all物理坐标Y-1': 11,
    'catch定位误差X(移动到拍照位)': 12, 'catch定位误差Y(移动到拍照位)': 13,
    'catch光栅反馈X(移动到拍照位)': 14, 'catch光栅反馈Y(移动到拍照位)': 15,
    '视觉误差X': 16, '视觉误差Y': 17,
    'catch复拍物件坐标X': 18, 'catch复拍物件坐标Y': 19, 'catch复拍物理坐标X': 20, 'catch复拍物理坐标Y': 21,
    'catchPLC坐标X': 22, 'catchPLC坐标Y': 23,
    'catch定位误差X': 24, 'catch定位误差Y': 25,
    'catch光栅反馈X': 26, 'catch光栅反馈Y': 27,
    'releasePLC坐标X': 28, 'releasePLC坐标Y': 29, 'release定位误差X': 30, 'release定位误差Y': 31,
    'release光栅反馈X': 32, 'release光栅反馈Y': 33, '贴合误差X': 34, '贴合误差Y': 35,
    'release视觉坐标X(像素)': 36, 'release视觉坐标Y(像素)': 37
}

def Cut_data_to_Single_workpiece(Origin_data):
    """将原始数据切割为单个工件的数据"""
    Origin_Result = []
    Single_work_PLC = []
    PLC_Step = 17
    write_Flag = False
    Last_PCL_Step = -1
    for item_data in Origin_data:
        if item_data[PLC_Step] == 2.1 and write_Flag == False:
            write_Flag = True
            Single_work_PLC.append(item_data)
        if item_data[12] != 0 and write_Flag == True:
            write_Flag = False
            Single_work_PLC.append(item_data)
            Origin_Result.append(Single_work_PLC)
            Single_work_PLC = []
            Single_work_PLC.append(item_data)
        if write_Flag == True:
            Single_work_PLC.append(item_data)
        Last_PCL_Step = item_data[PLC_Step]
    return Origin_Result

def index_area_For_1_4(Origin_data):
    """定位初始拍照索引"""
    result = [-1, -1, -1, -1, -1]
    for index in range(len(Origin_data)):
        if Origin_data[index][17] == 1.1 and result[0] == -1:
            result[0] = index
        if Origin_data[index][17] == 1.2 and result[1] == -1:
            result[1] = index
        if Origin_data[index][17] == 1.3 and result[2] == -1:
            result[2] = index
        if Origin_data[index][17] == 1.4 and result[3] == -1:
            result[3] = index
        if Origin_data[index][17] == 2.1 and result[4] == -1:
            result[4] = index - 1
    return result

def Output_normalize_PLC_single_workpiece(begin_time_str, end_time_str):
    """
    给单工件输出规范数据格式
    这是核心的数据处理函数，从数据库获取数据并规范化
    """
    try:
        begin_time = datetime.strptime(begin_time_str, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    except:
        # 尝试其他时间格式
        try:
            begin_time = datetime.strptime(begin_time_str, '%Y-%m-%d %H:%M')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
        except:
            print(f"时间格式错误: {begin_time_str}, {end_time_str}")
            return []
    
    print(f"正在处理数据: {begin_time} 到 {end_time}")
    
    columns_mapping = {
        'Speed_X': 0, 'Speed_Y': 1, 'Speed_Z': 2, 'Speed_R': 3,
        'positioing_error_X': 4, 'positioing_error_Y': 5,
        'acceleration_X': 6, 'acceleration_Y': 7, 'acceleration_Z': 8, 'acceleration_R': 9,
        'visual_error_X': 10, 'visual_error_Y': 11,
        'fit_error_X': 12, 'fit_error_Y': 13,
        'vibration_X': 14, 'vibration_Z': 15,
        'noise_X': 16, 'PLC_step': 17,
        'absolute_position_X': 18, 'absolute_position_Y': 19,
        'offset_X': 20, 'offset_Y': 21, 'offset_R': 22,
        'Grating_feedback_X': 23, 'Grating_feedback_Y': 24,
        'setting_red_circle_X': 25, 'setting_red_circle_Y': 26,
        'setting_blue_circle_X': 27, 'setting_blue_circle_Y': 28,
        'setting_yellow_square_X': 29, 'setting_yellow_square_Y': 30,
        'setting_blue_square_X': 31, 'setting_blue_square_Y': 32,
        'visual_scanning_pixel_coordinate_X': 33, 'visual_scanning_pixel_coordinate_Y': 34,
        'visual_setting_pixel_coordinate_X': 35, 'visual_setting_pixel_coordinate_Y': 36,
        'visual_scanning_world_coordinate_X': 37, 'visual_scanning_world_coordinate_Y': 38,
        'creation_date': 39
    }
    
    Final_Result = []
    Origin_Result = Connect_SQL(
        begin_time.strftime('%Y-%m-%d %H:%M:%S'),
        end_time.strftime('%Y-%m-%d %H:%M:%S'),
        table_name='BK_DataModel_Immediate'
    )
    
    if not Origin_Result:
        print("警告: 未获取到数据，返回空列表")
        return []
    
    db_index = init_database_index()
    Single_data_List = Cut_data_to_Single_workpiece(Origin_Result)
    
    Lenth = len(Single_data_List)
    print(f"工件个数为: {Lenth}")
    
    for i_num in range(Lenth):
        Final_Result.append([0] * 38)
    
    # 处理all工件误差
    Last_PLC = 0.0
    temp = []
    for num in range(len(Origin_Result)):
        if Origin_Result[num][columns_mapping['PLC_step']] == 1.1 and Last_PLC != 1.1:
            temp.append(Origin_Result[num][columns_mapping['positioing_error_X']])
            temp.append(Origin_Result[num][columns_mapping['positioing_error_Y']])
        if Origin_Result[num][columns_mapping['PLC_step']] == 1.2 and Last_PLC != 1.2:
            temp.append(Origin_Result[num][columns_mapping['positioing_error_X']])
            temp.append(Origin_Result[num][columns_mapping['positioing_error_Y']])
        if Origin_Result[num][columns_mapping['PLC_step']] == 1.3 and Last_PLC != 1.3:
            temp.append(Origin_Result[num][columns_mapping['positioing_error_X']])
            temp.append(Origin_Result[num][columns_mapping['positioing_error_Y']])
        if Origin_Result[num][columns_mapping['PLC_step']] == 1.4 and Last_PLC != 1.4:
            temp.append(Origin_Result[num][columns_mapping['positioing_error_X']])
            temp.append(Origin_Result[num][columns_mapping['positioing_error_Y']])
        Last_PLC = Origin_Result[num][columns_mapping['PLC_step']]
    
    for index in range(Lenth):
        if len(temp) >= 8:
            Final_Result[index][Yield_index['all定位误差X-1']] = temp[0]
            Final_Result[index][Yield_index['all定位误差Y-1']] = temp[1]
            Final_Result[index][Yield_index['all定位误差X-2']] = temp[2]
            Final_Result[index][Yield_index['all定位误差Y-2']] = temp[3]
            Final_Result[index][Yield_index['all定位误差X-3']] = temp[4]
            Final_Result[index][Yield_index['all定位误差Y-3']] = temp[5]
            Final_Result[index][Yield_index['all定位误差X-4']] = temp[6]
            Final_Result[index][Yield_index['all定位误差Y-4']] = temp[7]
    
    origin_photo_setting = index_area_For_1_4(Origin_Result)
    print(f"初始拍照位: {origin_photo_setting}")
    
    for index in range(0, min(4, Lenth)):
        Final_Result[index][Yield_index['all物件坐标X-1']] = get_naZero_data(
            Origin_Result, columns_mapping['visual_scanning_pixel_coordinate_X'],
            origin_photo_setting[index], origin_photo_setting[index + 1])
        Final_Result[index][Yield_index['all物件坐标Y-1']] = get_naZero_data(
            Origin_Result, columns_mapping['visual_scanning_pixel_coordinate_Y'],
            origin_photo_setting[index], origin_photo_setting[index + 1])
        Final_Result[index][Yield_index['all物理坐标X-1']] = get_naZero_data(
            Origin_Result, columns_mapping['visual_scanning_world_coordinate_X'],
            origin_photo_setting[index], origin_photo_setting[index + 1])
        Final_Result[index][Yield_index['all物理坐标Y-1']] = get_naZero_data(
            Origin_Result, columns_mapping['visual_scanning_world_coordinate_Y'],
            origin_photo_setting[index], origin_photo_setting[index + 1])
    
    # catch定位误差和光栅反馈
    Last_PLC = 0.0
    workpiece = -1
    for index in range(len(Origin_Result)):
        if Last_PLC == 2.1 and Origin_Result[index][columns_mapping['PLC_step']] == 2.2:
            workpiece += 1
            if workpiece < Lenth:
                Final_Result[workpiece][Yield_index['catch定位误差X(移动到拍照位)']] = Origin_Result[index][columns_mapping['positioing_error_X']]
                Final_Result[workpiece][Yield_index['catch定位误差Y(移动到拍照位)']] = Origin_Result[index][columns_mapping['positioing_error_Y']]
                Final_Result[workpiece][Yield_index['catch光栅反馈X(移动到拍照位)']] = Origin_Result[index][columns_mapping['Grating_feedback_X']]
                Final_Result[workpiece][Yield_index['catch光栅反馈Y(移动到拍照位)']] = Origin_Result[index][columns_mapping['Grating_feedback_Y']]
        Last_PLC = Origin_Result[index][columns_mapping['PLC_step']]
    
    # 视觉误差和其他数据
    for index in range(len(Single_data_List)):
        if index >= Lenth:
            break
        Final_Result[index][Yield_index['视觉误差X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_error_X"], 2.2)
        Final_Result[index][Yield_index['视觉误差Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_error_Y"], 2.2)
        Final_Result[index][Yield_index['catch复拍物件坐标X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_scanning_pixel_coordinate_X"], 2.2)
        Final_Result[index][Yield_index['catch复拍物件坐标Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_scanning_pixel_coordinate_Y"], 2.2)
        Final_Result[index][Yield_index['catch复拍物理坐标X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_scanning_world_coordinate_X"], 2.2)
        Final_Result[index][Yield_index['catch复拍物理坐标Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["visual_scanning_world_coordinate_Y"], 2.2)
        Final_Result[index][Yield_index['catchPLC坐标X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["absolute_position_X"], 2.2)
        Final_Result[index][Yield_index['catchPLC坐标Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["absolute_position_Y"], 2.2)
        Final_Result[index][Yield_index['catch光栅反馈X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["Grating_feedback_X"], 2.3)
        Final_Result[index][Yield_index['catch光栅反馈Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["Grating_feedback_Y"], 2.3)
        Final_Result[index][Yield_index['catch定位误差X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["positioing_error_X"], 2.3)
        Final_Result[index][Yield_index['catch定位误差Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["positioing_error_Y"], 2.3)
        Final_Result[index][Yield_index['releasePLC坐标X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["absolute_position_X"], 2.4)
        Final_Result[index][Yield_index['releasePLC坐标Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["absolute_position_Y"], 2.4)
        Final_Result[index][Yield_index['release定位误差X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["positioing_error_X"], 2.5)
        Final_Result[index][Yield_index['release定位误差Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["positioing_error_Y"], 2.5)
        Final_Result[index][Yield_index['release光栅反馈X']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["Grating_feedback_X"], 2.5)
        Final_Result[index][Yield_index['release光栅反馈Y']] = Get_NaZero_data_From_especific_PLC_step(
            Single_data_List[index], columns_mapping["Grating_feedback_Y"], 2.5)
        
        if len(Single_data_List[index]) > 0:
            last_item = Single_data_List[index][-1]
            Final_Result[index][Yield_index['贴合误差X']] = last_item[columns_mapping["fit_error_X"]]
            Final_Result[index][Yield_index['贴合误差Y']] = last_item[columns_mapping["fit_error_Y"]]
            Final_Result[index][Yield_index['release视觉坐标X(像素)']] = last_item[columns_mapping["visual_setting_pixel_coordinate_X"]]
            Final_Result[index][Yield_index['release视觉坐标Y(像素)']] = last_item[columns_mapping["visual_setting_pixel_coordinate_Y"]]
    
    return Final_Result

def calculate_yield_from_data(normalized_data):
    """
    从规范化数据计算良率统计
    基于贴合误差计算良率（误差>55为不良品）
    """
    if not normalized_data or len(normalized_data) == 0:
        return None
    
    total_count = len(normalized_data)
    defect_count = 0
    
    for item in normalized_data:
        # 获取贴合误差（索引34和35）
        fit_error_x = abs(item[34]) if len(item) > 34 and item[34] is not None else 0
        fit_error_y = abs(item[35]) if len(item) > 35 and item[35] is not None else 0
        
        # 如果任一方向的误差超过55，认为是缺陷
        if fit_error_x > 55 or fit_error_y > 55:
            defect_count += 1
    
    quality_count = total_count - defect_count
    yield_rate = (quality_count / total_count * 100) if total_count > 0 else 0
    
    return {
        'total': total_count,
        'defect': defect_count,
        'quality': quality_count,
        'yield_rate': round(yield_rate, 2)
    }

def get_latest_realtime_data():
    """获取最新的实时数据"""
    try:
        # 获取最近5秒的数据
        end_time = datetime.now()
        begin_time = end_time - timedelta(seconds=5)
        
        rows = Connect_SQL(
            begin_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S'),
            table_name='BK_DataModel'
        )
        
        if not rows or len(rows) == 0:
            return None
        
        # 返回最新的一条数据
        columns = init_database_index() + ['creation_date']
        latest_row = rows[-1]
        
        data_dict = {}
        for i, col in enumerate(columns):
            if i < len(latest_row):
                data_dict[col] = latest_row[i]
        
        return data_dict
    except Exception as e:
        print(f"获取实时数据失败: {e}")
        return None

def calculate_production_stats_from_db(days=1):
    """
    从数据库计算产量统计数据
    基于规范化数据中的工件数量计算
    """
    try:
        end_time = datetime.now()
        begin_time = end_time - timedelta(days=days)
        
        normalized_data = Output_normalize_PLC_single_workpiece(
            begin_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        if not normalized_data or len(normalized_data) == 0:
            return None
        
        # 计算良品和不良品数量
        total_production = len(normalized_data)
        quality_count = 0
        defect_count = 0
        
        for item in normalized_data:
            if len(item) > 34:
                fit_error_x = abs(item[34]) if item[34] is not None else 0
                fit_error_y = abs(item[35]) if item[35] is not None else 0
                
                # 误差小于等于55为良品
                if fit_error_x <= 55 and fit_error_y <= 55:
                    quality_count += 1
                else:
                    defect_count += 1
        
        quality_rate = (quality_count / total_production * 100) if total_production > 0 else 0
        
        return {
            "total_production": total_production,
            "quality_products": quality_count,
            "quality_rate": round(quality_rate, 1),
            "defect_count": defect_count
        }
    except Exception as e:
        print(f"计算产量统计失败: {e}")
        return None

def get_vibration_history_from_db(days=7):
    """
    从数据库获取振动历史数据
    """
    try:
        end_time = datetime.now()
        begin_time = end_time - timedelta(days=days)
        
        rows = Connect_SQL(
            begin_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S'),
            table_name='BK_DataModel'
        )
        
        if not rows or len(rows) == 0:
            return []
        
        # 按日期分组
        date_dict = {}
        for row in rows:
            if len(row) > 39:
                date = row[39]
                if isinstance(date, str):
                    try:
                        date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
                        except:
                            continue
                else:
                    date_obj = date
                
                date_key = date_obj.strftime('%m/%d')
                
                if date_key not in date_dict:
                    date_dict[date_key] = {
                        'vibrations_x': [],
                        'vibrations_z': [],
                        'noises': []
                    }
                
                if len(row) > 16:
                    if row[14] is not None and row[14] != 0:
                        date_dict[date_key]['vibrations_x'].append(float(row[14]))
                    if row[15] is not None and row[15] != 0:
                        date_dict[date_key]['vibrations_z'].append(float(row[15]))
                    if row[16] is not None and row[16] != 0:
                        date_dict[date_key]['noises'].append(float(row[16]))
        
        # 转换为列表格式
        result = []
        for i in range(days):
            date = end_time - timedelta(days=days-1-i)
            date_key = date.strftime('%m/%d')
            
            if date_key in date_dict:
                values = date_dict[date_key]
                avg_noise = int(sum(values['noises']) / len(values['noises'])) if values['noises'] else 0
                avg_x = round(sum(values['vibrations_x']) / len(values['vibrations_x']), 3) if values['vibrations_x'] else 0.0
                avg_z = round(sum(values['vibrations_z']) / len(values['vibrations_z']), 3) if values['vibrations_z'] else 0.0
                temperature = 0.0
                result.append({
                    "date": date_key,
                    "noise": avg_noise,
                    "x_vibration": avg_x,
                    "z_vibration": avg_z,
                    "temperature": temperature,
                    "noise_ratio": normalize_ratio(avg_noise, NOISE_THRESHOLD),
                    "x_vibration_ratio": normalize_ratio(avg_x, VIBRATION_THRESHOLD),
                    "z_vibration_ratio": normalize_ratio(avg_z, VIBRATION_THRESHOLD),
                    "temperature_ratio": normalize_ratio(temperature or 1.0, TEMPERATURE_THRESHOLD)
                })
            else:
                result.append({
                    "date": date_key,
                    "noise": 0,
                    "x_vibration": 0.0,
                    "z_vibration": 0.0,
                    "temperature": 0.0,
                    "noise_ratio": 0.0,
                    "x_vibration_ratio": 0.0,
                    "z_vibration_ratio": 0.0,
                    "temperature_ratio": 0.0
                })
        
        return result
    except Exception as e:
        print(f"获取振动历史数据失败: {e}")
        return []
