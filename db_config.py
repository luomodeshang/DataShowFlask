import json
import os
from configparser import ConfigParser

CONFIG_JSON = 'db_config.json'
LEGACY_INI = '新建文本文档.ini'

DEFAULT_DB_CONFIG = {
    'server': 'DESKTOP-MIV9PND',
    'database': 'DataAnalysis',
    'original_database': 'DataAnalysis',
    'cleaned_database': 'DataAnalysis',
    'table_original': 'BK_DataModel',
    'table_cleaned': 'BK_DataModel_Immediate',
    'driver': 'ODBC Driver 17 for SQL Server',
    'trusted_connection': True,
}


def _normalize_config(raw):
    config = dict(DEFAULT_DB_CONFIG)
    if not isinstance(raw, dict):
        return config

    mapping = {
        'server': ['server', 'Server'],
        'database': ['database', 'Database'],
        'original_database': ['original_database', 'OriginalDatabase', 'database_original', 'DatabaseOriginal'],
        'cleaned_database': ['cleaned_database', 'CleanedDatabase', 'database_cleaned', 'DatabaseCleaned'],
        'table_original': ['table_original', 'OriginalTable', 'tableOriginal'],
        'table_cleaned': ['table_cleaned', 'CleanedTable', 'tableCleaned'],
        'driver': ['driver', 'Driver'],
        'trusted_connection': ['trusted_connection', 'TrustedConnection'],
    }

    for key, aliases in mapping.items():
        for alias in aliases:
            value = raw.get(alias)
            if value not in (None, ''):
                config[key] = value
                break

    if not config['original_database']:
        config['original_database'] = config['database']
    if not config['cleaned_database']:
        config['cleaned_database'] = config['original_database']
    if not config['database']:
        config['database'] = config['original_database']

    return config


def _load_legacy_ini(path):
    parser = ConfigParser()
    parser.read(path, encoding='utf-8')
    if not parser.has_section('DEFAULT') and 'DEFAULT' not in parser:
        return {}

    return {
        'server': parser.get('DEFAULT', 'Server', fallback=''),
        'database': parser.get('DEFAULT', 'Database', fallback=''),
        'original_database': parser.get('DEFAULT', 'OriginalDatabase', fallback=parser.get('DEFAULT', 'Database', fallback='')),
        'cleaned_database': parser.get('DEFAULT', 'CleanedDatabase', fallback=parser.get('DEFAULT', 'Database', fallback='')),
        'table_original': parser.get('DEFAULT', 'OriginalTable', fallback=''),
        'table_cleaned': parser.get('DEFAULT', 'CleanedTable', fallback=''),
    }


def load_db_config(base_dir=None):
    base_dir = base_dir or os.path.dirname(__file__)
    json_path = os.path.join(base_dir, CONFIG_JSON)
    ini_path = os.path.join(base_dir, LEGACY_INI)

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return _normalize_config(json.load(f))
        except Exception:
            pass

    if os.path.exists(ini_path):
        try:
            return _normalize_config(_load_legacy_ini(ini_path))
        except Exception:
            pass

    return dict(DEFAULT_DB_CONFIG)


def save_db_config(config, base_dir=None):
    base_dir = base_dir or os.path.dirname(__file__)
    json_path = os.path.join(base_dir, CONFIG_JSON)
    merged = _normalize_config(config)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def build_conn_str(config, database_key='database'):
    conf = _normalize_config(config)
    database = conf.get(database_key) or conf.get('database') or conf.get('original_database')
    return (
        f"DRIVER={{{conf.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
        f"SERVER={conf['server']};"
        f"DATABASE={database};"
        'Trusted_Connection=yes;'
    )
