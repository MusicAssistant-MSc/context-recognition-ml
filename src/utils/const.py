from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

PANDAS_DF_UNNAMED_COL_PREFIX = 'Unnamed: '

# --- SiTunes dataset constants ---
SITUNES_DATA_DIR = RAW_DATA_DIR / 'SiTunes'
SITUNES_REDUNDANT_COL_SUFFIX = '_redundant'
SITUNES_FLOAT_ROUNDING_PRECISION = 6

SITUNES_MUSIC_METADATA_DIR = SITUNES_DATA_DIR / 'music_metadata'
SITUNES_MUSIC_METADATA_CONFIG = {
    'music_info': {
        'filename': 'music_info.csv',
        'filepath': SITUNES_MUSIC_METADATA_DIR / 'music_info.csv',
        'pkey_candidate_cols': ['item_id'],
        'drop_cols': [],
        'drop_unnamed_cols': True,
    },
    'music_info_withname': {
        'filename': 'music_info_withname.csv',
        'filepath': SITUNES_MUSIC_METADATA_DIR / 'music_info_withname.csv',
        'pkey_candidate_cols': ['i_id_c'],
        'drop_cols': [],
        'drop_unnamed_cols': True,
    },
}

SITUNES_STAGES = ['Stage2', 'Stage3']
SITUNES_STAGE_DATA_CONFIG = {
    'environment': {
        'filename': 'env.json',
        'pkey_col': 'inter_id',
        'cols': {
            'time': ['time_period'],
            'weather': ['weather_type', 'pressure', 'temperature', 'humidity'],
            'GPS': ['longitude', 'latitude', 'speed']
        },
        'rename_cols': {
            'pressure': 'weather_pressure',
            'temperature': 'weather_temp',
            'humidity': 'weather_humidity',
            'longitude': 'gps_lon',
            'latitude': 'gps_lat',
            'speed': 'gps_speed'
        },
        'drop_cols': [],
    },
    'interactions': {
        'filename': 'interactions.csv',
        'pkey_col': 'inter_id'
    },
    'wrist': {
        'filename': 'wrist.npy'
    },
}

