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

class SitunesStageDataKeys:
    ENVIRONMENT: str = 'environment'
    INTERACTIONS: str = 'interactions'
    WRIST: str = 'wrist'

class SitunesConfigKeys:
    FILENAME: str = 'filename'
    FILEPATH: str = 'filepath'
    PKEY_COL: str = 'pkey_col'
    PKEY_CANDIDATE_COLS: str = 'pkey_candidate_cols'
    COLS: str = 'cols'
    RENAME_COLS: str = 'rename_cols'
    DROP_COLS: str = 'drop_cols'
    DROP_UNNAMED_COLS: str = 'drop_unnamed_cols'

class SitunesEnvironmentDataKeys:
    TIME: str = 'time'
    WEATHER: str = 'weather'
    GPS: str = 'GPS'

SITUNES_MUSIC_METADATA_DIR = SITUNES_DATA_DIR / 'music_metadata'
SITUNES_MUSIC_METADATA_MERGE_TYPE = 'outer'
SITUNES_MUSIC_METADATA_CONFIG = {
    'music_info': {
        SitunesConfigKeys.FILENAME: 'music_info.csv',
        SitunesConfigKeys.FILEPATH: SITUNES_MUSIC_METADATA_DIR / 'music_info.csv',
        SitunesConfigKeys.PKEY_CANDIDATE_COLS: ['item_id'],
        SitunesConfigKeys.DROP_COLS: [],
        SitunesConfigKeys.DROP_UNNAMED_COLS: True,
    },
    'music_info_withname': {
        SitunesConfigKeys.FILENAME: 'music_info_withname.csv',
        SitunesConfigKeys.FILEPATH: SITUNES_MUSIC_METADATA_DIR / 'music_info_withname.csv',
        SitunesConfigKeys.PKEY_CANDIDATE_COLS: ['i_id_c'],
        SitunesConfigKeys.DROP_COLS: [],
        SitunesConfigKeys.DROP_UNNAMED_COLS: True,
    },
}

SITUNES_STAGES = ['Stage2', 'Stage3']
SITUNES_STAGE_NUM_EXTRACTION_PATTERN = r'Stage(\d+)'
SITUNES_STAGE_DATA_CONFIG = {
    SitunesStageDataKeys.ENVIRONMENT: {
        SitunesConfigKeys.FILENAME: 'env.json',
        SitunesConfigKeys.PKEY_COL: 'inter_id',
        SitunesConfigKeys.COLS: {
            SitunesEnvironmentDataKeys.TIME: ['time_period'],
            SitunesEnvironmentDataKeys.WEATHER: ['weather_type', 'pressure', 'temperature', 'humidity'],
            SitunesEnvironmentDataKeys.GPS: ['longitude', 'latitude', 'speed']
        },
        SitunesConfigKeys.RENAME_COLS: {
            'pressure': 'weather_pressure',
            'temperature': 'weather_temp',
            'humidity': 'weather_humidity',
            'longitude': 'gps_lon',
            'latitude': 'gps_lat',
            'speed': 'gps_speed'
        },
        SitunesConfigKeys.DROP_COLS: [],
    },
    SitunesStageDataKeys.INTERACTIONS: {
        SitunesConfigKeys.FILENAME: 'interactions.csv',
        SitunesConfigKeys.PKEY_COL: 'inter_id'
    },
    SitunesStageDataKeys.WRIST: {
        SitunesConfigKeys.FILENAME: 'wrist.npy'
    },
}

