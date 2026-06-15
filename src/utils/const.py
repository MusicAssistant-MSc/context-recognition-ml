from pathlib import Path
from enum import IntEnum
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'

PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_FEATURES_DATASET_FILENAME = 'processed_features_dataset.csv'

REPORTS_DIR = PROJECT_ROOT / 'src' / 'reports'

SAVED_MODELS_DIR = PROJECT_ROOT / 'models'

LOGS_DIR = PROJECT_ROOT / 'logs'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
ABLATION_STUDY_LOG_FILENAME = 'ablation_study'

# Standard timestamp format for filenames
# Example: 20240522_153000
FILE_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
# Regex to match the above format preceded by an underscore at the end of a string
FILE_TIMESTAMP_SUFFIX_PATTERN: Final[str] = r"_\d{8}_\d{6}$"

# --- Statistical tests constants ---
ALPHA_SIGNIFICANCE_LEVEL: Final[float] = 0.05
POSTHOC_CONOVER_P_ADJUST: Final[str] = "holm"

PANDAS_DF_UNNAMED_COL_PREFIX = 'Unnamed: '

class SiTunesInterimInterEnvCols:
    """Shared column names for interim interaction and environment data."""
    # IDs
    INTER_ID: str = 'inter_id'
    USER_ID: str = 'user_id'
    ITEM_ID: str = 'item_id'
    # Iteration timestamp
    TIMESTAMP: str = 'timestamp'
    # Iteration labels
    RATING: str = 'rating'
    PREFERENCE: str = 'preference'
    # Iteration and env features
    EMO_PRE_VALENCE: str = 'emo_pre_valence'
    EMO_PRE_AROUSAL: str = 'emo_pre_arousal'
    EMO_POST_VALENCE: str = 'emo_post_valence'
    EMO_POST_AROUSAL: str = 'emo_post_arousal'
    DURATION: str = 'duration'
    REC_TYPE: str = 'rec_type'
    TIME_PERIOD: str = 'time_period'
    WEATHER_TYPE: str = 'weather_type'
    WEATHER_PRESSURE: str = 'weather_pressure'
    WEATHER_TEMP: str = 'weather_temp'
    WEATHER_HUMIDITY: str = 'weather_humidity'
    GPS_LON: str = 'gps_lon'
    GPS_LAT: str = 'gps_lat'
    GPS_SPEED: str = 'gps_speed'

class SiTunesInterimMusicCols:
    """Shared column names for interim music metadata."""
    # IDs
    ITEM_ID: str = 'item_id'
    # Audio features (Spotify-like)
    POPULARITY: str = 'popularity'
    LOUDNESS: str = 'loudness'
    DANCEABILITY: str = 'danceability'
    ENERGY: str = 'energy'
    KEY: str = 'key'
    SPEECHINESS: str = 'speechiness'
    ACOUSTICNESS: str = 'acousticness'
    INSTRUMENTALNESS: str = 'instrumentalness'
    VALENCE: str = 'valence'
    TEMPO: str = 'tempo'
    DURATION: str = 'duration'
    # Genre information
    GENRE_ID: str = 'general_genre_id'
    GENRE: str = 'general_genre'
    # Low-level acoustic features (OpenSMILE/Signal Processing)
    F0_MEAN: str = 'F0final_sma_amean'
    F0_STD: str = 'F0final_sma_stddev'
    AUDSPEC_L1_STD: str = 'audspec_lengthL1norm_sma_stddev'
    RMS_ENERGY_STD: str = 'pcm_RMSenergy_sma_stddev'
    SHARPNESS_MEAN: str = 'pcm_fftMag_psySharpness_sma_amean'
    SHARPNESS_STD: str = 'pcm_fftMag_psySharpness_sma_stddev'
    ZCR_MEAN: str = 'pcm_zcr_sma_amean'
    ZCR_STD: str = 'pcm_zcr_sma_stddev'
    # Song metadata
    TITLE: str = 'music'
    ARTIST: str = 'singer'

class SiTunesInterimWristColsIdx(IntEnum):
    """Indices of the wrist signal columns in the interim data."""
    HEART_RATE = 0
    ACT_INTENSITY = 1
    ACT_STEPS = 2
    ACT_TYPE = 3