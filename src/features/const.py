from typing import Final
from enum import StrEnum, auto

from src.utils.const import SiTunesInterimWristColsIdx
from src.utils.const import SiTunesInterimInterEnvCols, SiTunesInterimMusicCols


class BioFeatureNames:
    """Names of the extracted biological features."""
    HR_MEAN = 'hr_mean'
    HR_STD = 'hr_std'
    HR_MIN = 'hr_min'
    HR_MAX = 'hr_max'
    HR_TREND_SLOPE = 'hr_trend_slope'
    HR_TREND_R_VALUE = 'hr_trend_r_value'
    HR_TREND_STD_ERR = 'hr_trend_std_err'

class MotionFeatureNames:
    """Names of the extracted motion features."""
    ACT_INTENSITY_MEAN = 'act_intensity_mean'
    ACT_INTENSITY_STD = 'act_intensity_std'
    ACT_INTENSITY_MAX = 'act_intensity_max'
    ACT_STEPS_SUM = 'act_steps_sum'
    ACT_STEPS_ACTIVE_RATIO = 'act_steps_active_ratio'
    ACT_TYPE_DOMINANT = 'act_type_dominant'
    ACT_TYPE_CHANGES_COUNT = 'act_type_changes_count'

class ContextFeatureNames:
    """Names of the extracted context features."""
    TIME_PERIOD = 'context_time_period'
    WEATHER_TYPE = 'context_weather_type'
    WEATHER_PRESSURE = 'context_weather_pressure'
    WEATHER_TEMP: str = 'context_weather_temp'
    WEATHER_HUMIDITY: str = 'context_weather_humidity'
    GPS_SPEED: str = 'context_gps_speed'

class MusicFeatureNames:
    """Names of the extracted music features."""
    POPULARITY: str = 'music_popularity'
    LOUDNESS: str = 'music_loudness'
    DANCEABILITY: str = 'music_danceability'
    ENERGY: str = 'music_energy'
    KEY: str = 'music_key'
    SPEECHINESS: str = 'music_speechiness'
    ACOUSTICNESS: str = 'music_acousticness'
    INSTRUMENTALNESS: str = 'music_instrumentalness'
    VALENCE: str = 'music_valence'
    TEMPO: str = 'music_tempo'
    DURATION: str = 'music_duration'
    GENRE_ID: str = 'music_genre_id'
    # Low-level
    F0_MEAN: str = 'music_f0_mean'
    RMS_ENERGY_STD: str = 'music_rms_std'
    ZCR_MEAN: str = 'music_zcr_mean'
    PSY_SHARPNESS_MEAN: str = 'music_psy_sharpness_mean'

class EmotionFeatureNames:
    """Names of the extracted emotion features."""
    EMO_PRE_VALENCE: str = 'emo_pre_valence'
    EMO_PRE_AROUSAL: str = 'emo_pre_arousal'


class FeaturePreprocessingStats(StrEnum):
    """Names of statistics used for preprocessing pipeline decisions."""
    SKEWNESS = auto()
    KURTOSIS = auto()
    STD_DEV = auto()

class PreprocessingStrategy(StrEnum):
    """Available preprocessing strategies for ML model families."""
    TREE_BASED = auto()
    DISTANCE_BASED = auto()

class DistanceMLPreprocessingSubStrategies(StrEnum):
    """
    Names of sub-strategies (groups) of features used for distance-based ML preprocessing decisions.
    Also used as transformer names in ColumnTransformer for distance-based ML algorithms.
    """
    QUANTILE = auto()
    POWER = auto()
    BOUNDED = auto()
    ROBUST = auto()
    STANDARD = auto()
    CATEGORICAL = auto()


# Thresholds for statistical preprocessing decisions
SKEW_EXTREME_THRESHOLD: Final[float] = 2.0
SKEW_MODERATE_THRESHOLD: Final[float] = 1.0
# Fisher excess kurtosis threshold (normal distribution kurtosis == 0)
KURT_EXTREME_THRESHOLD: Final[float] = 3.0
KURT_MODERATE_THRESHOLD: Final[float] = 1.0

# Quantile transformer defaults
QUANTILE_OUTPUT_DISTRIBUTION: Final[str] = "normal"
QUANTILE_RANDOM_STATE: Final[int] = 42

# Power transformer defaults
POWER_METHOD: Final[str] = "yeo-johnson"
POWER_STANDARDIZE: Final[bool] = True

# OneHotEncoder defaults
OHE_HANDLE_UNKNOWN: Final[str] = "ignore"
OHE_SPARSE_OUTPUT: Final[bool] = False

# ColumnTransformer defaults
COL_TRANSFORMER_REMAINDER = "drop"

# Tree-based sub-transformers defaults
TREE_NUM_TRANSFORMER_NAME: Final[str] = "numeric"
TREE_CAT_TRANSFORMER_NAME: Final[str] = "categorical"

# SimpleImputer defaults
IMPUTER_NAME: Final[str] = "imputer"
NUMERIC_IMPUTER_STRATEGY: Final[str] = "median"
CATEGORICAL_IMPUTER_STRATEGY: Final[str] = "most_frequent"
IMPUTER_FILL_VALUE: Final[str | None] = None