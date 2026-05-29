from dataclasses import dataclass
from typing import Final
from enum import StrEnum, auto

from src.features.const import (
    BioFeatureNames, MotionFeatureNames, ContextFeatureNames, MusicFeatureNames, EmotionFeatureNames, 
    SiTunesInterimWristColsIdx, SiTunesInterimInterEnvCols, SiTunesInterimMusicCols
)

class FeatureGroup(StrEnum):
    BIO = auto()
    MOTION = auto()
    CONTEXT = auto()
    MUSIC = auto()
    EMOTION = auto()

# type FeatureType = Literal['numerical', 'categorical']
class FeatureKind(StrEnum):
    NUMERICAL = auto()
    CATEGORICAL = auto()

@dataclass(frozen=True)
class FeatureMetadata:
    """Unified metadata for features.
    
    Attributes:
        name: The standardized name of the feature to be used in the final dataset.
        interim_name: The original column name in the interim data from which this feature is derived.
        group: The group to which the feature belongs (e.g. BIO, CONTEXT, MUSIC).
        kind: The type of the feature, either 'numerical' or 'categorical'.
        is_bounded: Whether the feature has a known fixed range (e.g., [0, 1] for normalized features).
        min_value: The minimum value of the feature if it is bounded, otherwise None.
        max_value: The maximum value of the feature if it is bounded, otherwise None.
    """
    name: str
    interim_name: str
    group: FeatureGroup
    kind: FeatureKind
    is_bounded: bool = False
    min_value: float | None = None
    max_value: float | None = None


SITUNES_FEATURES_METADATA: Final[list[FeatureMetadata]] = [
    # --- BIO (Wrist HR) ---
    FeatureMetadata(
        name=BioFeatureNames.HR_MEAN,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_STD,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_MIN,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_MAX,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_TREND_SLOPE,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_TREND_R_VALUE,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=-1.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=BioFeatureNames.HR_TREND_STD_ERR,
        interim_name=SiTunesInterimWristColsIdx.HEART_RATE.value,
        group=FeatureGroup.BIO,
        kind=FeatureKind.NUMERICAL,
    ),
    # --- MOTION (Wrist Activity Intensity/Steps/Type) ---
    FeatureMetadata(
        name=MotionFeatureNames.ACT_INTENSITY_MEAN,
        interim_name=SiTunesInterimWristColsIdx.ACT_INTENSITY.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_INTENSITY_STD,
        interim_name=SiTunesInterimWristColsIdx.ACT_INTENSITY.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_INTENSITY_MAX,
        interim_name=SiTunesInterimWristColsIdx.ACT_INTENSITY.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_STEPS_SUM,
        interim_name=SiTunesInterimWristColsIdx.ACT_STEPS.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_STEPS_ACTIVE_RATIO,
        interim_name=SiTunesInterimWristColsIdx.ACT_STEPS.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_TYPE_DOMINANT,
        interim_name=SiTunesInterimWristColsIdx.ACT_TYPE.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.CATEGORICAL,
    ),
    FeatureMetadata(
        name=MotionFeatureNames.ACT_TYPE_CHANGES_COUNT,
        interim_name=SiTunesInterimWristColsIdx.ACT_TYPE.value,
        group=FeatureGroup.MOTION,
        kind=FeatureKind.NUMERICAL,
    ),
    # --- CONTEXT (InterEnv Context Time/Weather/GPS) ---
    FeatureMetadata(
        name=ContextFeatureNames.TIME_PERIOD,
        interim_name=SiTunesInterimInterEnvCols.TIME_PERIOD,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.CATEGORICAL,
    ),
    FeatureMetadata(
        name=ContextFeatureNames.WEATHER_TYPE,
        interim_name=SiTunesInterimInterEnvCols.WEATHER_TYPE,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.CATEGORICAL,
    ),
    FeatureMetadata(
        name=ContextFeatureNames.WEATHER_PRESSURE,
        interim_name=SiTunesInterimInterEnvCols.WEATHER_PRESSURE,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=ContextFeatureNames.WEATHER_TEMP,
        interim_name=SiTunesInterimInterEnvCols.WEATHER_TEMP,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=ContextFeatureNames.WEATHER_HUMIDITY,
        interim_name=SiTunesInterimInterEnvCols.WEATHER_HUMIDITY,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=100.0
    ),
    FeatureMetadata(
        name=ContextFeatureNames.GPS_SPEED,
        interim_name=SiTunesInterimInterEnvCols.GPS_SPEED,
        group=FeatureGroup.CONTEXT,
        kind=FeatureKind.NUMERICAL,
    ),
    # --- MUSIC (MusicMetadata SpotifyLikeFeatures/LowLevelFeatures) ---
    FeatureMetadata(
        name=MusicFeatureNames.POPULARITY,
        interim_name=SiTunesInterimMusicCols.POPULARITY,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=100.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.LOUDNESS,
        interim_name=SiTunesInterimMusicCols.LOUDNESS,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.DANCEABILITY,
        interim_name=SiTunesInterimMusicCols.DANCEABILITY,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.ENERGY,
        interim_name=SiTunesInterimMusicCols.ENERGY,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.KEY,
        interim_name=SiTunesInterimMusicCols.KEY,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.CATEGORICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.SPEECHINESS,
        interim_name=SiTunesInterimMusicCols.SPEECHINESS,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.ACOUSTICNESS,
        interim_name=SiTunesInterimMusicCols.ACOUSTICNESS,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.INSTRUMENTALNESS,
        interim_name=SiTunesInterimMusicCols.INSTRUMENTALNESS,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.VALENCE,
        interim_name=SiTunesInterimMusicCols.VALENCE,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=0.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=MusicFeatureNames.TEMPO,
        interim_name=SiTunesInterimMusicCols.TEMPO,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.DURATION,
        interim_name=SiTunesInterimMusicCols.DURATION,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.GENRE_ID,
        interim_name=SiTunesInterimMusicCols.GENRE_ID,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.CATEGORICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.F0_MEAN,
        interim_name=SiTunesInterimMusicCols.F0_MEAN,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.RMS_ENERGY_STD,
        interim_name=SiTunesInterimMusicCols.RMS_ENERGY_STD,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.ZCR_MEAN,
        interim_name=SiTunesInterimMusicCols.ZCR_MEAN,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    FeatureMetadata(
        name=MusicFeatureNames.PSY_SHARPNESS_MEAN,
        interim_name=SiTunesInterimMusicCols.SHARPNESS_MEAN,
        group=FeatureGroup.MUSIC,
        kind=FeatureKind.NUMERICAL,
    ),
    # --- EMOTION (InterEnv Emo Pre Valence/Arousal) ---
    FeatureMetadata(
        name=EmotionFeatureNames.EMO_PRE_VALENCE,
        interim_name=SiTunesInterimInterEnvCols.EMO_PRE_VALENCE,
        group=FeatureGroup.EMOTION,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=-1.0,
        max_value=1.0
    ),
    FeatureMetadata(
        name=EmotionFeatureNames.EMO_PRE_AROUSAL,
        interim_name=SiTunesInterimInterEnvCols.EMO_PRE_AROUSAL,
        group=FeatureGroup.EMOTION,
        kind=FeatureKind.NUMERICAL,
        is_bounded=True,
        min_value=-1.0,
        max_value=1.0
    ),
]