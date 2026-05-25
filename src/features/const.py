from src.utils.const import SiTunesInterimWristColsIdx as WristSignalsIdx
from src.utils.const import SiTunesInterimInterEnvCols
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