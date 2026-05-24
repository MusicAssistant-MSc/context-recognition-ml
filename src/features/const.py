from enum import IntEnum

class WristSignalsIdx(IntEnum):
    """Indices of the wrist signals in the data array."""
    HEART_RATE = 0
    ACT_INTENSITY = 1
    ACT_STEPS = 2
    ACT_TYPE = 3

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

