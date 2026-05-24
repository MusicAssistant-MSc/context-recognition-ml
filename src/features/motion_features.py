import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from src.features import const


@dataclass
class MotionFeatureNamesMapping:
    act_intensity_mean: str
    act_intensity_std: str
    act_intensity_max: str
    act_steps_sum: str
    act_steps_active_ratio: str
    act_type_dominant: str
    act_type_changes_count: str


def _calculate_non_zero_values_ratio(signal: np.ndarray) -> float:
    """Calculate the ratio of non-zero values in a signal.
    
    Args:
        signal (np.ndarray): The input signal for which to calculate the ratio.

    Returns:
        float: The ratio of non-zero values in the signal.
    """
    if len(signal) == 0:
        return 0.0
    non_zero_count = np.count_nonzero(signal)
    return non_zero_count / len(signal)


def _find_dominant_value[T](signal: np.ndarray) -> T | None:
    """Find the most frequent value in a signal.
    
    Args:
        signal (np.ndarray): The input signal for which to find the dominant value.

    Returns:
        T | None: The most frequent value in the signal. Returns None if the signal is empty. If there are 
            multiple values with the same highest frequency, one of them is returned.
    """
    if len(signal) == 0:
        return None
    values, counts = np.unique(signal, return_counts=True)
    dominant_value = values[np.argmax(counts)]
    return dominant_value.item() if hasattr(dominant_value, 'item') else dominant_value


def _count_value_changes(signal: np.ndarray) -> int:
    """Count the number of times the value changes in a signal.
    
    Args:
        signal (np.ndarray): The input signal for which to count value changes.

    Returns:
        int: The number of value changes in the signal.
    """
    if len(signal) < 2:
        return 0
    changes = signal[1:] != signal[:-1]
    return np.sum(changes).item() if hasattr(changes, 'item') else np.sum(changes)


def extract_motion_features(
    wrist_data: np.ndarray, 
    inter_ids: list[str] | np.ndarray, 
    act_intensity_col_idx: int = const.WristSignalsIdx.ACT_INTENSITY.value,
    act_steps_col_idx: int = const.WristSignalsIdx.ACT_STEPS.value,
    act_type_col_idx: int = const.WristSignalsIdx.ACT_TYPE.value,
    feature_names_mapping: MotionFeatureNamesMapping = MotionFeatureNamesMapping(
        act_intensity_mean=const.MotionFeatureNames.ACT_INTENSITY_MEAN,
        act_intensity_std=const.MotionFeatureNames.ACT_INTENSITY_STD,
        act_intensity_max=const.MotionFeatureNames.ACT_INTENSITY_MAX,
        act_steps_sum=const.MotionFeatureNames.ACT_STEPS_SUM,
        act_steps_active_ratio=const.MotionFeatureNames.ACT_STEPS_ACTIVE_RATIO,
        act_type_dominant=const.MotionFeatureNames.ACT_TYPE_DOMINANT,
        act_type_changes_count=const.MotionFeatureNames.ACT_TYPE_CHANGES_COUNT
    )
) -> pd.DataFrame:
    """Extract motion features from wrist data.
    
    Args:
        wrist_data (np.ndarray): The input wrist data array containing the signals.
        inter_ids (list[str] | np.ndarray): The list or array of interaction IDs corresponding to the wrist data.
        act_intensity_col_idx (int): The column index for the activity intensity signal in the wrist data.
        act_steps_col_idx (int): The column index for the activity steps signal in the wrist data.
        act_type_col_idx (int): The column index for the activity type signal in the wrist data.
        feature_names_mapping (MotionFeatureNamesMapping): A mapping of feature names to be used in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted motion features for each interaction ID.
    """
    act_intensity_matrix = wrist_data[:, :, act_intensity_col_idx]
    act_steps_matrix = wrist_data[:, :, act_steps_col_idx]
    act_type_matrix = wrist_data[:, :, act_type_col_idx]

    features = {
        feature_names_mapping.act_intensity_mean: np.mean(act_intensity_matrix, axis=1),
        feature_names_mapping.act_intensity_std: np.std(act_intensity_matrix, axis=1),
        feature_names_mapping.act_intensity_max: np.max(act_intensity_matrix, axis=1),
        feature_names_mapping.act_steps_sum: np.sum(act_steps_matrix, axis=1),
        feature_names_mapping.act_steps_active_ratio: np.array([
            _calculate_non_zero_values_ratio(inter_matrix) for inter_matrix in act_steps_matrix]),
        feature_names_mapping.act_type_dominant: np.array([
            _find_dominant_value(inter_matrix) for inter_matrix in act_type_matrix]),
        feature_names_mapping.act_type_changes_count: np.array([
            _count_value_changes(inter_matrix) for inter_matrix in act_type_matrix])
    }

    return pd.DataFrame(features, index=inter_ids)
            