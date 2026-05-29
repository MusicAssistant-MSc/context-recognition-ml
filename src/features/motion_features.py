import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from src.features import const
from src.features.feature_metadata import FeatureMetadata
from src.features.feature_registry import FEATURE_REGISTRY


@dataclass
class MotionFeatureMapping:
    act_intensity_mean: FeatureMetadata
    act_intensity_std: FeatureMetadata
    act_intensity_max: FeatureMetadata
    act_steps_sum: FeatureMetadata
    act_steps_active_ratio: FeatureMetadata
    act_type_dominant: FeatureMetadata
    act_type_changes_count: FeatureMetadata

    @classmethod
    def from_registry(cls) -> "MotionFeatureMapping":
        """Factory method to create MotionFeatureMapping using global registry of features."""
        return cls(
            act_intensity_mean=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_INTENSITY_MEAN),
            act_intensity_std=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_INTENSITY_STD),
            act_intensity_max=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_INTENSITY_MAX),
            act_steps_sum=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_STEPS_SUM),
            act_steps_active_ratio=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_STEPS_ACTIVE_RATIO),
            act_type_dominant=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_TYPE_DOMINANT),
            act_type_changes_count=FEATURE_REGISTRY.get(const.MotionFeatureNames.ACT_TYPE_CHANGES_COUNT)
        )


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
    act_intensity_col_idx: int = const.SiTunesInterimWristColsIdx.ACT_INTENSITY.value,
    act_steps_col_idx: int = const.SiTunesInterimWristColsIdx.ACT_STEPS.value,
    act_type_col_idx: int = const.SiTunesInterimWristColsIdx.ACT_TYPE.value,
    feature_mapping: MotionFeatureMapping = MotionFeatureMapping.from_registry()
) -> pd.DataFrame:
    """Extract motion features from wrist data.
    
    Args:
        wrist_data (np.ndarray): The input wrist data array containing the signals.
        inter_ids (list[str] | np.ndarray): The list or array of interaction IDs corresponding to the wrist data.
        act_intensity_col_idx (int): The column index for the activity intensity signal in the wrist data.
        act_steps_col_idx (int): The column index for the activity steps signal in the wrist data.
        act_type_col_idx (int): The column index for the activity type signal in the wrist data.
        feature_mapping (MotionFeatureMapping): A mapping of feature objects with names to be used in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted motion features for each interaction ID.
    """
    act_intensity_matrix = wrist_data[:, :, act_intensity_col_idx]
    act_steps_matrix = wrist_data[:, :, act_steps_col_idx]
    act_type_matrix = wrist_data[:, :, act_type_col_idx]

    features = {
        feature_mapping.act_intensity_mean.name: np.mean(act_intensity_matrix, axis=1),
        feature_mapping.act_intensity_std.name: np.std(act_intensity_matrix, axis=1),
        feature_mapping.act_intensity_max.name: np.max(act_intensity_matrix, axis=1),
        feature_mapping.act_steps_sum.name: np.sum(act_steps_matrix, axis=1),
        feature_mapping.act_steps_active_ratio.name: np.array([
            _calculate_non_zero_values_ratio(inter_matrix) for inter_matrix in act_steps_matrix]),
        feature_mapping.act_type_dominant.name: np.array([
            _find_dominant_value(inter_matrix) for inter_matrix in act_type_matrix]),
        feature_mapping.act_type_changes_count.name: np.array([
            _count_value_changes(inter_matrix) for inter_matrix in act_type_matrix])
    }

    return pd.DataFrame(features, index=inter_ids)
            