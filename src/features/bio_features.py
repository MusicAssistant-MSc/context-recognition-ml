import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from src.features import const


@dataclass
class BioFeatureNamesMapping:
    hr_mean: str
    hr_std: str
    hr_min: str
    hr_max: str
    hr_trend_slope: str
    hr_trend_r_value: str
    hr_trend_std_err: str


def _calculate_regression_stats(
    signal: np.ndarray,
    slope_key: str,
    r_value_key: str,
    std_err_key: str
) -> dict[str, float]:
    """Calculate the slope, R-value, and standard error of a signal using linear regression.
    
    Args:
        signal (np.ndarray): The input signal for which to calculate the statistics.
        slope_key (str): The key name for the slope in the returned dictionary.
        r_value_key (str): The key name for the R-value in the returned dictionary.
        std_err_key (str): The key name for the standard error in the returned dictionary.

    Returns:
        dict[str, float]: A dictionary containing the calculated slope, R-value, and standard error.
    """
    # Create x-axis indices corresponding to the length of the signal
    time_indices = np.arange(len(signal))

    slope, _, r_value, _, std_err = stats.linregress(time_indices, signal)
    return {
        slope_key: float(slope),
        r_value_key: float(r_value),
        std_err_key: float(std_err)
    }


def extract_hr_features(
    wrist_data: np.ndarray, 
    inter_ids: list[str] | np.ndarray, 
    hr_col_idx: int = const.WristSignalsIdx.HEART_RATE.value,
    feature_names_mapping: BioFeatureNamesMapping = BioFeatureNamesMapping(
        hr_mean=const.BioFeatureNames.HR_MEAN,
        hr_std=const.BioFeatureNames.HR_STD,
        hr_min=const.BioFeatureNames.HR_MIN,
        hr_max=const.BioFeatureNames.HR_MAX,
        hr_trend_slope=const.BioFeatureNames.HR_TREND_SLOPE,
        hr_trend_r_value=const.BioFeatureNames.HR_TREND_R_VALUE,
        hr_trend_std_err=const.BioFeatureNames.HR_TREND_STD_ERR
    )
) -> pd.DataFrame:
    """Extract heart rate features from wrist data for specified intervals.
    
    Args:
        wrist_data (np.ndarray): The raw wrist data containing heart rate information.
        inter_ids (list[str] | np.ndarray): A list or array of interval identifiers corresponding to the wrist data.
        hr_col_idx (int): The column index in the wrist data that contains heart rate values.
        feature_names_mapping (BioFeatureNamesMapping): A mapping of feature names to be used in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted heart rate features for each interval.
    """
    # Extract heart rate column for all interactions based on the specified HR column index
    hr_matrix = wrist_data[:, :, hr_col_idx] 

    # Calculate basic statistics for each interaction HR column
    features = {
        feature_names_mapping.hr_mean: np.mean(hr_matrix, axis=1),
        feature_names_mapping.hr_std: np.std(hr_matrix, axis=1),
        feature_names_mapping.hr_min: np.min(hr_matrix, axis=1),
        feature_names_mapping.hr_max: np.max(hr_matrix, axis=1),
    }

    # Calculate regression statistics for each interaction HR column
    regression_stats = [
        _calculate_regression_stats(
            inter_matrix,
            feature_names_mapping.hr_trend_slope,
            feature_names_mapping.hr_trend_r_value,
            feature_names_mapping.hr_trend_std_err
        ) for inter_matrix in hr_matrix
    ]
    
    # Add the regression statistics to the features dictionary
    features[feature_names_mapping.hr_trend_slope] = [
        stat[feature_names_mapping.hr_trend_slope] for stat in regression_stats]
    features[feature_names_mapping.hr_trend_r_value] = [
        stat[feature_names_mapping.hr_trend_r_value] for stat in regression_stats]
    features[feature_names_mapping.hr_trend_std_err] = [
        stat[feature_names_mapping.hr_trend_std_err] for stat in regression_stats]

    return pd.DataFrame(features, index=inter_ids)