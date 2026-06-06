import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PowerTransformer

from src.features.feature_metadata import FeatureKind
from src.features.feature_registry import FeatureRegistry


def build_one_hot_encoder(handle_unknown: str, sparse_output: bool) -> OneHotEncoder:
    """Build configured OneHotEncoder."""
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=sparse_output
    )


def build_quantile_transformer(quantile_out_dist: str, random_state: int) -> QuantileTransformer:
    """Build configured QuantileTransformer."""
    return QuantileTransformer(
        output_distribution=quantile_out_dist,
        random_state=random_state
    )


def build_power_transformer(method: str, stardardize: bool) -> PowerTransformer:
    """Build configured PowerTransformer."""
    return PowerTransformer(
        method=method,
        standardize=stardardize
    )


def calculate_feature_scaling_stats(df: pd.DataFrame, feature_name: str) -> tuple[float, float, float]:
    """Calculate statistics for feature scaling: abs(skewness), kurtosis, and standard deviation.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the feature.
        feature_name (str): The name of the feature to calculate statistics for.

    Returns:
        tuple[float, float, float]: A tuple containing abs(skewness), kurtosis, and standard deviation.
    """
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")
    abs_skewness = abs(df[feature_name].skew())
    kurtosis = df[feature_name].kurtosis()
    std_dev = df[feature_name].std()
    return abs_skewness, kurtosis, std_dev


def get_numeric_feature_names(feature_names: list[str], feature_registry: FeatureRegistry) -> list[str]:
    """Get numeric features from a list of feature names and a feature registry.
    
    Args:
        feature_names (list[str]): A list of feature names.
        feature_registry (FeatureRegistry): The feature registry containing the FeatureMetadata objects.

    Returns:
        list[str]: A list of feature names representing numeric features.
    """
    feature_metadata = feature_registry.get_features(feature_names)
    numeric_features = feature_registry.filter(kind=FeatureKind.NUMERICAL, source=feature_metadata)
    numeric_names = feature_registry.get_names(numeric_features)
    return numeric_names


def get_categorical_feature_names(feature_names: list[str], feature_registry: FeatureRegistry) -> list[str]:
    """Get categorical features from a list of feature names and a feature registry.
    
    Args:
        features (list[str]): A list of feature names.
        feature_registry (FeatureRegistry): The feature registry containing the FeatureMetadata objects.

    Returns:
        list[str]: A list of feature names representing categorical features.
    """
    feature_metadata = feature_registry.get_features(feature_names)
    categorical_features = feature_registry.filter(kind=FeatureKind.CATEGORICAL, source=feature_metadata)
    categorical_names = feature_registry.get_names(categorical_features)
    return categorical_names