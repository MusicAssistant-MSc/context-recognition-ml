from typing import Any
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

from src.features.const import (
    PreprocessingStrategy,
    DistanceMLPreprocessingSubStrategies,
    TREE_NUM_TRANSFORMER_NAME,
    TREE_CAT_TRANSFORMER_NAME,
    SKEW_EXTREME_THRESHOLD,
    SKEW_MODERATE_THRESHOLD,
    KURT_EXTREME_THRESHOLD,
    KURT_MODERATE_THRESHOLD,
    QUANTILE_OUTPUT_DISTRIBUTION,
    QUANTILE_RANDOM_STATE,
    POWER_METHOD,
    POWER_STANDARDIZE,
    OHE_HANDLE_UNKNOWN,
    OHE_SPARSE_OUTPUT,
    COL_TRANSFORMER_REMAINDER
)
from src.features.feature_registry import FeatureRegistry, FEATURE_REGISTRY
from src.features.preprocessing_utils import (
    build_one_hot_encoder,
    build_quantile_transformer,
    build_power_transformer,
    calculate_feature_scaling_stats, 
    get_numeric_feature_names, 
    get_categorical_feature_names
)


def build_tree_based_preprocessor(
    feature_names: list[str], 
    feature_registry: FeatureRegistry = FEATURE_REGISTRY,
    ohe_handle_unknown: str = OHE_HANDLE_UNKNOWN,
    ohe_sparse_output: bool = OHE_SPARSE_OUTPUT,
    num_transformer_name: str = TREE_NUM_TRANSFORMER_NAME,
    cat_transformer_name: str = TREE_CAT_TRANSFORMER_NAME,
    col_trans_remainder: str = COL_TRANSFORMER_REMAINDER
) -> ColumnTransformer:
    """Build preprocessing for tree-based models.

    Tree-based models are mostly scale invariant.
    Minimal preprocessing is preferred.

    Args:
        feature_names (list[str]): Feature names to process.
        feature_registry (FeatureRegistry): Feature metadata registry.
        ohe_handle_unknown (str): OneHotEncoder handle_unknown parameter.
        ohe_sparse_output (bool): OneHotEncoder sparse_output parameter.
        num_transformer_name (str): Name of numeric transformer.
        cat_transformer_name (str): Name of categorical transformer.
        col_trans_remainder (str): ColumnTransformer remainder parameter.

    Returns:
        Configured ColumnTransformer for tree-based models.
    """
    # Extract numeric and categorical feature names using registry
    num_feat_names = get_numeric_feature_names(feature_names, feature_registry)
    cat_feat_names = get_categorical_feature_names(feature_names, feature_registry)

    return ColumnTransformer(
        transformers=[
            (
                num_transformer_name,
                "passthrough",
                num_feat_names
            ),
            (
                cat_transformer_name,
                build_one_hot_encoder(ohe_handle_unknown, ohe_sparse_output),
                cat_feat_names
            )
        ],
        remainder=col_trans_remainder,
    )


def _divide_features_into_dist_sub_groups(
    feature_df: pd.DataFrame,
    feature_names: list[str], 
    feature_registry: FeatureRegistry,
    preproc_sub_strats_names: DistanceMLPreprocessingSubStrategies,
    extreme_skew: float,
    moderate_skew: float,
    extreme_kurt: float,
    moderate_kurt: float  
) -> dict[str, list[str]]:
    """Divide features into preprocessing sub-groups (sub-strategies) for distance-based models.

    Args:
        feature_df (pd.DataFrame): DataFrame containing feature data for statistics calculation.
        feature_names (list[str]): Feature names to process.
        feature_registry (FeatureRegistry): Feature metadata registry.
        preproc_sub_strats_names (DistanceMLPreprocessingSubStrategies): Preprocessing sub-strategy (feature groups) names.
        extreme_skew (float): Extreme skewness threshold.
        moderate_skew (float): Moderate skewness threshold.
        extreme_kurt (float): Extreme kurtosis threshold.
        moderate_kurt (float): Moderate kurtosis threshold.

    Returns:
        dict[str, list[str]]: Dictionary mapping preprocessing group names to feature names assigned to each group.
    """
    # Define preprocessing sub-strategy groups for distance-based models
    preprocessing_groups = {
        preproc_sub_strats_names.QUANTILE: [],
        preproc_sub_strats_names.POWER: [],
        preproc_sub_strats_names.BOUNDED: [],
        preproc_sub_strats_names.ROBUST: [],
        preproc_sub_strats_names.STANDARD: [],
        preproc_sub_strats_names.CATEGORICAL: [],
    }

    # Extract numeric and categorical feature names using registry
    num_feat_names = get_numeric_feature_names(feature_names, feature_registry)
    cat_feat_names = get_categorical_feature_names(feature_names, feature_registry)

    # Assign categorical features to categorical group
    preprocessing_groups[preproc_sub_strats_names.CATEGORICAL].extend(cat_feat_names)
    
    # Iterate through numeric features and assign them to preprocessing groups
    for feature_name in num_feat_names:
        # Get abs(skewness), kurtosis and std_dev for feature
        feat_abs_skew, feat_kurt, feat_std_dev = calculate_feature_scaling_stats(feature_df, feature_name)
        # Get feature metadata
        feat_metadata = feature_registry.get(feature_name)
        
        # Heavy-tailed distributions
        if (feat_abs_skew >= extreme_skew or feat_kurt >= extreme_kurt):
            preprocessing_groups[preproc_sub_strats_names.QUANTILE].append(feature_name)
            continue

        # Moderately skewed distributions
        if feat_abs_skew >= moderate_skew:
            preprocessing_groups[preproc_sub_strats_names.POWER].append(feature_name)
            continue

        # Stable bounded features
        if feat_metadata.is_bounded:
            preprocessing_groups[preproc_sub_strats_names.BOUNDED].append(feature_name)
            continue

        # Stable features with moderate kurtosis
        if feat_kurt >= moderate_kurt:
            preprocessing_groups[preproc_sub_strats_names.ROBUST].append(feature_name)
            continue

        # Default continuous features
        preprocessing_groups[preproc_sub_strats_names.STANDARD].append(feature_name)

    return preprocessing_groups


def build_distance_based_preprocessor(
    feature_df: pd.DataFrame,
    feature_names: list[str], 
    feature_registry: FeatureRegistry = FEATURE_REGISTRY,
    preproc_sub_strats_names: type[DistanceMLPreprocessingSubStrategies] = DistanceMLPreprocessingSubStrategies,
    extreme_skew: float = SKEW_EXTREME_THRESHOLD,
    moderate_skew: float = SKEW_MODERATE_THRESHOLD,
    extreme_kurt: float = KURT_EXTREME_THRESHOLD,
    moderate_kurt: float = KURT_MODERATE_THRESHOLD,
    ohe_handle_unknown: str = OHE_HANDLE_UNKNOWN,
    ohe_sparse_output: bool = OHE_SPARSE_OUTPUT,
    quantile_out_dist: str = QUANTILE_OUTPUT_DISTRIBUTION,
    quantile_rand_state: int = QUANTILE_RANDOM_STATE,
    power_method: str = POWER_METHOD,
    power_standardize: bool = POWER_STANDARDIZE,
    col_trans_remainder: str = COL_TRANSFORMER_REMAINDER
) -> ColumnTransformer:
    """Build preprocessing for scale-sensitive models.
    
    Includes:
    - SVR
    - Ridge Regression
    - MLP

    Strategy:
    - Extreme skew/heavy tails -> QuantileTransformer
    - Moderate skew -> PowerTransformer
    - Stable bounded features -> MinMaxScaler
    - Default continuous features -> StandardScaler
    - Outlier-prone features -> RobustScaler

    Args:
        feature_df (pd.DataFrame): DataFrame containing feature data.
        feature_names (list[str]): Feature names to process.
        feature_registry (FeatureRegistry): Feature metadata registry.


    Returns:
        Configured ColumnTransformer for distance-based models.
    """
    preprocessing_groups: dict[str, list[str]] = _divide_features_into_dist_sub_groups(
        feature_df,
        feature_names,
        feature_registry,
        preproc_sub_strats_names,
        extreme_skew,
        moderate_skew,
        extreme_kurt,
        moderate_kurt
    )

    transformers: list[tuple[str, Any, list[str]]] = []

    if preprocessing_groups[preproc_sub_strats_names.CATEGORICAL]:
        transformers.append(
            (
                preproc_sub_strats_names.CATEGORICAL,
                build_one_hot_encoder(ohe_handle_unknown, ohe_sparse_output),
                preprocessing_groups[preproc_sub_strats_names.CATEGORICAL]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.QUANTILE]:
        transformers.append(
            (
                preproc_sub_strats_names.QUANTILE,
                build_quantile_transformer(quantile_out_dist, quantile_rand_state),
                preprocessing_groups[preproc_sub_strats_names.QUANTILE]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.POWER]:
        transformers.append(
            (
                preproc_sub_strats_names.POWER,
                build_power_transformer(power_method, power_standardize),
                preprocessing_groups[preproc_sub_strats_names.POWER]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.BOUNDED]:
        transformers.append(
            (
                preproc_sub_strats_names.BOUNDED,
                MinMaxScaler(),
                preprocessing_groups[preproc_sub_strats_names.BOUNDED]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.ROBUST]:
        transformers.append(
            (
                preproc_sub_strats_names.ROBUST,
                RobustScaler(),
                preprocessing_groups[preproc_sub_strats_names.ROBUST]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.STANDARD]:
        transformers.append(
            (
                preproc_sub_strats_names.STANDARD,
                StandardScaler(),
                preprocessing_groups[preproc_sub_strats_names.STANDARD]
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=col_trans_remainder)


def build_feature_preprocessor(
    strategy: PreprocessingStrategy,
    feature_names: list[str],
    feature_df: pd.DataFrame
) -> ColumnTransformer:
    """Build preprocessing pipeline for selected ML strategy.

    Args:
        strategy (PreprocessingStrategy): Selected preprocessing strategy (e.g. TREE_BASED or DISTANCE_BASED).
        feature_names (list[str]): List of feature names to process.
        feature_df (pd.DataFrame): DataFrame containing feature data for statistics calculation.

    Returns:
        Configured sklearn ColumnTransformer.

    Raises:
        ValueError: If unsupported strategy is provided.
    """
    match strategy:
        case PreprocessingStrategy.TREE_BASED:
            return build_tree_based_preprocessor(feature_names)
        case PreprocessingStrategy.DISTANCE_BASED:
            return build_distance_based_preprocessor(feature_df, feature_names)
        case _:
            raise ValueError(f"Unsupported preprocessing strategy: {strategy}")
