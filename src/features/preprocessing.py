from typing import Any
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    COL_TRANSFORMER_REMAINDER,
    IMPUTER_NAME,
    NUMERIC_IMPUTER_STRATEGY,
    CATEGORICAL_IMPUTER_STRATEGY,
    IMPUTER_FILL_VALUE
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


def _build_sub_strat_imputer_pipeline(
    main_transformer: Any,
    main_transformer_name: str,
    imputer_name: str,
    imputer_strategy: str,
    fill_value: str | None = None,
    add_indicator: bool = False
) -> Pipeline:
    """Build a Pipeline with SimpleImputer at the beginning and passed transformer afterwards.
    
    Args:
        main_transformer (Any): Transformer to apply after imputer.
        main_transformer_name (str): Name of main transformer to use as a step name.
        imputer_name (str): Name of imputer to use as a step name.
        imputer_strategy (str): Imputer strategy to use (e.g. 'constant', 'mean', 'median', 'most_frequent').
        fill_value (str | None, optional): Imputer fill value used with strategy='constant'. Defaults to None.
        add_indicator (bool, optional): Whether to add missing value indicator features. Defaults to False.

    Returns:
        Pipeline: Pipeline with SimpleImputer at the beginning and passed transformer afterwards.
    """
    return Pipeline(
        steps= [
            (imputer_name, 
                SimpleImputer(
                    strategy=imputer_strategy, 
                    fill_value=fill_value,
                    add_indicator=add_indicator
                )
            ),
            (main_transformer_name, main_transformer)
        ]
    )


def build_tree_based_preprocessor(
    feature_names: list[str], 
    use_imputer: bool,
    feature_registry: FeatureRegistry = FEATURE_REGISTRY,
    ohe_handle_unknown: str = OHE_HANDLE_UNKNOWN,
    ohe_sparse_output: bool = OHE_SPARSE_OUTPUT,
    num_transformer_name: str = TREE_NUM_TRANSFORMER_NAME,
    cat_transformer_name: str = TREE_CAT_TRANSFORMER_NAME,
    imputer_name: str = IMPUTER_NAME,
    num_imputer_strategy: str = NUMERIC_IMPUTER_STRATEGY,
    cat_imputer_strategy: str = CATEGORICAL_IMPUTER_STRATEGY,
    imputer_fill_value: str | None = IMPUTER_FILL_VALUE,
    col_trans_remainder: str = COL_TRANSFORMER_REMAINDER
) -> ColumnTransformer:
    """Build preprocessing for tree-based models.

    Tree-based models are mostly scale invariant.
    Minimal preprocessing is preferred.

    Args:
        feature_names (list[str]): Feature names to process.
        use_imputer (bool): Whether to use SimpleImputer to impute missing values for tree-based models.
        feature_registry (FeatureRegistry): Feature metadata registry.
        ohe_handle_unknown (str): OneHotEncoder handle_unknown parameter.
        ohe_sparse_output (bool): OneHotEncoder sparse_output parameter.
        num_transformer_name (str): Name of numeric transformer.
        cat_transformer_name (str): Name of categorical transformer.
        imputer_name (str): Name of imputer step in pipeline.
        num_imputer_strategy (str): Imputer strategy for numeric features.
        cat_imputer_strategy (str): Imputer strategy for categorical features.
        imputer_fill_value (str | None): Imputer fill value for strategy='constant'.
        col_trans_remainder (str): ColumnTransformer remainder parameter.

    Returns:
        Configured ColumnTransformer for tree-based models.
    """
    # Extract numeric and categorical feature names using registry
    num_feat_names = get_numeric_feature_names(feature_names, feature_registry)
    cat_feat_names = get_categorical_feature_names(feature_names, feature_registry)

    # Build transformers
    if use_imputer:
        num_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer="passthrough",
            main_transformer_name=num_transformer_name,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value
        )
        cat_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=build_one_hot_encoder(ohe_handle_unknown, ohe_sparse_output),
            main_transformer_name=cat_transformer_name,
            imputer_name=imputer_name,
            imputer_strategy=cat_imputer_strategy,
            fill_value=imputer_fill_value
        )
    else:
        num_transformer = "passthrough"
        cat_transformer = build_one_hot_encoder(ohe_handle_unknown, ohe_sparse_output)

    return ColumnTransformer(
        transformers=[
            (
                num_transformer_name,
                num_transformer,
                num_feat_names
            ),
            (
                cat_transformer_name,
                cat_transformer,
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
    imputer_name: str = IMPUTER_NAME,
    num_imputer_strategy: str = NUMERIC_IMPUTER_STRATEGY,
    cat_imputer_strategy: str = CATEGORICAL_IMPUTER_STRATEGY,
    imputer_fill_value: str | None = IMPUTER_FILL_VALUE,
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
        preproc_sub_strats_names (DistanceMLPreprocessingSubStrategies): Preprocessing sub-strategy (feature groups) names.
        extreme_skew (float): Extreme skewness threshold.
        moderate_skew (float): Moderate skewness threshold.
        extreme_kurt (float): Extreme kurtosis threshold.
        moderate_kurt (float): Moderate kurtosis threshold.
        ohe_handle_unknown (str): OneHotEncoder handle_unknown parameter.
        ohe_sparse_output (bool): OneHotEncoder sparse_output parameter.
        quantile_out_dist (str): QuantileTransformer output_distribution parameter.
        quantile_rand_state (int): QuantileTransformer random_state parameter.
        power_method (str): PowerTransformer method parameter.
        power_standardize (bool): PowerTransformer standardize parameter.
        imputer_name (str): Name of imputer step in pipeline.
        num_imputer_strategy (str): Imputer strategy for numeric features.
        cat_imputer_strategy (str): Imputer strategy for categorical features.
        imputer_fill_value (str | None): Imputer fill_value parameter.
        col_trans_remainder (str): ColumnTransformer remainder parameter.

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
        cat_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=build_one_hot_encoder(ohe_handle_unknown, ohe_sparse_output),
            main_transformer_name=preproc_sub_strats_names.CATEGORICAL,
            imputer_name=imputer_name,
            imputer_strategy=cat_imputer_strategy,
            fill_value=imputer_fill_value,
        )
        transformers.append(
            (
                preproc_sub_strats_names.CATEGORICAL,
                cat_transformer,
                preprocessing_groups[preproc_sub_strats_names.CATEGORICAL]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.QUANTILE]:
        quant_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=build_quantile_transformer(quantile_out_dist, quantile_rand_state),
            main_transformer_name=preproc_sub_strats_names.QUANTILE,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value,
            add_indicator=True
        )
        transformers.append(
            (
                preproc_sub_strats_names.QUANTILE,
                quant_transformer,
                preprocessing_groups[preproc_sub_strats_names.QUANTILE]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.POWER]:
        power_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=build_power_transformer(power_method, power_standardize),
            main_transformer_name=preproc_sub_strats_names.POWER,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value,
            add_indicator=True
        )
        transformers.append(
            (
                preproc_sub_strats_names.POWER,
                power_transformer,
                preprocessing_groups[preproc_sub_strats_names.POWER]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.BOUNDED]:
        bound_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=MinMaxScaler(),
            main_transformer_name=preproc_sub_strats_names.BOUNDED,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value,
            add_indicator=True
        )
        transformers.append(
            (
                preproc_sub_strats_names.BOUNDED,
                bound_transformer,
                preprocessing_groups[preproc_sub_strats_names.BOUNDED]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.ROBUST]:
        rob_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=RobustScaler(),
            main_transformer_name=preproc_sub_strats_names.ROBUST,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value,
            add_indicator=True
        )
        transformers.append(
            (
                preproc_sub_strats_names.ROBUST,
                rob_transformer,
                preprocessing_groups[preproc_sub_strats_names.ROBUST]
            )
        )

    if preprocessing_groups[preproc_sub_strats_names.STANDARD]:
        std_transformer = _build_sub_strat_imputer_pipeline(
            main_transformer=StandardScaler(),
            main_transformer_name=preproc_sub_strats_names.STANDARD,
            imputer_name=imputer_name,
            imputer_strategy=num_imputer_strategy,
            fill_value=imputer_fill_value,
            add_indicator=True
        )
        transformers.append(
            (
                preproc_sub_strats_names.STANDARD,
                std_transformer,
                preprocessing_groups[preproc_sub_strats_names.STANDARD]
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=col_trans_remainder)


def build_feature_preprocessor(
    strategy: PreprocessingStrategy,
    feature_names: list[str],
    feature_df: pd.DataFrame,
    use_imputer_for_tree: bool
) -> ColumnTransformer:
    """Build preprocessing pipeline for selected ML strategy.

    Args:
        strategy (PreprocessingStrategy): Selected preprocessing strategy based on ML model family (e.g. TREE_BASED or DISTANCE_BASED).
        feature_names (list[str]): List of feature names to process.
        feature_df (pd.DataFrame): DataFrame containing feature data for statistics calculation.
        use_imputer_for_tree (bool): Whether to use SimpleImputer to impute missing values for tree-based models.

    Returns:
        Configured sklearn ColumnTransformer.

    Raises:
        ValueError: If unsupported strategy is provided.
    """
    match strategy:
        case PreprocessingStrategy.TREE_BASED:
            return build_tree_based_preprocessor(feature_names=feature_names, use_imputer=use_imputer_for_tree)
        case PreprocessingStrategy.DISTANCE_BASED:
            return build_distance_based_preprocessor(feature_df=feature_df, feature_names=feature_names)
        case _:
            raise ValueError(f"Unsupported preprocessing strategy: {strategy}")
