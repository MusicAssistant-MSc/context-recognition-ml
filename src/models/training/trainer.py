import logging
import pandas as pd
from typing import Any
from sklearn.model_selection import RandomizedSearchCV

from src.models.const import (
    ModelAlgorithm, 
    CrossValidationStrategy, 
    USER_ID_COL, 
    TARGET_COL,
    DEFAULT_OUTER_CV_STRATEGY,
    DEFAULT_OUTER_CV_SPLITS,
    DEFAULT_INNER_CV_STRATEGY,
    DEFAULT_INNER_CV_SPLITS
)
from src.models.training import model_factory, hp_tuner, pipeline_builder
from src.models.cross_validation import get_nested_cv_setup
from src.models.metrics import calculate_regression_metrics
from src.models.utils import use_imputer_for_tree_model
from src.features.preprocessing import build_feature_preprocessor, PreprocessingStrategy

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    algorithm: ModelAlgorithm,
    preprocessing_strategy: PreprocessingStrategy,
    data: pd.DataFrame,
    feature_names: list[str],
    target_col: str = TARGET_COL,
    group_split_col: str = USER_ID_COL,
    cv_outer_strategy: CrossValidationStrategy = DEFAULT_OUTER_CV_STRATEGY,
    cv_outer_splits: int = DEFAULT_OUTER_CV_SPLITS,
    cv_inner_strategy: CrossValidationStrategy = DEFAULT_INNER_CV_STRATEGY,
    cv_inner_splits: int = DEFAULT_INNER_CV_SPLITS
) -> list[dict[str, Any]]:
    """Train and evaluate a model using nested cross-validation.
    
    Args:
        algorithm (ModelAlgorithm): The machine learning algorithm to use.
        preprocessing_strategy (PreprocessingStrategy): The strategy for feature preprocessing.
        data (pd.DataFrame): The input dataset containing features and target variable.
        feature_names (list[str]): List of feature column names to use for training.
        target_col (str): The name of the target variable column in the dataset.
        group_split_col (str): The name of the column used for group splitting in cross-validation.
        cv_outer_strategy (CrossValidationStrategy): The strategy for outer cross-validation.
        cv_outer_splits (int): Number of splits for outer cross-validation.
        cv_inner_strategy (CrossValidationStrategy): The strategy for inner cross-validation.
        cv_inner_splits (int): Number of splits for inner cross-validation.

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing evaluation metrics for each outer fold.
    """
    # Prepare features, target and groups for cross-validation
    X = data[feature_names]
    y = data[target_col]
    groups = data[group_split_col]

    # Get Nested CV splitters
    outer_cv, inner_cv = get_nested_cv_setup(
        outer_strategy=cv_outer_strategy,
        inner_strategy=cv_inner_strategy,
        n_splits_outer=cv_outer_splits,
        n_splits_inner=cv_inner_splits
    )

    # Store results for each outer fold
    results = []

    # Outer loop for performance estimation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        logger.info(f"--- Outer fold {fold_idx + 1}/{outer_cv.get_n_splits()} ---")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        # Build feature preprocessor using only current train fold
        preprocessor = build_feature_preprocessor(
            strategy=preprocessing_strategy,
            feature_names=feature_names,
            feature_df=X_train,
            use_imputer_for_tree=use_imputer_for_tree_model(algorithm)
        )

        # Build model pipeline
        model = model_factory.get_unfitted_model(algorithm)
        model_pipeline = pipeline_builder.build_model_pipeline(preprocessor, model)

        # Tune hyperparameters using inner CV
        tuned_search: RandomizedSearchCV = hp_tuner.tune_hyperparameters(
            pipeline=model_pipeline,
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            groups=groups_train,
            cv_splitter=inner_cv
        )

        # Get the best model from the search
        best_model = tuned_search.best_estimator_

        # Evaluate on the test set
        predictions = best_model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, predictions)
        
        logger.info(f"Outer fold {fold_idx + 1} - Metrics: {metrics}")
        
        results.append({
            "fold": fold_idx + 1,
            "metrics": metrics,
            "best_params": tuned_search.best_params_
        })

    return results