import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Any
from sklearn.model_selection import RandomizedSearchCV

from src.models.const import (
    ModelAlgorithm, 
    CrossValidationStrategy, 
    FoldResult,
    ExperimentResult,
    RegressionMetrics,
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


def _setup_training_environment() -> None:
    """
    Configure the training environment to silence known non-critical noise.
    
    Specifically address:
    1. Scikit-learn warning regarding feature names when transitioning 
       between DataFrames and NumPy arrays in SVR/Pipelines.
    2. Joblib/Scikit-learn parallel processing thread configuration warning.
    """
    # Fix for SVR/Scikit-learn known issue: 
    # "X does not have valid feature names, but SVR was fitted with feature names"
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*does not have valid feature names, but.*was fitted with feature names.*"
    )
    # Fix for joblib/sklearn race condition warning during parallel training (e.g., Random Forest):
    # "sklearn.utils.parallel.delayed should be used with sklearn.utils.parallel.Parallel..."
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*sklearn.utils.parallel.delayed.*should be used with.*sklearn.utils.parallel.Parallel.*"
    )


def _aggregate_folds_results(
    fold_results: list[FoldResult],
    algorithm: ModelAlgorithm,
    variant_name: str,
    feature_names: list[str],
    duration_seconds: float,
    mae_metric_name: str = RegressionMetrics.MAE,
    mse_metric_name: str = RegressionMetrics.MSE,
    rmse_metric_name: str = RegressionMetrics.RMSE
) -> ExperimentResult:
    """Aggregate fold results into an ExperimentResult object.
    
    Args:
        fold_results (list[FoldResult]): A list of FoldResult objects.
        algorithm (ModelAlgorithm): The model algorithm used in the experiment.
        variant_name (str): The name of the feature set used in the experiment.
        feature_names (list[str]): A list of feature names used in the experiment.
        mae_metric_name (str, optional): The name of the MAE metric. Defaults to RegressionMetrics.MAE.
        mse_metric_name (str, optional): The name of the MSE metric. Defaults to RegressionMetrics.MSE.
        rmse_metric_name (str, optional): The name of the RMSE metric. Defaults to RegressionMetrics.RMSE.

    Returns:
        ExperimentResult: An ExperimentResult object containing the aggregated results.
    """
    maes = [fold.metrics[mae_metric_name] for fold in fold_results]
    mses = [fold.metrics[mse_metric_name] for fold in fold_results]
    rmses = [fold.metrics[rmse_metric_name] for fold in fold_results]

    mean_metrics = {
        mae_metric_name: float(np.mean(maes)), 
        mse_metric_name: float(np.mean(mses)), 
        rmse_metric_name: float(np.mean(rmses))
    }
    std_metrics = {
        mae_metric_name: np.std(maes), 
        mse_metric_name: np.std(mses), 
        rmse_metric_name: np.std(rmses)
    }

    return ExperimentResult(
        algorithm=algorithm, 
        variant_name=variant_name, 
        feature_names=feature_names, 
        folds=fold_results, 
        mean_metrics=mean_metrics, 
        std_metrics=std_metrics,
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration_seconds
    )


def train_and_evaluate_model(
    algorithm: ModelAlgorithm,
    preprocessing_strategy: PreprocessingStrategy,
    data: pd.DataFrame,
    feature_names: list[str],
    variant_name: str,
    target_col: str = TARGET_COL,
    group_split_col: str = USER_ID_COL,
    cv_outer_strategy: CrossValidationStrategy = DEFAULT_OUTER_CV_STRATEGY,
    cv_outer_splits: int = DEFAULT_OUTER_CV_SPLITS,
    cv_inner_strategy: CrossValidationStrategy = DEFAULT_INNER_CV_STRATEGY,
    cv_inner_splits: int = DEFAULT_INNER_CV_SPLITS,
    mae_metric_name: str = RegressionMetrics.MAE
) -> ExperimentResult:
    """Train and evaluate a model using nested cross-validation.
    
    Args:
        algorithm (ModelAlgorithm): The machine learning algorithm to use.
        preprocessing_strategy (PreprocessingStrategy): The strategy for feature preprocessing.
        data (pd.DataFrame): The input dataset containing features and target variable.
        feature_names (list[str]): List of feature column names to use for training.
        variant_name (str): Name of the experiment variant.
        target_col (str): The name of the target variable column in the dataset.
        group_split_col (str): The name of the column used for group splitting in cross-validation.
        cv_outer_strategy (CrossValidationStrategy): The strategy for outer cross-validation.
        cv_outer_splits (int): Number of splits for outer cross-validation.
        cv_inner_strategy (CrossValidationStrategy): The strategy for inner cross-validation.
        cv_inner_splits (int): Number of splits for inner cross-validation.

    Returns:
        ExperimentResult: An ExperimentResult object containing the results of the experiment.
    """
    # Configure training environment
    _setup_training_environment()
    # Prepare features, target and groups for cross-validation
    data_cp = data.copy()
    X = data_cp[feature_names]
    y = data_cp[target_col]
    groups = data_cp[group_split_col]

    # Get Nested CV splitters
    outer_cv, inner_cv = get_nested_cv_setup(
        outer_strategy=cv_outer_strategy,
        inner_strategy=cv_inner_strategy,
        n_splits_outer=cv_outer_splits,
        n_splits_inner=cv_inner_splits
    )

    # Store results for each outer fold
    fold_results: list[FoldResult] = []

    logger.info(f"STARTING EXPERIMENT: {algorithm.value} | Variant: {variant_name}")
    logger.info(f"Features count: {len(feature_names)}")
    start_time = time.time()

    # Outer loop for performance estimation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        logger.info(f"Fold {fold_idx + 1}/{outer_cv.get_n_splits()} - Training...")
        
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

        # Get the best model and parameters from the search
        best_model = tuned_search.best_estimator_
        best_params = tuned_search.best_params_

        # Evaluate on the test set
        predictions = best_model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, predictions)
        
        fold_results.append(FoldResult(
            fold=fold_idx + 1,
            metrics=metrics,
            best_params=best_params
        ))
        logger.info(f"Fold {fold_idx + 1} Results: MAE={metrics[mae_metric_name]:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds.")

    return _aggregate_folds_results(fold_results, algorithm, variant_name, feature_names, total_time)