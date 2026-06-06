import logging
from typing import Any
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.const import ModelAlgorithm, RANDOMIZED_SEARCH_ITER, RANDOM_STATE, RANDOMIZED_SEARCH_SCORING, RANDOMIZED_SEARCH_N_JOBS, HYPERPARAMETER_MODEL_PREFIX
from src.models.hyperparameters.registry import HYPERPARAMETERS_REGISTRY

logger = logging.getLogger(__name__)


def _add_pipeline_prefix(
    params: dict[str, Any],
    prefix: str = HYPERPARAMETER_MODEL_PREFIX
) -> dict[str, Any]:
    return {f"{prefix}__{key}": value for key, value in params.items()}


def tune_hyperparameters(
    pipeline: Pipeline,
    algorithm: ModelAlgorithm,
    X_train: Any,
    y_train: Any,
    groups: Any,
    cv_splitter: Any,
    n_iter: int = RANDOMIZED_SEARCH_ITER,
    random_state: int = RANDOM_STATE,
    scoring: str = RANDOMIZED_SEARCH_SCORING,
    n_jobs: int = RANDOMIZED_SEARCH_N_JOBS
) -> RandomizedSearchCV:
    """Tune hyperparameters for a given model pipeline using RandomizedSearchCV.
    
    Args:
        pipeline (Pipeline): The sklearn ML pipeline containing the model to tune.
        algorithm (ModelAlgorithm): The algorithm type to determine which hyperparameters to tune.
        X_train (Any): Training features.
        y_train (Any): Training target variable.
        groups (Any): Group identifiers.
        cv_splitter (Any): Cross-validation splitter to use for tuning (inner strategy).
        n_iter (int): Number of parameter settings that are sampled in RandomizedSearchCV.
        random_state (int): Seed for reproducibility.
        scoring (str): Scoring metric for evaluating hyperparameters.
        n_jobs (int): Number of parallel jobs to run for hyperparameter tuning.

    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object with the best parameters found.
    """
    logger.info(f"Starting hyperparameter tuning for {algorithm.value}...")
    
    # Get the parameter grid for tuning and add prefixes
    param_grid = HYPERPARAMETERS_REGISTRY.get_tuning_params_grid(algorithm)
    param_grid = _add_pipeline_prefix(param_grid)

    if not param_grid:
        logger.warning(f"No tunable hyperparameters found for {algorithm.value}. Skipping tuning.")
        logger.info(f"Returning original pipeline for {algorithm.value}.")
        return pipeline
    
    # Perform randomized search for hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_splitter,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    search.fit(X_train, y_train, groups=groups)
    
    logger.info(f"Tuning completed. Best score: {search.best_score_:.4f}")
    logger.info(f"Best hyperparameters for {algorithm.value}: {search.best_params_}")
    
    return search