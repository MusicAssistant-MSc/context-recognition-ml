import json
from pathlib import Path
from typing import Any, Literal
import re
import logging
import pandas as pd

from src.models.const import RegressionMetrics, METRICS_DIR, FILE_TIMESTAMP_SUFFIX_PATTERN

logger = logging.getLogger(__name__)


def load_experimental_results_metrics(
    metrics_dir: Path = METRICS_DIR,
    metrics_names: list[str] = list(RegressionMetrics),
    timestamp_suffix_pattern: str = FILE_TIMESTAMP_SUFFIX_PATTERN
) -> dict[str, dict[str, list[float]]]:
    """Load both aggregated and non-aggregated experimental results metrics from a directory of JSON files.
    
    Args:
        metrics_dir (Path): The directory containing JSON files with metrics.
            Defaults to METRICS_DIR.
        metrics_names (list[str]): A list of metric names to extract. 
            Defaults to all available metrics.

    Returns:
        dict[str, dict[str, list[float]]]: A dictionary where keys are 
            experiment names and values are dictionaries of extracted metrics.
    """
    all_model_results: dict[str, list[float]] = {}

    # Iterate over all JSON files
    for json_file in metrics_dir.glob("*.json"):
        metrics = _load_metrics_from_json(json_file, metrics_names)
        # Extract experiment name from JSON filename
        file_stem: str = json_file.stem
        experiment_name: str = re.sub(timestamp_suffix_pattern, "", file_stem)
        
        all_model_results[experiment_name] = metrics

    return all_model_results


def get_metric_matrix(
    all_results: dict[str, dict[str, list[float]]], 
    target_metric: str
) -> pd.DataFrame:
    """
    Extract a specific metric for all experiments into a DataFrame.

    This matrix is formatted for statistical tests:
    - Columns: Experiment names (Models)
    - Rows: Fold indices

    Args:
        all_results (dict[str, dict[str, list[float]]]): Output from load_experimental_results_metrics.
        target_metric (str): The name of the metric to extract (e.g., 'mae').

    Returns:
        pd.DataFrame: A DataFrame where each column is a model and each row is a fold.
    """
    extracted_data: dict[str, list[float]] = {}

    for exp_name, metrics_dict in all_results.items():
        if target_metric in metrics_dict:
            extracted_data[exp_name] = metrics_dict[target_metric]
        else:
            logger.warning(f"Metric '{target_metric}' not found for experiment '{exp_name}'. Skipping...")
            continue

    return pd.DataFrame(extracted_data)


def aggregate_metrics_for_main_effects(
    results_matrix: pd.DataFrame,
    group_by: Literal["algorithm", "variant"]
) -> pd.DataFrame:
    """Aggregate full metrics results matrix into results grouped by algorithms or variants.

    Args:
        results_matrix: Full metrics DataFrame where columns are 'algorithm_variant'.
        group_by: Dimension to aggregate by.

    Returns:
        pd.DataFrame: Aggregated Nxk matrix.
    """
    results_copy = results_matrix.copy()
    # Parse column names into MultiIndex [Algorithm, Variant]
    parsed_cols = [col.rsplit('_', 1) for col in results_copy.columns]
    results_copy.columns = pd.MultiIndex.from_tuples(
        parsed_cols, names=['algorithm', 'variant']
    )

    # Group and calculate mean across the other dimension
    if group_by == "algorithm":
        # Mean of all variants for each algorithm per fold
        aggregated = results_copy.T.groupby(level='algorithm').mean().T
    else:
        # Mean of all algorithms for each variant per fold
        aggregated = results_copy.T.groupby(level='variant').mean().T
    
    logger.info(f"Aggregated matrix for '{group_by}' impact created. Shape: {aggregated.shape}")
    return aggregated


def _load_metrics_from_json(
    file_path: Path,
    metrics_names: list[str]
) -> dict[str, list[float]]:
    """
    Extract specific metrics from a JSON result file for each fold.

    Args:
        file_path (Path): Path to the JSON file containing model results.
        metrics_names (list[str]): List of metric names to extract.

    Returns:
        dict[str, list[float]]: A dictionary where keys are metric names 
            and values are lists of metric values across all folds.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    return _extract_fold_metrics(data, metrics_names)


def _extract_fold_metrics(
    data: dict[str, Any],
    metrics_names: list[str]
) -> dict[str, list[float]]:
    """
    Private helper to parse the 'folds' section of the result JSON.

    Args:
        data (dict[str, Any]): The raw dictionary from the JSON file.
        metrics_names (list[str]): List of metric names to extract.

    Returns:
        dict[str, list[float]]: Extracted metrics organized by name.
    """
    extracted_metrics: dict[str, list[float]] = {metric: [] for metric in metrics_names}

    # Iterate through folds and collect metrics
    for fold_data in data.get("folds", []):
        metrics_dict: dict[str, float] = fold_data.get("metrics", {})
        for metric_name in metrics_names:
            value = metrics_dict.get(metric_name)
            if value is not None:
                extracted_metrics[metric_name].append(value)

    return extracted_metrics