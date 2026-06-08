import json
import joblib
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Final
from datetime import datetime
from dataclasses import asdict

from src.models.const import (
    ExperimentResult,
    MODEL_EXT, 
    METRICS_EXT, 
    SUMMARY_CSV_NAME, 
    SAVED_MODELS_DIR, 
    REPORTS_DIR, 
    METRICS_DIR
)

logger = logging.getLogger(__name__)


def save_model(
    model: Any,
    model_name: str,
    output_dir: Path = SAVED_MODELS_DIR,
    model_ext: str = MODEL_EXT
) -> Path:
    """Save a trained model or pipeline to a joblib file.

    Args:
        model (Any): The fitted model or pipeline to save.
        model_name (str): Unique name for the model file.
        output_dir (Path): The directory where the model file should be saved.

    Returns:
        Path: The path to the saved model file.
    """
    _ensure_dir_exists(output_dir)
    file_path = output_dir / f"{model_name}{model_ext}"

    logger.info(f"Saving model to {file_path}...")
    joblib.dump(model, file_path)

    return file_path


def load_model(file_path: Path) -> Any:
    """Load a model or pipeline from a joblib file.

    Args:
        file_path (Path): The path to the joblib file containing the model or pipeline.

    Returns:
        Any: The loaded model or pipeline.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at {file_path}")
    
    logger.info(f"Loading model from {file_path}...")
    return joblib.load(file_path)
    

def save_experiment_results(
    experiment_result: ExperimentResult,
    experiment_name: str,
    metrics_output_dir: Path = METRICS_DIR,
    metrics_ext: str = METRICS_EXT,
    summary_dir: Path | None = REPORTS_DIR,
    summary_csv_name: str = SUMMARY_CSV_NAME
) -> None:
    """
    Save detailed experiment metrics and parameters to JSON.
    Optionally update a global summary CSV file.

    Args:
        experiment_result (ExperimentResult): The experiment results object containing metrics and parameters to save.
        experiment_name (str): The name of the experiment.
        metrics_output_dir (Path): The directory for detailed JSON report.
        metrics_ext (str): The file extension for the JSON report.
        summary_dir (Path | None): Optional directory where the global summary CSV file should be saved.
        summary_csv_name (str): The name of the global summary CSV file.

    Returns:
        None
    """
    _ensure_dir_exists(metrics_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = metrics_output_dir / f"{experiment_name}_{timestamp}{metrics_ext}"

    logger.info(f"Saving experiment results to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(experiment_result), f, indent=4)

    if summary_dir:
        _update_summary_results(experiment_result, summary_dir, summary_csv_name)


def _update_summary_results(
    results: ExperimentResult,
    summary_dir: Path,
    summary_csv_name: str
) -> None:
    """Append experiment results to the global summary CSV file.

    Args:
        experiment_result (ExperimentResult): The experiment results object containing metrics and parameters to save.
        summary_dir (Path): The directory where the global summary CSV file is saved.
        summary_csv_name (str): The name of the global summary CSV file.

    Returns:
        None
    """
    _ensure_dir_exists(summary_dir)
    summary_path = summary_dir / summary_csv_name

    summary_row = {
        "Algorithm": results.algorithm.value,
        "Variant": results.variant_name,
        "Timestamp": results.timestamp,
        "Duration (s)": results.duration_seconds,
        "Features_Count": len(results.feature_names)
    }

    for k, v in results.mean_metrics.items():
        summary_row[f"mean_{k}"] = v
    for k, v in results.std_metrics.items():
        summary_row[f"std_{k}"] = v

    df_row = pd.DataFrame([summary_row])
    
    header = not summary_path.exists()
    df_row.to_csv(summary_path, mode="a", index=False, header=header)
    logger.info(f"Experiment summary updated in {summary_path}")


def _ensure_dir_exists(dir_path: Path) -> None:
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")