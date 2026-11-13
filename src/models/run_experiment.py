import logging

from src.utils.logging_config import setup_logging
from src.models.experiment_orchestrator import ExperimentOrchestrator
from src.models.const import ModelAlgorithm
from src.data.file_loader import load_csv_data
from src.utils.const import (
    PROCESSED_DATA_DIR, 
    PROCESSED_FEATURES_DATASET_FILENAME, 
    LOGS_DIR,
    EXPERIMENT_RUN_LOG_FILENAME,
)
from src.utils.time_utils import get_current_timestamp_for_filename

logger = logging.getLogger(__name__)


def _setup_logging_for_experiment() -> None:
    """Setup logging configuration for the experiment run."""
    timestamp = get_current_timestamp_for_filename()
    log_filename = f"{EXPERIMENT_RUN_LOG_FILENAME}_{timestamp}.log"
    log_file_path = LOGS_DIR / log_filename
    setup_logging(log_file=log_file_path)


def run_full_experiment() -> None:
    """Run full experiment for all algorithms."""
    logger.info(' --- Starting full experiment ---')

    # Load processed dataset
    data_path = PROCESSED_DATA_DIR / PROCESSED_FEATURES_DATASET_FILENAME
    data = load_csv_data(data_path)

    # Extract all ModelAlgorithms into list
    all_experiment_algorithms: list[ModelAlgorithm] = list(ModelAlgorithm)

    # Create ExperimentOrchestrator and run full experiment
    experiment_orchestrator = ExperimentOrchestrator(data)
    experiment_orchestrator.run_full_experiment(algorithms=all_experiment_algorithms)


if __name__ == "__main__":
    _setup_logging_for_experiment()
    run_full_experiment()