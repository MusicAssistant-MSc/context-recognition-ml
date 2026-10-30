import logging
from datetime import datetime

from src.models.experiment_orchestrator import ExperimentOrchestrator
from src.models.const import ModelAlgorithm
from src.data.file_loader import load_csv_data
from src.utils.const import (
    PROCESSED_DATA_DIR, 
    PROCESSED_FEATURES_DATASET_FILENAME, 
    LOGS_DIR,
    EXPERIMENT_RUN_LOG_FILENAME,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)


def run_full_experiment() -> None:
    """Run full experiment for all algorithms."""
    # Setup logging to both console and file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOGS_DIR / f"{EXPERIMENT_RUN_LOG_FILENAME}_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
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
    run_full_experiment()