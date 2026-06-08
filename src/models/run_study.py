import logging
from datetime import datetime

from src.models.ablation_study import AblationOrchestrator
from src.models.const import ModelAlgorithm
from src.data.file_loader import load_csv_data
from src.utils.const import (
    PROCESSED_DATA_DIR, 
    PROCESSED_FEATURES_DATASET_FILENAME, 
    LOGS_DIR,
    ABLATION_STUDY_LOG_FILENAME,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)


def run_ablation_study() -> None:
    """Run ablation study for all algorithms."""
    # Setup logging to both console and file
    timestamp = datetime.now().isoformat()
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOGS_DIR / f"{ABLATION_STUDY_LOG_FILENAME}_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    logger.info(' --- Starting ablation study ---')

    # Load processed dataset
    data_path = PROCESSED_DATA_DIR / PROCESSED_FEATURES_DATASET_FILENAME
    data = load_csv_data(data_path)

    # Extract all ModelAlgorithms into list
    all_ablation_algorithms: list[ModelAlgorithm] = list(ModelAlgorithm)

    # Create AblationOrchestrator and run study
    ablation_orchestrator = AblationOrchestrator(data)
    ablation_orchestrator.run_full_study(algorithms=all_ablation_algorithms)


if __name__ == "__main__":
    run_ablation_study()