from typing import Any

from src.data import file_loader
from src.utils import const


def load_stage_data(stage: str) -> dict[str, Any]:
    """Load the SiTunes dataset for a specific stage.

    Args:
        stage (str): The stage of the SiTunes dataset to load (e.g., 'Stage2', 'Stage3').
    Returns:
        dict: A dictionary containing the loaded data for the specified stage.
    """
    stage_dir = const.SITUNES_DATA_DIR / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"SiTunes stage directory not found: {stage_dir}")
    
    loaded_data = {}

    # Load environment, interactions, and wrist data for the specified stage
    for key, filename in const.SITUNES_STAGE_FILES.items():
        file_path = stage_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Expected file '{filename}' not found in stage directory: {stage_dir}")
        
        match file_path.suffix:
            case '.json':
                loaded_data[key] = file_loader.load_json_data(file_path)
            case '.csv':
                loaded_data[key] = file_loader.load_csv_data(file_path)
            case '.npy':
                loaded_data[key] = file_loader.load_npy_data(file_path)
            case _:
                pass  # Unsupported file type, add logging if needed

    return loaded_data