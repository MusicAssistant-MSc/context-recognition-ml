import pandas as pd
import numpy as np
from typing import Any

from data import file_loader
from src.utils import const

def load_music_metadata() -> dict[str, pd.DataFrame]:
    """
    Load the music metadata from the SiTunes dataset.

    Returns:
        dict[str, pd.DataFrame]: Mapping of music metadata CSV file names to loaded data.
    """
    music_metadata_files = {}

    for file in const.SITUNES_MUSIC_METADATA_DIR.glob('*.csv'):
        if file.is_file():
            music_metadata_files[file.stem] = file_loader.load_csv_data(file)
    
    return music_metadata_files


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
    loaded_data['environment'] = file_loader.load_json_data(stage_dir / const.SITUNES_STAGE_FILES['environment'])
    loaded_data['interactions'] = file_loader.load_csv_data(stage_dir / const.SITUNES_STAGE_FILES['interactions'])
    loaded_data['wrist'] = file_loader.load_npy_data(stage_dir / const.SITUNES_STAGE_FILES['wrist'])

    return loaded_data


