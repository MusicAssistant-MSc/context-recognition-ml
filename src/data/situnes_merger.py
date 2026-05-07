import pandas as pd
import numpy as np
from typing import Any

from data import file_loader
from src.utils import const
from src.utils.data_utils import check_column_uniqueness, get_shared_columns

def load_music_metadata() -> dict[str, pd.DataFrame]:
    """
    Load the music metadata from the SiTunes dataset.

    Returns:
        dict[str, pd.DataFrame]: Mapping of music metadata CSV filenames to loaded data.
    """
    music_metadata_filenames = [file_cfg['filename'] for file_cfg in const.SITUNES_MUSIC_METADATA_CONFIG.values()]
    music_metadata_files = {}

    for file in const.SITUNES_MUSIC_METADATA_DIR.glob('*.csv'):
        if file.is_file() and file.name in music_metadata_filenames:
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


def merge_music_metadata(music_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the music metadata DataFrames on shared columns.

    Args:
        music_dfs (dict[str, pd.DataFrame]): A dictionary of music metadata DataFrames.

    Returns:
        pd.DataFrame: The merged music metadata DataFrame.
    """
    merged_df = None

    for name, df in music_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            fingerprint_cols = get_shared_columns(merged_df, df)
            fignerprint_cols_uniqueness = check_column_uniqueness(df, fingerprint_cols)

            if fignerprint_cols_uniqueness['is_unique']:
                merged_df = pd.merge(left=merged_df, right=df, on=fingerprint_cols, how='left', suffixes=('', f'_{name}'))
    
    return merged_df