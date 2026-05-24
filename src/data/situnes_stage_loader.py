from pathlib import Path
from dataclasses import dataclass
import re
import pandas as pd
import numpy as np
from typing import Any

from src.data import file_loader
from src.data import const


@dataclass
class StageDataMapping:
    environment: str
    interactions: str
    wrist: str


def load_stage_data(
    stage: str,
    data_dir: Path = const.SITUNES_DATA_DIR, 
    stage_config: dict = const.SITUNES_STAGE_DATA_CONFIG, 
    filename_key: str = const.SitunesConfigKeys.FILENAME
) -> dict[str, Any]:
    """Load the SiTunes dataset for a specific stage.

    Args:
        stage (str): The stage of the SiTunes dataset to load (e.g., 'Stage2', 'Stage3').
        data_dir (Path): The directory containing the stage data.
        stage_config (dict): The configuration dictionary for the stage data.
        filename_key (str): The key in the stage configuration that specifies the filename for each data type.
    
    Returns:
        dict: A dictionary containing the loaded data for the specified stage.
    """
    stage_dir = data_dir / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"SiTunes stage directory not found: {stage_dir}")
    
    loaded_data = {}

    # Load environment, interactions, and wrist data for the specified stage
    for key, data_config in stage_config.items():
        filename = data_config[filename_key]
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


def _get_stage_dirs(stage_list: list[str], data_dir: Path) -> list[Path]:
    """Get the list of available stage directories in the SiTunes dataset.
    
    Args:
        stage_list (list[str]): A list of stage names to look for (e.g., ['Stage2', 'Stage3']).
        data_dir (Path): The directory containing the stage data.

    Returns:
        list[Path]: A list of stage directory paths (e.g., ['Stage2', 'Stage3']).
    """
    stage_dirs = []
    for stage in stage_list:
        stage_dir = data_dir / stage
        if stage_dir.is_dir():
            stage_dirs.append(stage_dir)
    return stage_dirs


def _generate_unique_interaction_ids(
    inter_df: pd.DataFrame, 
    stage: str, 
    stage_num_extraction_pattern: str,
    pkey_col: str
) -> pd.Series:
    """Generate unique interaction IDs for the interactions DataFrame.

    Args:
        inter_df (pd.DataFrame): The interactions DataFrame to process.
        stage (str): The stage of the SiTunes dataset (e.g., 'Stage2', 'Stage3')
        stage_num_extraction_pattern (str): The regex pattern to extract the stage number from the stage name.
        pkey_col (str): The name of the primary key column in the interactions DataFrame.

    Returns:
        pd.Series: A series containing the unique interaction IDs with stage information.
    """
    if pkey_col not in inter_df.columns:
        raise ValueError(f"Primary key column '{pkey_col}' not found in interactions DataFrame for {stage}")
    
    stage_num = int(re.search(stage_num_extraction_pattern, stage).group(1)) # Extract the stage number
    return inter_df[pkey_col].apply(lambda x: f"S{stage_num}_I{x}")


def _flatten_environment_data(
    env_data: dict, 
    env_conf_col_map: dict[str, list[str]],
    pkey_col: str
) -> pd.DataFrame:
    """Flatten the nested environment data into a DataFrame.

    Args:
        env_data (dict): The nested environment data to flatten.
        env_conf_col_map (dict): The column mapping for the environment data.
        pkey_col (str): The primary key column name.

    Returns:
        pd.DataFrame: A flattened DataFrame containing the environment data.
    """
    flattened_data = []

    # Create a flat record for each interaction
    for inter_id, inter_values in env_data.items():
        # Initialize the record with the primary key value
        record = {pkey_col: inter_id}

        # Flatten the nested environment data based on the column mapping in the configuration
        for category, col_list in env_conf_col_map.items():
            category_data = inter_values.get(category)
            if isinstance(category_data, list):
                for i, col_name in enumerate(col_list):
                    record[col_name] = category_data[i] if i < len(category_data) else None
            else: # Handle non-list values (e.g., single time value)
                if col_list:
                    record[col_list[0]] = category_data

        flattened_data.append(record)
    
    return pd.DataFrame(flattened_data)


def _clean_environment_data(
    env_df: pd.DataFrame, 
    rename_cols: dict[str, str],
    drop_cols: list[str]
) -> pd.DataFrame:
    """Clean the environment DataFrame by renaming columns and dropping unnecessary ones.

    Args:
        env_df (pd.DataFrame): The environment DataFrame to clean.
        rename_cols (dict): A dictionary mapping original column names to new column names.
        drop_cols (list): A list of column names to drop from the DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing the environment data.
    """
    # Rename columns according to the configuration
    env_df = env_df.rename(columns=rename_cols)
    
    # Drop unnecessary columns
    env_df = env_df.drop(columns=drop_cols, errors='ignore')

    return env_df


def merge_interactions_with_environment_data(
    inter_df: pd.DataFrame, 
    env_df: pd.DataFrame, 
    inter_config: dict, 
    env_config: dict,
    pkey_col_key: str
) -> pd.DataFrame:
    """Merge the interactions DataFrame with the environment DataFrame based on the primary key column.
    
    Args:
        inter_df (pd.DataFrame): The interactions DataFrame to merge.
        env_df (pd.DataFrame): The environment DataFrame to merge.
        inter_config (dict): The configuration dictionary for the interactions data.
        env_config (dict): The configuration dictionary for the environment data.
        pkey_col_key (str): The key in the interactions and environment configuration 
            that specifies the primary key column name.

    Returns:
        pd.DataFrame: A merged DataFrame containing both interactions and environment data.
    """
    inter_pkey_col = inter_config[pkey_col_key]
    env_pkey_col = env_config[pkey_col_key]

    inter_df = inter_df.copy()
    env_df = env_df.copy()

    inter_df[inter_pkey_col] = inter_df[inter_pkey_col].astype(str).str.strip()
    env_df[env_pkey_col] = env_df[env_pkey_col].astype(str).str.strip()

    if inter_pkey_col not in inter_df.columns:
        raise ValueError(f"Primary key column '{inter_pkey_col}' not found in interactions DataFrame")
    if env_pkey_col not in env_df.columns:
        raise ValueError(f"Primary key column '{env_pkey_col}' not found in environment DataFrame")

    merged_df = pd.merge(inter_df, env_df, left_on=inter_pkey_col, right_on=env_pkey_col, how='left')
    return merged_df


def merge_wrist_data(wrist_data_list: list[np.ndarray]) -> np.ndarray:
    """Merge the wrist data from all stages into a single NumPy array.

    Args:
        wrist_data_list (list[np.ndarray]): A list of wrist data arrays from different stages.

    Returns:
        np.ndarray: A merged NumPy array containing all wrist data.
    """
    if not wrist_data_list:
        return np.array([])  # Return an empty array if there is no wrist data
    
    merged_wrist_data = np.concatenate(wrist_data_list, axis=0)
    return merged_wrist_data


def process_environment_data(
    env_data: dict, 
    env_config: dict,
    pkey_col_key: str = const.SitunesConfigKeys.PKEY_COL,
    cols_key: str = const.SitunesConfigKeys.COLS,
    rename_cols_key: str = const.SitunesConfigKeys.RENAME_COLS,
    drop_cols_key: str = const.SitunesConfigKeys.DROP_COLS
) -> pd.DataFrame:
    """Process the environment data by flattening and cleaning it.

    Args:
        env_data (dict): The nested environment data to process.
        env_config (dict): The configuration dictionary for the environment data.
        pkey_col_key (str): The key in the environment configuration that specifies the primary key column name.
        cols_key (str): The key in the environment configuration that specifies the column mapping 
            for flattening the environment data.
        rename_cols_key (str): The key in the environment configuration that specifies the column renaming mapping.
        drop_cols_key (str): The key in the environment configuration that specifies the list of columns

    Returns:
        pd.DataFrame: A processed DataFrame containing the environment data.
    """
    # Extract configuration values for processing environment data
    pkey_col = env_config[pkey_col_key]
    col_map = env_config.get(cols_key, {})
    rename_cols = env_config.get(rename_cols_key, {})
    drop_cols = env_config.get(drop_cols_key, [])

    flattened_env_df = _flatten_environment_data(env_data, col_map, pkey_col)
    cleaned_env_df = _clean_environment_data(flattened_env_df, rename_cols, drop_cols)
    return cleaned_env_df


def process_stage_data(
    stage_data: dict, 
    stage: str,
    stage_config: dict = const.SITUNES_STAGE_DATA_CONFIG,
    mapping: StageDataMapping = StageDataMapping(
        environment=const.SitunesStageDataKeys.ENVIRONMENT,
        interactions=const.SitunesStageDataKeys.INTERACTIONS,
        wrist=const.SitunesStageDataKeys.WRIST
    ),
    pkey_col_key: str = const.SitunesConfigKeys.PKEY_COL,
    stage_num_extraction_pattern: str = const.SITUNES_STAGE_NUM_EXTRACTION_PATTERN
) -> tuple[pd.DataFrame, np.ndarray]:
    """Process the data for a specific stage by processing environment data, and merging it with interactions. 

    Args:
        stage_data (dict): The raw data for the stage to process.
        stage_config (dict): The configuration dictionary for the stage data.
        mapping (StageDataMapping): The mapping of data keys for environment, interactions, and wrist data.
        pkey_col_key (str): The key in the interactions configuration that specifies the primary key column name.
        stage_num_extraction_pattern (str): The regex pattern to extract the stage number from the stage name.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the processed DataFrame with merged interactions 
            and environment data, and loaded wrist data array.
    """
    inter_df = stage_data.get(mapping.interactions)
    env_data = stage_data.get(mapping.environment)
    wrist_data = stage_data.get(mapping.wrist)

    inter_config = stage_config.get(mapping.interactions, {})
    env_config = stage_config.get(mapping.environment, {})

    # Process environment data
    env_df = process_environment_data(env_data, env_config)
    # Merge interactions with environment data
    merged_df = merge_interactions_with_environment_data(inter_df, env_df, inter_config, env_config, pkey_col_key)
    # Generate unique interaction IDs across stages and replace the original interaction IDs with the new unique IDs
    inter_pkey_col_name = inter_config[pkey_col_key]
    merged_df[inter_pkey_col_name] = _generate_unique_interaction_ids(
        merged_df, 
        stage, 
        stage_num_extraction_pattern,
        inter_pkey_col_name
    )
    
    return merged_df, wrist_data


def load_merge_stage_data(
    stages: list[str] = const.SITUNES_STAGES, 
    stage_config: dict = const.SITUNES_STAGE_DATA_CONFIG
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge the data from multiple stages into a single DataFrame and wrist data array.

    Args:
        stages (list[str]): A list of stage names to merge (e.g., ['Stage2', 'Stage3']).
        stage_config (dict): The configuration dictionary for the stage data.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the merged DataFrame with interactions and environment data, 
            and the merged wrist data array.
    """
    all_stage_data = []
    all_wrist_data = []

    for stage in stages:
        stage_data = load_stage_data(stage)
        inter_env_df, wrist_data = process_stage_data(stage_data, stage, stage_config)

        all_stage_data.append(inter_env_df)
        all_wrist_data.append(wrist_data)

    merged_stage_df = pd.concat(all_stage_data, ignore_index=True)
    merged_wrist_data = merge_wrist_data(all_wrist_data)

    return merged_stage_df, merged_wrist_data
    