from pathlib import Path

import pandas as pd
import numpy as np
from typing import Any

from src.data import file_loader
from src.utils import const


def load_stage_data(stage: str, stage_config: dict, data_dir: Path) -> dict[str, Any]:
    """Load the SiTunes dataset for a specific stage.

    Args:
        stage (str): The stage of the SiTunes dataset to load (e.g., 'Stage2', 'Stage3').
        stage_config (dict): The configuration dictionary for the stage data.
        data_dir (Path): The directory containing the stage data.
    
    Returns:
        dict: A dictionary containing the loaded data for the specified stage.
    """
    stage_dir = data_dir / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"SiTunes stage directory not found: {stage_dir}")
    
    loaded_data = {}

    # Load environment, interactions, and wrist data for the specified stage
    for key, data_config in stage_config.items():
        filename = data_config['filename']
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


def _generate_unique_interaction_ids(inter_df: pd.DataFrame, stage: str, inter_config: dict) -> pd.Series:
    """Generate unique interaction IDs for the interactions DataFrame.

    Args:
        inter_df (pd.DataFrame): The interactions DataFrame to process.
        stage (str): The stage of the SiTunes dataset (e.g., 'Stage2', 'Stage3')
        inter_config (dict): The configuration dictionary for the interactions.

    Returns:
        pd.Series: A series containing the unique interaction IDs with stage information.
    """
    pkey_col = inter_config['pkey_col']
    if pkey_col not in inter_df.columns:
        raise ValueError(f"Primary key column '{pkey_col}' not found in interactions DataFrame for {stage}")
    
    stage_num = int(stage.replace('Stage', '')) # Extract the stage number
    return inter_df[pkey_col].apply(lambda x: f"S{stage_num}_I{x}")


def _flatten_environment_data(env_data: dict, env_config: dict) -> pd.DataFrame:
    """Flatten the nested environment data into a DataFrame.

    Args:
        env_data (dict): The nested environment data to flatten.
        env_config (dict): The configuration dictionary for the environment data.

    Returns:
        pd.DataFrame: A flattened DataFrame containing the environment data.
    """
    flattened_data = []

    pkey_col = env_config['pkey_col']
    col_map = env_config.get('cols', {})

    # Create a flat record for each interaction
    for inter_id, values in env_data.items():
        # Initialize the record with the primary key value
        record = {pkey_col: inter_id}
        # Process single time value
        record[col_map['time'][0]] = values.get('time')
        # Process weather values
        weather_values = values.get('weather', [])
        for i, weather_col in enumerate(col_map['weather']):
            record[weather_col] = weather_values[i] if i < len(weather_values) else None
        # Process GPS values
        gps_values = values.get('GPS', [])
        for i, gps_col in enumerate(col_map['GPS']):
            record[gps_col] = gps_values[i] if i < len(gps_values) else None

        flattened_data.append(record)
    
    return pd.DataFrame(flattened_data)


def _clean_environment_data(env_df: pd.DataFrame, env_config: dict) -> pd.DataFrame:
    """Clean the environment DataFrame by renaming columns and dropping unnecessary ones.

    Args:
        env_df (pd.DataFrame): The environment DataFrame to clean.
        env_config (dict): The configuration dictionary for the environment data.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing the environment data.
    """
    rename_cols = env_config.get('rename_cols', {})
    drop_cols = env_config.get('drop_cols', [])

    # Rename columns according to the configuration
    env_df = env_df.rename(columns=rename_cols)
    
    # Drop unnecessary columns
    env_df = env_df.drop(columns=drop_cols, errors='ignore')

    return env_df


def merge_interactions_with_environment_data(
    inter_df: pd.DataFrame, 
    env_df: pd.DataFrame, 
    inter_config: dict, 
    env_config: dict
) -> pd.DataFrame:
    """Merge the interactions DataFrame with the environment DataFrame based on the primary key column.
    
    Args:
        inter_df (pd.DataFrame): The interactions DataFrame to merge.
        env_df (pd.DataFrame): The environment DataFrame to merge.
        inter_config (dict): The configuration dictionary for the interactions data.
        env_config (dict): The configuration dictionary for the environment data.

    Returns:
        pd.DataFrame: A merged DataFrame containing both interactions and environment data.
    """
    inter_pkey_col = inter_config['pkey_col']
    env_pkey_col = env_config['pkey_col']

    if inter_pkey_col not in inter_df.columns:
        raise ValueError(f"Primary key column '{inter_pkey_col}' not found in interactions DataFrame")
    if env_pkey_col not in env_df.columns:
        raise ValueError(f"Primary key column '{env_pkey_col}' not found in environment DataFrame")

    merged_df = pd.merge(inter_df, env_df, left_on=inter_pkey_col, right_on=env_pkey_col, how='left')
    return merged_df


def merge_wrist_data(wrist_data_list: list) -> np.ndarray:
    """Merge the wrist data from all stages into a single NumPy array.

    Args:
        wrist_data_list (list): A list of wrist data arrays from different stages.

    Returns:
        np.ndarray: A merged NumPy array containing all wrist data.
    """
    if not wrist_data_list:
        return np.array([])  # Return an empty array if there is no wrist data
    
    merged_wrist_data = np.concatenate(wrist_data_list, axis=0)
    return merged_wrist_data


def process_environment_data(env_data: dict, env_config: dict) -> pd.DataFrame:
    """Process the environment data by flattening and cleaning it.

    Args:
        env_data (dict): The nested environment data to process.
        env_config (dict): The configuration dictionary for the environment data.

    Returns:
        pd.DataFrame: A processed DataFrame containing the environment data.
    """
    flattened_env_df = _flatten_environment_data(env_data, env_config)
    cleaned_env_df = _clean_environment_data(flattened_env_df, env_config)
    return cleaned_env_df


def process_stage_data(
    stage_data: dict, 
    stage_config: dict = const.SITUNES_STAGE_DATA_CONFIG
) -> tuple[pd.DataFrame, np.ndarray]:
    """Process the data for a specific stage by processing environment data, and merging it with interactions. 

    Args:
        stage_data (dict): The raw data for the stage to process.
        stage_config (dict): The configuration dictionary for the stage data.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the processed DataFrame with merged interactions 
            and environment data, and loaded wrist data array.
    """
    inter_df = stage_data['interactions']
    env_data = stage_data['environment']
    wrist_data = stage_data['wrist']

    inter_config = stage_config['interactions']
    env_config = stage_config['environment']

    # Process environment data
    env_df = process_environment_data(env_data, env_config)
    # Merge interactions with environment data
    merged_df = merge_interactions_with_environment_data(inter_df, env_df, inter_config, env_config)
    
    return merged_df, wrist_data


def load_merge_stage_data(
    data_dir: Path = const.SITUNES_DATA_DIR, 
    stages: list[str] = const.SITUNES_STAGES, 
    stage_config: dict = const.SITUNES_STAGE_DATA_CONFIG
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge the data from multiple stages into a single DataFrame and wrist data array.

    Args:
        data_dir (Path): The directory containing the stage data.
        stages (list[str]): A list of stage names to merge (e.g., ['Stage2', 'Stage3']).
        stage_config (dict): The configuration dictionary for the stage data.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the merged DataFrame with interactions and environment data, 
            and the merged wrist data array.
    """
    all_stage_data = []
    all_wrist_data = []

    for stage in stages:
        stage_data = load_stage_data(stage, stage_config, data_dir)
        inter_env_df, wrist_data = process_stage_data(stage_data, stage_config)
        # Generate unique interaction IDs across stages and add them to the inter_env DataFrame
        inter_env_df['unique_inter_id'] = _generate_unique_interaction_ids(
            inter_env_df, 
            stage, 
            stage_config['interactions']
        )
        all_stage_data.append(inter_env_df)
        all_wrist_data.append(wrist_data)

    merged_stage_df = pd.concat(all_stage_data, ignore_index=True)
    merged_wrist_data = merge_wrist_data(all_wrist_data)

    return merged_stage_df, merged_wrist_data
    