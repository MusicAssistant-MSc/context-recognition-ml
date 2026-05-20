import pandas as pd
import numpy as np
from typing import Any

from src.data import file_loader
from src.utils import const
from src.utils.data_utils import *

def load_music_metadata(
    music_config: dict = const.SITUNES_MUSIC_METADATA_CONFIG,
    filepath_key: str = const.SitunesConfigKeys.FILEPATH
) -> dict[str, pd.DataFrame]:
    """Load the music metadata from the SiTunes dataset.

    Args:
        music_config (dict): The configuration dictionary for the music metadata files.
        filepath_key (str): The key in the configuration dictionary that contains the file path.

    Returns:
        dict[str, pd.DataFrame]: Mapping of music metadata CSV filenames to loaded data.
    """
    music_metadata_files = {}

    for name, file_cfg in music_config.items():
        filepath = file_cfg[filepath_key]
        if not filepath.is_file():
            raise FileNotFoundError(f"Music metadata file '{name}' not found: {filepath}")
        music_metadata_files[name] = file_loader.load_csv_data(filepath)
    
    return music_metadata_files


def _clean_music_metadata_df(
    df: pd.DataFrame, 
    drop_cols: list[str],
    drop_unnamed_cols: bool = False
) -> pd.DataFrame:
    """Clean a music metadata DataFrame based on the provided configuration.

    Args:
        df (pd.DataFrame): The music metadata DataFrame to clean.
        drop_cols (list[str]): The list of columns to drop.
        drop_unnamed_cols (bool): Whether to drop unnamed columns.

    Returns:
        pd.DataFrame: The cleaned music metadata DataFrame.
    """
    # drop_cols = config.get('drop_cols', [])
    # drop_unnamed_cols = config.get('drop_unnamed_cols', False)

    df = df.copy()
    df = drop_df_columns(df=df, columns_to_drop=drop_cols, drop_unnamed=drop_unnamed_cols)
    df = round_numeric_columns(df, precision=const.SITUNES_FLOAT_ROUNDING_PRECISION)

    return df


def _consolidate_music_metadata_id_cols(df: pd.DataFrame, primary_id_col: str, secondary_id_col: str) -> pd.DataFrame:
    """Consolidate two ID columns in a music metadata DataFrame into a single primary ID column.

    Args:
        df (pd.DataFrame): The music metadata DataFrame containing the ID columns.
        primary_id_col (str): The name of the primary ID column to keep.
        secondary_id_col (str): The name of the secondary ID column to consolidate.

    Returns:
        pd.DataFrame: The DataFrame with consolidated ID columns.
    """
    if not primary_id_col or not secondary_id_col or primary_id_col == secondary_id_col:
        return df
    if primary_id_col not in df.columns or secondary_id_col not in df.columns:
        return df
    
    overlap_mask = df[primary_id_col].notna() & df[secondary_id_col].notna()
    overlap_df = df[overlap_mask]

    if not overlap_df.empty:
        ids_consistent = verify_series_identity(
            overlap_df[primary_id_col].astype(float), 
            overlap_df[secondary_id_col].astype(float)
        )
        if not ids_consistent:
            raise ValueError(f"ID columns '{primary_id_col}' and '{secondary_id_col}'" 
                             "have conflicting values in overlapping rows.")
        
    return consolidate_column_pair(df, primary_id_col, secondary_id_col)


def merge_music_metadata(
    music_dfs: dict[str, pd.DataFrame], 
    configs: dict[str, dict] = const.SITUNES_MUSIC_METADATA_CONFIG, 
    drop_cols_key: str = const.SitunesConfigKeys.DROP_COLS,
    drop_unnamed_cols_key: str = const.SitunesConfigKeys.DROP_UNNAMED_COLS,
    pkey_candidate_cols_key: str = const.SitunesConfigKeys.PKEY_CANDIDATE_COLS,
    redundant_col_suffix: str = const.SITUNES_REDUNDANT_COL_SUFFIX,
    join_type: str = const.SITUNES_MUSIC_METADATA_MERGE_TYPE
) -> pd.DataFrame:
    """Merge the music metadata DataFrames on shared columns.

    Args:
        music_dfs (dict[str, pd.DataFrame]): A dictionary of music metadata DataFrames.
        configs (dict[str, dict]): A dictionary of configurations for each music metadata DataFrame.
        drop_cols_key (str): The key in the config dict that specifies columns to drop.
        drop_unnamed_cols_key (str): The key in the config dict that specifies whether to drop unnamed columns.
        pkey_candidate_cols_key (str): The key in the config dict that specifies primary key candidate columns.
        redundant_col_suffix (str): The suffix to use for redundant columns created by the merge.
        join_type (str): The type of join to perform (e.g., 'left', 'right', 'outer', 'inner').

    Returns:
        pd.DataFrame: The merged music metadata DataFrame.
    """
    merged_df = None
    main_id_col = None

    for name, df in music_dfs.items():
        # Get config for the current df by its name, or use an empty config if not found
        config = configs.get(name, {})
        drop_cols = config.get(drop_cols_key, [])
        drop_unnamed_cols = config.get(drop_unnamed_cols_key, False)
        # Clean the df based on the config before merging
        current_df = _clean_music_metadata_df(df, drop_cols, drop_unnamed_cols)

        # Get the first primary key candidate column
        curr_pkey_candidate = config.get(pkey_candidate_cols_key, [None])[0]

        if merged_df is None:
            merged_df = current_df
            main_id_col = curr_pkey_candidate
            continue

        fingerprint_cols = get_shared_columns(merged_df, current_df)
        merged_fprint_cols_uniqueness = check_column_uniqueness(merged_df, fingerprint_cols)
        cleaned_fprint_cols_uniqueness = check_column_uniqueness(current_df, fingerprint_cols)

        # Merge the current df with the accumulated merged_df using the shared columns as keys if they are unique in both dfs, 
        # otherwise perform a cartesian product merge
        merged_df = pd.merge(
            left=merged_df, 
            right=current_df, 
            on=fingerprint_cols 
                if merged_fprint_cols_uniqueness['is_unique'] and cleaned_fprint_cols_uniqueness['is_unique'] 
                else None,
            how=join_type, 
            suffixes=('', redundant_col_suffix)
        )

        # Consolidate the primary key candidate columns if they exist in both dfs
        merged_df = _consolidate_music_metadata_id_cols(
            df=merged_df, 
            primary_id_col=main_id_col, 
            secondary_id_col=curr_pkey_candidate
        )

        # Drop any redundant columns created by the merge
        merged_df = consolidate_redundant_columns(df=merged_df, redundant_col_suffix=redundant_col_suffix)
    
    return merged_df