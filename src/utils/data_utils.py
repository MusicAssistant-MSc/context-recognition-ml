import pandas as pd
from typing import Any

from src.utils import const

def check_column_uniqueness(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Check if a specific column or set of columns can serve as a unique key.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        columns (list[str]): A list of column names to check for uniqueness.

    Returns:
        dict[str, Any]: A dictionary containing:
            - 'is_unique' (bool): True if the columns uniquely identify rows.
            - 'unique_count' (int): Number of unique combinations.
            - 'total_count' (int): Total number of rows in the DataFrame.
            - 'duplicate_count' (int): Number of redundant rows.
    """
    total_count = len(df)
    unique_combinations = df.drop_duplicates(subset=columns)
    unique_count = len(unique_combinations)
    duplicate_count = total_count - unique_count
    is_unique = total_count == unique_count

    return {
        'is_unique': is_unique,
        'unique_count': unique_count,
        'total_count': total_count,
        'duplicate_count': duplicate_count
    }


def get_shared_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
    """Identify shared columns between two DataFrames.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        List[str]: A list of column names that are present in both DataFrames.
    """
    return list(set(df1.columns).intersection(set(df2.columns)))


def drop_df_columns(
    df: pd.DataFrame, 
    columns_to_drop: list[str], 
    drop_unnamed: bool = False, 
    unnamed_cols_prefix: str = const.PANDAS_DF_UNNAMED_COL_PREFIX
) -> pd.DataFrame:
    """Drop specific columns from df and unnamed ones.

    Args:
        df (pd.DataFrame): The DataFrame from which to drop unnamed columns.
        columns_to_drop (list[str]): A list of column names to drop.
        drop_unnamed (bool): Whether to drop unnamed columns.
        unnamed_cols_prefix (str): The prefix of unnamed columns.

    Returns:
        pd.DataFrame: A new DataFrame with unnamed columns removed.
    """
    unnamed_cols = [col for col in df.columns if col.startswith(unnamed_cols_prefix) or col == ""]
    columns_to_drop = list(set(columns_to_drop + unnamed_cols)) if drop_unnamed else columns_to_drop
    return df.drop(columns=columns_to_drop)


def verify_series_identity(series1: pd.Series, series2: pd.Series) -> bool:
    """Verify if two pandas Series are identical in terms of index and values.

    Args:
        series1 (pd.Series): The first Series to compare.
        series2 (pd.Series): The second Series to compare.

    Returns:
        bool: True if the Series are identical, False otherwise.
    """
    if len(series1) != len(series2):
        return False
    return series1.equals(series2)


def consolidate_column_pair(
    df: pd.DataFrame, 
    original_col: str, 
    redundant_col: str
) -> pd.DataFrame:
    """Consolidate two columns by filling missing values in target from source.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        original_col (str): The column to be updated with missing values from the redundant column.
        redundant_col (str): The redundant column providing values to original_col and to be dropped.

    Returns:
        pd.DataFrame: DataFrame with consolidated columns.
    """
    if original_col not in df.columns or redundant_col not in df.columns:
        return df

    if original_col == redundant_col:
        return df

    df = df.copy()
    # Fill nulls in original_col with values from redundant_col
    df[original_col] = df[original_col].fillna(df[redundant_col])

    return df.drop(columns=[redundant_col])


def consolidate_redundant_columns(df: pd.DataFrame, redundant_col_suffix: str) -> pd.DataFrame:
    """Find and consolidate all columns ending with a specific suffix.

    Args:
        df (pd.DataFrame): The DataFrame after a merge operation.
        redundant_col_suffix (str): The suffix identifying redundant columns (e.g., '_redundant').

    Returns:
        pd.DataFrame: DataFrame with all redundant columns consolidated.
    """
    redundant_cols = [col for col in df.columns if col.endswith(redundant_col_suffix)]
    current_df = df.copy()

    for redundant_col in redundant_cols:
        original_col = redundant_col.replace(redundant_col_suffix, "")
        current_df = consolidate_column_pair(current_df, original_col, redundant_col)
        
    return current_df


def round_numeric_columns(df: pd.DataFrame, precision: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Round numeric columns in a DataFrame to a specified number of decimal places.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        precision (int): The number of decimal places to round to.
        columns (list[str] | None): A list of column names to round. If None, all numeric columns will be rounded.

    Returns:
        pd.DataFrame: A new DataFrame with rounded numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number', 'float32', 'float64']).columns.tolist()
    
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(precision)
    
    return df


def rename_columns(df: pd.DataFrame, column_mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns in a DataFrame based on a provided mapping.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_mapping (dict[str, str]): A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: A new DataFrame with renamed columns.
    """
    return df.rename(columns=column_mapping)