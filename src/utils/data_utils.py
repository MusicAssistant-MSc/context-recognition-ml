import pandas as pd
from typing import Any

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