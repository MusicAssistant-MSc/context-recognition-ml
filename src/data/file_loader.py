import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any

def load_json_data(file_path: str | Path, encoding: str = 'utf-8') -> dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    Args:
        file_path (str | Path): The path to the JSON file.
        encoding (str): The encoding of the JSON file.

    Returns:
        dict[str, Any]: The loaded data as a dictionary.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with path.open('r', encoding=encoding) as f:
        return json.load(f)


def load_npy_data(file_path: str | Path, allow_pickle_load: bool = False, nmap_mode: str | None = None) -> np.ndarray:
    """Load a NumPy binary file (.npy).

    Args:
        file_path (str | Path): The path to the .npy file.
        allow_pickle_load (bool): Whether to allow pickle loading.
        nmap_mode (Optional[str]): The mode for loading the NumPy array.

    Returns:
        np.ndarray: The loaded NumPy array.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"NumPy file not found: {file_path}")

    return np.load(str(path), allow_pickle=allow_pickle_load, mmap_mode=nmap_mode)


def load_csv_data(
    file_path: str | Path, 
    sep: str = ',', 
    encoding: str | None = None, 
    index_col: str | int | None = None
) -> pd.DataFrame:
    """Load a CSV file into a Pandas DataFrame.

    Args:
        file_path (str | Path): The path to the CSV file.
        sep (str): The delimiter used in the CSV file.
        encoding (str | None): The encoding of the CSV file.
        index_col (str | int | None): The column to use as the row index.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(path, sep=sep, encoding=encoding, index_col=index_col)


def save_csv_data(
    df: pd.DataFrame, 
    filepath: str | Path, 
    sep: str = ',', 
    encoding: str | None = None, 
    index: bool = False
) -> None:
    """Save a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str | Path): The path where the CSV file will be saved.
        sep (str): The delimiter to use in the CSV file.
        encoding (str | None): The encoding of the CSV file.
        index (bool): Whether to write row names (index).

    Returns:
        None
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, encoding=encoding, index=index)    


def save_npy_data(
    array: np.ndarray, 
    filepath: str | Path, 
    allow_pickle_save: bool = False
) -> None:
    """Save a NumPy array to a binary file (.npy).

    Args:
        array (np.ndarray): The NumPy array to save.
        filepath (str | Path): The path where the .npy file will be saved.
        allow_pickle_save (bool): Whether to allow pickle saving.
        nmap_mode (Optional[str]): The mode for saving the NumPy array.

    Returns:
        None
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array, allow_pickle=allow_pickle_save)