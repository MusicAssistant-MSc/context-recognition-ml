import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def load_json_data(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.
        encoding (str): The encoding of the JSON file.

    Returns:
        Dict[str, Any]: The loaded data as a dictionary.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)


def load_npy_data(file_path: str, allow_pickle_load: bool = False, nmap_mode: Optional[str] = None) -> np.ndarray:
    """Load a NumPy binary file (.npy).

    Args:
        file_path (str): The path to the .npy file.
        allow_pickle_load (bool): Whether to allow pickle loading.
        nmap_mode (Optional[str]): The mode for loading the NumPy array.

    Returns:
        np.ndarray: The loaded NumPy array.
    """
    return np.load(file_path, allow_pickle=allow_pickle_load, mmap_mode=nmap_mode)


def load_csv_data(file_path: str, sep: str = ',', encoding: Optional[str] = None, index_col: Optional[int | str] = None) -> pd.DataFrame:
    """Load a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        sep (str): The delimiter used in the CSV file.
        encoding (Optional[str]): The encoding of the CSV file.
        index_col (Optional[int | str]): The column to use as the row index.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    """
    return pd.read_csv(file_path, sep=sep, encoding=encoding, index_col=index_col)


