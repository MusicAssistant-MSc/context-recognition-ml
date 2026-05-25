import pandas as pd
from dataclasses import dataclass, field

from src.features import const

@dataclass
class ContextFeatureNamesMapping:
    """Mapping between interim column names and context feature names.
    
    Attributes:
        column_mapping (dict[str, str]): A dictionary mapping interim column names to context feature names.
    """
    column_mapping: dict[str, str] = field(default_factory=lambda: {
        const.SiTunesInterimInterEnvCols.TIME_PERIOD: const.ContextFeatureNames.TIME_PERIOD,
        const.SiTunesInterimInterEnvCols.WEATHER_TYPE: const.ContextFeatureNames.WEATHER_TYPE,
        const.SiTunesInterimInterEnvCols.WEATHER_PRESSURE: const.ContextFeatureNames.WEATHER_PRESSURE,
        const.SiTunesInterimInterEnvCols.WEATHER_TEMP: const.ContextFeatureNames.WEATHER_TEMP,
        const.SiTunesInterimInterEnvCols.WEATHER_HUMIDITY: const.ContextFeatureNames.WEATHER_HUMIDITY,
        const.SiTunesInterimInterEnvCols.GPS_SPEED: const.ContextFeatureNames.GPS_SPEED
    })

    def get_source_columns(self) -> list[str]:
        """Get the list of source column names from the interim data."""
        return list(self.column_mapping.keys())
    

def extract_context_features(
    inter_env_df: pd.DataFrame,
    inter_id_col: str = const.SiTunesInterimInterEnvCols.INTER_ID,
    feature_names_mapping: ContextFeatureNamesMapping = ContextFeatureNamesMapping()
) -> pd.DataFrame:
    """Extract context features from the interim interaction and environment DataFrame and rename them.
    
    Args:
        inter_env_df (pd.DataFrame): The input DataFrame containing interim interaction and environment data.
        inter_id_col (str): The name of the column in inter_env_df that contains the interaction IDs.
        feature_names_mapping (ContextFeatureNamesMapping): An instance of ContextFeatureNamesMapping 
            that defines the mapping between interim column names and context feature names.

    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted context features with standardized column names.
    """
    source_columns = feature_names_mapping.get_source_columns()
    
    if not all(col in inter_env_df.columns for col in source_columns):
        missing_cols = [col for col in source_columns if col not in inter_env_df.columns]
        raise ValueError(f"Some required columns are missing from the input interim DataFrame: {missing_cols}")
    if not inter_id_col in inter_env_df.columns:
        raise ValueError(f"The specified interaction ID column '{inter_id_col}' is not present in the input interim DataFrame.")
    
    # Add the interaction ID column to the source columns if it's not already included
    source_columns = [inter_id_col] + source_columns if inter_id_col not in source_columns else source_columns
    # Select only the relevant columns and rename them according to the mapping
    context_features_df = inter_env_df[source_columns].rename(columns=feature_names_mapping.column_mapping)
    
    return context_features_df
