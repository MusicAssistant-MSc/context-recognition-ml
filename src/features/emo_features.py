import pandas as pd
from dataclasses import dataclass, field

from src.features import const

@dataclass
class EmotionFeatureNamesMapping:
    """Mapping between interim column names and emotion feature names.
    
    Attributes:
        column_mapping (dict[str, str]): A dictionary mapping interim column names to emotion feature names.
    """
    column_mapping: dict[str, str] = field(default_factory=lambda: {
        const.SiTunesInterimInterEnvCols.EMO_PRE_VALENCE: const.EmotionFeatureNames.EMO_PRE_VALENCE,
        const.SiTunesInterimInterEnvCols.EMO_PRE_AROUSAL: const.EmotionFeatureNames.EMO_PRE_AROUSAL
    })

    def get_source_columns(self) -> list[str]:
        """Get the list of source column names from the interim data."""
        return list(self.column_mapping.keys())
    

def extract_emotion_features(
    inter_env_df: pd.DataFrame,
    inter_id_col: str = const.SiTunesInterimInterEnvCols.INTER_ID,
    feature_names_mapping: EmotionFeatureNamesMapping = EmotionFeatureNamesMapping()
) -> pd.DataFrame:
    """Extract emotion features from the interim interaction and environment DataFrame and rename them.
    
    Args:
        inter_env_df (pd.DataFrame): The input DataFrame containing interim interaction and environment data.
        inter_id_col (str): The name of the column in inter_env_df that contains the interaction IDs used for indexing.
        feature_names_mapping (EmotionFeatureNamesMapping): An instance of EmotionFeatureNamesMapping 
            that defines the mapping between interim column names and emotion feature names.

    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted emotion features with standardized column names.
    """
    source_columns = feature_names_mapping.get_source_columns()

    if not all(col in inter_env_df.columns for col in source_columns):
        missing_cols = [col for col in source_columns if col not in inter_env_df.columns]
        raise ValueError(f"Some required columns are missing from the input interim DataFrame: {missing_cols}")
    if not inter_id_col in inter_env_df.columns:
        raise ValueError(f"The specified interaction ID column '{inter_id_col}' is not present in the input interim DataFrame.")
    
    # Add the interaction ID column to the source columns if it's not already included
    source_columns = [inter_id_col] + source_columns if inter_id_col not in source_columns else source_columns
    # Select only the relevant columns, add index based on the interaction ID and rename them according to the mapping
    emotion_features_df = inter_env_df[source_columns].copy()
    emotion_features_df.set_index(inter_id_col, drop=True, inplace=True)
    emotion_features_df = emotion_features_df.rename(columns=feature_names_mapping.column_mapping)

    return emotion_features_df
