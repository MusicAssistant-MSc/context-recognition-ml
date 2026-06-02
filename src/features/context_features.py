import pandas as pd
from dataclasses import dataclass

from src.features import const
from src.features.feature_metadata import FeatureMetadata, FeatureGroup
from src.features.feature_registry import FEATURE_REGISTRY

@dataclass
class ContextFeatureMapping:
    context_features: list[FeatureMetadata]

    @classmethod
    def from_registry(cls) -> "ContextFeatureMapping":
        """Factory method to create ContextFeatureMapping using global registry of features."""
        return cls(context_features=FEATURE_REGISTRY.get_by_group(FeatureGroup.CONTEXT))
    
    def get_source_columns(self) -> list[str]:
        """Get the list of source column names from the interim data."""
        return [feature.interim_name for feature in self.context_features]
    
    def get_feature_names_mapping(self) -> dict[str, str]:
        """Get the dict for interim to feature names mapping."""
        return {
            feature.interim_name: feature.name
            for feature in self.context_features
        }
    

def extract_context_features(
    inter_env_df: pd.DataFrame,
    inter_id_col: str = const.SiTunesInterimInterEnvCols.INTER_ID,
    feature_mapping: ContextFeatureMapping = ContextFeatureMapping.from_registry()
) -> pd.DataFrame:
    """Extract context features from the interim interaction and environment DataFrame and rename them.
    
    Args:
        inter_env_df (pd.DataFrame): The input DataFrame containing interim interaction and environment data.
        inter_id_col (str): The name of the column in inter_env_df that contains the interaction IDs used for indexing.
        feature_mapping (ContextFeatureMapping): An instance of ContextFeatureMapping with a list of FeatureMetadata
            objects that allows to get the mapping between interim column names and context feature names.

    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted context features with standardized column names.
    """
    source_columns = feature_mapping.get_source_columns()
    
    if not all(col in inter_env_df.columns for col in source_columns):
        missing_cols = [col for col in source_columns if col not in inter_env_df.columns]
        raise ValueError(f"Some required columns are missing from the input interim DataFrame: {missing_cols}")
    if not inter_id_col in inter_env_df.columns:
        raise ValueError(f"The specified interaction ID column '{inter_id_col}' is not present in the input interim DataFrame.")
    
    # Add the interaction ID column to the source columns if it's not already included
    source_columns = [inter_id_col] + source_columns if inter_id_col not in source_columns else source_columns
    # Select only the relevant columns, add index based on the interaction ID and rename them according to the mapping
    context_features_df = inter_env_df[source_columns].copy()
    context_features_df.set_index(inter_id_col, drop=True, inplace=True)
    context_features_df = context_features_df.rename(columns=feature_mapping.get_feature_names_mapping())
    
    return context_features_df
