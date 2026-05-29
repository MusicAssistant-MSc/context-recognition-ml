import pandas as pd
from dataclasses import dataclass

from src.features import const
from src.features.feature_metadata import FeatureMetadata, FeatureGroup
from src.features.feature_registry import FEATURE_REGISTRY

@dataclass
class MusicFeatureMapping:
    music_features: list[FeatureMetadata]

    @classmethod
    def from_registry(cls) -> "MusicFeatureMapping":
        """Factory method to create MusicFeatureMapping using global registry of features."""
        return cls(music_features=FEATURE_REGISTRY.get_by_group(FeatureGroup.MUSIC))
    
    def get_source_columns(self) -> list[str]:
        """Get the list of source column names from the interim data."""
        return [feature.interim_name for feature in self.music_features]
    
    def get_feature_names_mapping(self) -> dict[str, str]:
        """Get the dict for interim to feature names mapping."""
        return {
            feature.interim_name: feature.name
            for feature in self.music_features
        }
    

def extract_music_features(
    music_df: pd.DataFrame,
    item_id_col: str = const.SiTunesInterimMusicCols.ITEM_ID,
    feature_mapping: MusicFeatureMapping = MusicFeatureMapping.from_registry()
) -> pd.DataFrame:
    """Extract music features from the interim music DataFrame and rename them.
    
    Args:
        music_df (pd.DataFrame): The input DataFrame containing interim music data.
        item_id_col (str): The name of the column in music_df that contains the item IDs used for indexing.
        feature_mapping (MusicFeatureMapping): An instance of MusicFeatureMapping with a list of FeatureMetadata
            objects that allows the mapping between interim column names and music feature names.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted music features 
            with standardized column names, indexed by item ID column.
    """
    source_columns = feature_mapping.get_source_columns()

    if not all(col in music_df.columns for col in source_columns):
        missing_cols = [col for col in source_columns if col not in music_df.columns]
        raise ValueError(f"Some required columns are missing from the input interim music DataFrame: {missing_cols}")
    if not item_id_col in music_df.columns:
        raise ValueError(f"The specified item ID column '{item_id_col}' is not present in the input interim music DataFrame.")
    
    # Add the item (music) ID column to the source columns if it's not already included
    source_columns = [item_id_col] + source_columns if item_id_col not in source_columns else source_columns
    # Select only the relevant columns, add index based on the item ID and rename them according to the mapping
    music_features_df = music_df[source_columns].copy()
    music_features_df.set_index(item_id_col, drop=True, inplace=True)
    music_features_df = music_features_df.rename(columns=feature_mapping.get_feature_names_mapping())
    
    return music_features_df