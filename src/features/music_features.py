import pandas as pd
from dataclasses import dataclass, field

from src.features import const

@dataclass
class MusicFeatureNamesMapping:
    """Mapping between interim column names and music feature names.
    
    Attributes:
        column_mapping (dict[str, str]): A dictionary mapping interim column names to music feature names.
    """
    column_mapping: dict[str, str] = field(default_factory=lambda: {
        const.SiTunesInterimMusicCols.POPULARITY: const.MusicFeatureNames.POPULARITY,
        const.SiTunesInterimMusicCols.LOUDNESS: const.MusicFeatureNames.LOUDNESS,
        const.SiTunesInterimMusicCols.DANCEABILITY: const.MusicFeatureNames.DANCEABILITY,
        const.SiTunesInterimMusicCols.ENERGY: const.MusicFeatureNames.ENERGY,
        const.SiTunesInterimMusicCols.KEY: const.MusicFeatureNames.KEY,
        const.SiTunesInterimMusicCols.SPEECHINESS: const.MusicFeatureNames.SPEECHINESS,
        const.SiTunesInterimMusicCols.ACOUSTICNESS: const.MusicFeatureNames.ACOUSTICNESS,
        const.SiTunesInterimMusicCols.INSTRUMENTALNESS: const.MusicFeatureNames.INSTRUMENTALNESS,
        const.SiTunesInterimMusicCols.VALENCE: const.MusicFeatureNames.VALENCE,
        const.SiTunesInterimMusicCols.TEMPO: const.MusicFeatureNames.TEMPO,
        const.SiTunesInterimMusicCols.DURATION: const.MusicFeatureNames.DURATION,
        const.SiTunesInterimMusicCols.GENRE_ID: const.MusicFeatureNames.GENRE_ID,
        # Low-level acoustic features
        const.SiTunesInterimMusicCols.F0_MEAN: const.MusicFeatureNames.F0_MEAN,
        const.SiTunesInterimMusicCols.RMS_ENERGY_STD: const.MusicFeatureNames.RMS_ENERGY_STD,
        const.SiTunesInterimMusicCols.ZCR_MEAN: const.MusicFeatureNames.ZCR_MEAN,
        const.SiTunesInterimMusicCols.SHARPNESS_MEAN: const.MusicFeatureNames.PSY_SHARPNESS_MEAN,
    })

    def get_source_columns(self) -> list[str]:
        """Get the list of source column names from the interim data."""
        return list(self.column_mapping.keys())
    

def extract_music_features(
    music_df: pd.DataFrame,
    item_id_col: str = const.SiTunesInterimMusicCols.ITEM_ID,
    feature_names_mapping: MusicFeatureNamesMapping = MusicFeatureNamesMapping()
) -> pd.DataFrame:
    """Extract music features from the interim music DataFrame and rename them.
    
    Args:
        music_df (pd.DataFrame): The input DataFrame containing interim music data.
        item_id_col (str): The name of the column in music_df that contains the item IDs used for indexing.
        feature_names_mapping (MusicFeatureNamesMapping): An instance of MusicFeatureNamesMapping 
            that defines the mapping between interim column names and music feature names.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted music features 
            with standardized column names, indexed by item ID column.
    """
    source_columns = feature_names_mapping.get_source_columns()

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
    music_features_df = music_features_df.rename(columns=feature_names_mapping.column_mapping)
    
    return music_features_df