import logging
from pathlib import Path
import pandas as pd

from src.utils.const import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_FEATURES_DATASET_FILENAME
from src.data.const import (
    SITUNES_MERGED_MUSIC_METADATA_FILENAME, 
    SITUNES_MERGED_INTER_ENV_DATA_FILENAME, 
    SITUNES_MERGED_WRIST_DATA_FILENAME
)
from src.features import const
from src.data.file_loader import load_csv_data, load_npy_data
from src.features.bio_features import extract_hr_features
from src.features.motion_features import extract_motion_features
from src.features.context_features import extract_context_features
from src.features.music_features import extract_music_features
from src.features.emo_features import extract_emotion_features


def build_features(
    interim_data_dir: Path = INTERIM_DATA_DIR,
    inter_env_filename: str = SITUNES_MERGED_INTER_ENV_DATA_FILENAME,
    music_metadata_filename: str = SITUNES_MERGED_MUSIC_METADATA_FILENAME,
    wrist_data_filename: str = SITUNES_MERGED_WRIST_DATA_FILENAME,
    output_path: Path = PROCESSED_DATA_DIR,
    output_filename: str = PROCESSED_FEATURES_DATASET_FILENAME,
    inter_id_col: str = const.SiTunesInterimInterEnvCols.INTER_ID,
    item_id_col: str = const.SiTunesInterimInterEnvCols.ITEM_ID,
    user_id_col: str = const.SiTunesInterimInterEnvCols.USER_ID,
    target_col: str = const.SiTunesInterimInterEnvCols.RATING
) -> None:
    """
    Build the features dataset by extracting features from the interim datasets and merging them into a single DataFrame, 
    then save it as a CSV file.

    Args:
        interim_data_dir (Path): The directory where the interim datasets are stored.
        inter_env_filename (str): The filename of the interim interaction and environment dataset.
        music_metadata_filename (str): The filename of the interim music metadata dataset.
        wrist_data_filename (str): The filename of the interim wrist data dataset.
        output_path (Path): The directory where the processed features dataset will be saved.
        output_filename (str): The filename for the processed features dataset CSV file.
        inter_id_col (str): The name of the column representing interaction IDs in the datasets.
        item_id_col (str): The name of the column representing item IDs in the datasets.
        user_id_col (str): The name of the column representing user IDs in the datasets.
        target_col (str): The name of the column representing the target variable (e.g., rating) in the datasets.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction and merging...")

    # Load interim datasets
    logger.info("Loading interim datasets...")
    inter_env_df = load_csv_data(interim_data_dir / inter_env_filename)
    music_metadata_df = load_csv_data(interim_data_dir / music_metadata_filename)
    wrist_data = load_npy_data(interim_data_dir / wrist_data_filename)
    logger.info("Interim datasets loaded successfully.")

    # Extract interaction IDs for bio/motion alignment
    inter_ids = inter_env_df[inter_id_col].tolist()

    # Extract features by modality
    logger.info("Extracting biological features from wrist data...")
    bio_df = extract_hr_features(wrist_data=wrist_data, inter_ids=inter_ids)

    logger.info("Extracting motion features from wrist data...")
    motion_df = extract_motion_features(wrist_data=wrist_data, inter_ids=inter_ids)

    logger.info("Extracting context features from interaction and environment data...")
    context_df = extract_context_features(inter_env_df=inter_env_df)

    logger.info("Extracting emotion features from interaction and environment data...")
    emotion_df = extract_emotion_features(inter_env_df=inter_env_df)

    logger.info("Extracting music features from music metadata...")
    music_df = extract_music_features(music_df=music_metadata_df)

    # Merge intreaction level features indexed by inter_id_col (bio, motion, context, emotion)
    logger.info("Merging interaction-level features (bio, motion, context, emotion)...")
    inter_master_df = pd.concat([bio_df, motion_df, context_df, emotion_df], axis=1)

    mapping_df = inter_env_df[[inter_id_col, item_id_col, target_col, user_id_col]].copy().set_index(inter_id_col)

    logger.info("Joining interaction-level features with music features...")
    # Join the interaction-level features with music features using indices
    inter_mapping_df = inter_master_df.join(mapping_df)
    inter_mapping_df.index.name = inter_id_col  # Ensure the index is named for merging

    final_df = (
        inter_mapping_df.reset_index()
        .merge(music_df, left_on=item_id_col, right_index=True, how='left')
        .set_index(inter_id_col)
    )
    logger.info(f"Feature DataFrames merged. Final table shape: {final_df.shape}")

    # Save the final features dataset
    logger.info(f"Saving the final features dataset to '{output_path / output_filename}'...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path / output_filename, index=True)

    logger.info("Feature extraction and merging completed successfully.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    build_features()
