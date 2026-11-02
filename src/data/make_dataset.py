from pathlib import Path
import logging
import argparse

from src.data import file_loader, situnes_music_loader, situnes_stage_loader, download_dataset
from src.data import const


def build_interim_data(
    interim_data_dir: Path = const.INTERIM_DATA_DIR,
    output_music_metadata_filename: str = const.SITUNES_MERGED_MUSIC_METADATA_FILENAME,
    output_inter_env_data_filename: str = const.SITUNES_MERGED_INTER_ENV_DATA_FILENAME,
    output_wrist_data_filename: str = const.SITUNES_MERGED_WRIST_DATA_FILENAME
) -> None:
    """Main pipeline function to download and process raw SiTunes data and transform it into interim datasets.
    
    Args:
        interim_data_dir (Path): The directory where interim datasets will be saved.
        output_music_metadata_filename (str): The filename for the merged music metadata CSV.
        output_inter_env_data_filename (str): The filename for the merged interaction and environment data CSV.
        output_wrist_data_filename (str): The filename for the merged wrist data NumPy array.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Raw SiTunes data download and processing orchestrator")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading the raw dataset if it already exists.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of the raw dataset even if it exists.")
    args = parser.parse_args()

    if not args.skip_download:
        logger.info("Downloading raw SiTunes dataset.")
        download_dataset.download_and_extract_github_dataset(
            dataset_zip_url=const.SITUNES_GITHUB_REPO_ZIP_URL,
            relative_source_data_dir=const.SITUNES_GITHUB_REPO_DATA_DIR,
            target_dir=const.SITUNES_DATA_DIR,
            force_download=args.force_download
        )
    else:
        logger.info("Skipping raw SiTunes dataset download.")

    logger.info("Starting SiTunes processing: raw to interim.")

    # Load and merge music metadata
    logger.info("Loading music metadata.")
    music_metadata = situnes_music_loader.load_music_metadata()
    logger.info("Merging music metadata.")
    merged_music_metadata_df = situnes_music_loader.merge_music_metadata(music_metadata)
    logger.info("Music metadata merge completed.")
    
    # Save merged music metadata to interim directory
    merged_music_metadata_path = interim_data_dir / output_music_metadata_filename
    logger.info(f"Saving merged music metadata to '{merged_music_metadata_path}'.")
    file_loader.save_csv_data(df=merged_music_metadata_df, filepath=merged_music_metadata_path)

    # Load and process stage data
    logger.info("Loading and merging stage data.")
    merged_inter_env_data_df, merged_wrist_data_array = situnes_stage_loader.load_merge_stage_data()
    logger.info("Stage data merge completed.")

    # Save merged interaction and environment data to interim directory
    merged_inter_env_data_path = interim_data_dir / output_inter_env_data_filename
    logger.info(f"Saving merged interaction and environment data to '{merged_inter_env_data_path}'.")
    file_loader.save_csv_data(df=merged_inter_env_data_df, filepath=merged_inter_env_data_path)

    # Save merged wrist data to interim directory
    merged_wrist_data_path = interim_data_dir / output_wrist_data_filename
    logger.info(f"Saving merged wrist data to '{merged_wrist_data_path}'.")
    file_loader.save_npy_data(array=merged_wrist_data_array, filepath=merged_wrist_data_path)

    logger.info("SiTunes processing completed: raw to interim.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    build_interim_data()