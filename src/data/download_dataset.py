from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
import logging

logger = logging.getLogger(__name__)

def download_and_extract_github_dataset(
    dataset_zip_url: str,
    relative_source_data_dir: str,
    target_dir: Path,
    force_download: bool = False
) -> None:
    """
    Download and extract a dataset from a GitHub URL to a target directory.

    Args:
        dataset_zip_url (str): The URL of the dataset zip file to download.
        relative_source_data_dir (str): The relative path to the source data directory inside the dataset directory.
        target_dir (Path): The target directory to extract the dataset to.
        force_download (bool, optional): Whether to force a re-download of the dataset. Defaults to False.

    Returns:
        None
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


    # Check if the dataset already exists and force_download is False
    if (
        not force_download 
        and target_dir.exists() 
        and any(target_dir.iterdir())
    ):
        logger.info(f"Dataset already exists at '{target_dir}'. Skipping download.")
        return

    # Create a temporary file to download the zip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "dataset.zip"
        extract_dir = Path(tmp_dir) / "extracted"
        
        logger.info(f"Downloading raw data archive from '{dataset_zip_url}'...")
        urllib.request.urlretrieve(dataset_zip_url, tmp_file)
        logger.info(f"Dataset downloaded to temporary file '{tmp_file.name}'.")

        # Extract the zip file
        logger.info(f"Extracting dataset...")
        with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the extracted source data directory
        dataset_root_dir = next(extract_dir.iterdir())
        source_data_dir = dataset_root_dir / relative_source_data_dir

        if not source_data_dir.exists():
            logger.error(f"Source data directory '{source_data_dir}' not found in the extracted dataset.")
            raise FileNotFoundError(f"Source data directory '{source_data_dir}' not found in the extracted dataset.")

        logger.info(f"Moving extracted source data directory '{source_data_dir}' to target directory '{target_dir}'...")
        for item in source_data_dir.iterdir():
            destination = target_dir / item.name
            if item.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

    logger.info(f"Dataset successfully downloaded and extracted to '{target_dir}'.")
