"""
Data Download Script for Telco Customer Churn Dataset.

This module handles downloading the Telco Customer Churn dataset from IBM's
GitHub repository or alternative sources.
"""

import logging
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL with progress bar.

    Args:
        url: URL of the file to download
        destination: Path where the file should be saved
        chunk_size: Size of chunks to download at a time (in bytes)

    Returns:
        True if download successful, False otherwise

    Raises:
        requests.RequestException: If download fails
    """
    try:
        logger.info(f"Downloading data from {url}")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'wb') as file, \
             tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=destination.name) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        logger.info(f"Successfully downloaded data to {destination}")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download data: {e}")
        return False


def verify_data_file(file_path: Path, min_size_kb: int = 10) -> bool:
    """
    Verify that the downloaded data file exists and has content.

    Args:
        file_path: Path to the data file
        min_size_kb: Minimum expected file size in KB

    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        logger.error(f"Data file does not exist: {file_path}")
        return False

    file_size_kb = file_path.stat().st_size / 1024
    if file_size_kb < min_size_kb:
        logger.error(f"Data file is too small ({file_size_kb:.2f} KB). Expected at least {min_size_kb} KB")
        return False

    logger.info(f"Data file verified: {file_path} ({file_size_kb:.2f} KB)")
    return True


def download_telco_churn_data(
    url: Optional[str] = None,
    destination: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """
    Download the Telco Customer Churn dataset.

    Args:
        url: URL to download from (defaults to config.DATA_URL)
        destination: Where to save the file (defaults to config.RAW_DATA_FILE)
        force_download: If True, download even if file already exists

    Returns:
        Path to the downloaded data file

    Raises:
        RuntimeError: If download fails or data is invalid
    """
    url = url or config.DATA_URL
    destination = destination or config.RAW_DATA_FILE

    # Check if file already exists
    if destination.exists() and not force_download:
        logger.info(f"Data file already exists: {destination}")
        if verify_data_file(destination):
            logger.info("Using existing data file")
            return destination
        else:
            logger.warning("Existing file is invalid, re-downloading...")

    # Download the data
    success = download_file(url, destination)

    if not success:
        raise RuntimeError(f"Failed to download data from {url}")

    # Verify the downloaded file
    if not verify_data_file(destination):
        raise RuntimeError("Downloaded file is invalid")

    return destination


def main():
    """Main function to download the dataset."""
    try:
        logger.info("="*60)
        logger.info("Starting Telco Customer Churn Data Download")
        logger.info("="*60)

        data_path = download_telco_churn_data()

        logger.info("="*60)
        logger.info(f"✓ Data download complete: {data_path}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"✗ Data download failed: {e}")
        raise


if __name__ == "__main__":
    main()
