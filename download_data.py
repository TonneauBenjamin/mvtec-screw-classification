#!/usr/bin/env python3
"""
Download the MVTec Screw dataset from Google Drive.

This script uses gdown to download and extract the dataset to the
specified directory.

Usage:
    python download_data.py --output-dir ./data
"""

import argparse
import subprocess
import sys
from pathlib import Path
import zipfile
 

def _run_gdown_download(url: str, destination: Path, retries: int = 3) -> None:
    """Download a file with gdown CLI using resume mode and basic retries."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'gdown',
                    '--continue',
                    url,
                    '-O',
                    str(destination),
                ],
                check=True,
            )
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < retries:
                print(f"Download attempt {attempt}/{retries} failed. Retrying...")

    raise RuntimeError(f"gdown failed after {retries} attempts") from last_error


def download_dataset(output_dir: str = './data') -> None:
    """
    Download and extract the MVTec Screw dataset.
    
    Args:
        output_dir: Directory where the dataset will be extracted
    """
    # Google Drive file ID for MVTec Screw dataset
    DRIVE_FILE_ID = '1z-57McRCQ5PT6UYbF640BafBJmP3L_Jj'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_path / 'mvtec_screw.zip'
    extract_path = output_path / 'mvtec_screw'
    
    # Check if already downloaded and extracted
    if extract_path.exists():
        print(f"[OK] Dataset already exists at: {extract_path.absolute()}")
        if (extract_path / 'train' / 'good').exists():
            good_count = len(list((extract_path / 'train' / 'good').glob('*.png')))
            print(f"  Train: {good_count} good samples")
        if (extract_path / 'train' / 'not-good').exists():
            not_good_count = len(list((extract_path / 'train' / 'not-good').glob('*.png')))
            print(f"  Train: {not_good_count} not-good samples")
        if extract_path.joinpath('test').exists():
            test_count = len(list(extract_path.joinpath('test').glob('*.png')))
            print(f"  Test: {test_count} test samples")
        return
    
    # Download from Google Drive
    print(f"Downloading MVTec Screw dataset from Google Drive...")
    print(f"File ID: {DRIVE_FILE_ID}")
    print(f"Destination: {output_path.absolute()}")
    
    try:
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        _run_gdown_download(url, zip_path)
        print(f"\n[OK] Download completed: {zip_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nQuick checks:")
        print("  - Ensure internet access and no proxy/firewall blocks to drive.google.com")
        print("  - Verify gdown is installed: python -m pip install gdown")
        print("  - Retry with: python download_data.py --output-dir ./data")
        print(f"\nManual download alternative:")
        print(f"  1. Visit: https://drive.google.com/file/d/{DRIVE_FILE_ID}/view")
        print(f"  2. Download the file manually")
        print(f"  3. Extract the ZIP file to: {output_path.absolute()}")
        sys.exit(1)
    
    # Extract the zip file
    print(f"\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"[OK] Extraction completed")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        sys.exit(1)

    # Normalize extracted folder name when ZIP root is not mvtec_screw.
    archive_extract_path = output_path / 'archive'
    if not extract_path.exists() and archive_extract_path.exists():
        archive_extract_path.rename(extract_path)
        print(f"[OK] Renamed extracted folder: {archive_extract_path.name} -> {extract_path.name}")
    
    # Verify extracted structure
    print(f"\n[OK] Dataset structure verified:")
    if (extract_path / 'train' / 'good').exists():
        good_count = len(list((extract_path / 'train' / 'good').glob('*.png')))
        print(f"  Train/Good: {good_count} images")
    
    if (extract_path / 'train' / 'not-good').exists():
        not_good_count = len(list((extract_path / 'train' / 'not-good').glob('*.png')))
        print(f"  Train/Not-Good: {not_good_count} images")
    
    if extract_path.joinpath('test').exists():
        test_count = len(list(extract_path.joinpath('test').glob('*.png')))
        print(f"  Test: {test_count} images")
    
    # Clean up zip file
    zip_path.unlink()
    print(f"\n[OK] Dataset ready for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the MVTec Screw dataset from Google Drive'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for extracting the dataset (default: ./data)'
    )
    parser.add_argument(
        '--dataset-only',
        action='store_true',
        help='Download only the dataset (default behavior)'
    )
    
    args = parser.parse_args()

    # 1. Download and extract dataset images
    download_dataset(args.output_dir)

    print("\n[OK] Setup complete. Run the notebook to train the model and generate best_model.pth.")