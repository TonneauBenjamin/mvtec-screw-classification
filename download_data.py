"""
MVTec Screw Dataset Downloader
================================
This script downloads the MVTec screw dataset from Google Drive and extracts it.
The dataset is used for binary classification (Good vs Defective).

Usage:
    python download_data.py
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

# Check for required dependencies
try:
    import gdown
except ImportError:
    print("Error: gdown module not found.")
    print("\nPlease install the required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def download_dataset(output_dir: str = "./data", verbose: bool = True) -> str:
    """
    Download the MVTec screw dataset from Google Drive.
    
    Args:
        output_dir: Directory to save the dataset
        verbose: Whether to print progress information
        
    Returns:
        Path to the extracted dataset
    """
    # Google Drive file ID for the MVTec screw dataset
    file_id = "1z-57McRCQ5PT6UYbF640BafBJmP3L_Jj"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download file
    zip_path = output_path / "mvtec_screw.zip"
    extract_path = output_path / "mvtec_screw"
    
    if verbose:
        print(f"Downloading MVTec screw dataset...")
        print(f"   Output directory: {output_path.absolute()}")
    
    try:
        gdown.download(url, str(zip_path), quiet=not verbose)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Download failed: {zip_path} not found")
        
        if verbose:
            print(f"Download completed!")
            print(f"Extracting dataset...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        # Handle the archive folder structure
        archive_path = output_path / "archive"
        if archive_path.exists():
            if extract_path.exists():
                shutil.rmtree(extract_path)
            shutil.move(str(archive_path), str(extract_path))
        
        if verbose:
            print(f"Dataset extracted successfully!")
            print(f"Dataset location: {extract_path}")
        
        return str(extract_path)
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


def verify_dataset(dataset_path: str) -> bool:
    """
    Verify that the dataset has the expected structure.
    
    Args:
        dataset_path: Path to the extracted dataset
        
    Returns:
        True if dataset structure is valid, False otherwise
    """
    required_dirs = ["train", "test"]
    required_subdirs = ["good", "defective"]
    
    dataset_root = Path(dataset_path)
    
    # Check main directories
    for main_dir in required_dirs:
        dir_path = dataset_root / main_dir
        if not dir_path.exists():
            print(f"Missing directory: {main_dir}")
            return False
        
        # Check subdirectories
        for subdir in required_subdirs:
            subdir_path = dir_path / subdir
            if not subdir_path.exists():
                print(f"Missing directory: {main_dir}/{subdir}")
                return False
            
            # Count files
            files = list(subdir_path.glob("*.png"))
            if len(files) == 0:
                print(f"No images found in {main_dir}/{subdir}")
            else:
                print(f"Found {len(files)} images in {main_dir}/{subdir}")
    
    return True


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download the MVTec screw dataset from Google Drive"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save the dataset (default: ./data)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset structure after download"
    )
    
    args = parser.parse_args()
    
    try:
        # Download dataset
        dataset_path = download_dataset(args.output_dir, verbose=True)
        
        # Verify dataset if requested
        if args.verify:
            print("\nVerifying dataset structure...")
            if verify_dataset(dataset_path):
                print("\nDataset verification passed!")
            else:
                print("\nDataset verification failed!")
                return 1
        
        print("\nDownload process completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nDownload process failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
