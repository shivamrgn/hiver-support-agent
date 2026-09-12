"""
Helper script to download the Customer Support on Twitter dataset from Kaggle.

Requires the Kaggle CLI to be installed and authenticated:
    pip install kaggle
    # Set up ~/.kaggle/kaggle.json with your API key

Alternatively, download manually from:
    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Usage:
    python scripts/download_dataset.py
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_SLUG = "thoughtvector/customer-support-on-twitter"


def main():
    target_file = DATA_DIR / "twcs.csv"

    if target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists at {target_file} ({size_mb:.0f} MB)")
        return

    print("Downloading dataset from Kaggle...")
    print("(Requires 'kaggle' CLI with valid API credentials)")
    print()

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", DATASET_SLUG,
                "-p", str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
        if target_file.exists():
            size_mb = target_file.stat().st_size / (1024 * 1024)
            print(f"\n✅ Dataset downloaded to {target_file} ({size_mb:.0f} MB)")
        else:
            print("\n⚠️ Download completed but twcs.csv not found. "
                  "Check if the file has a different name in data/")
    except FileNotFoundError:
        print(
            "❌ 'kaggle' CLI not found. Install it with:\n"
            "    pip install kaggle\n"
            "    # Then set up ~/.kaggle/kaggle.json\n"
            "\n"
            "Or download manually from:\n"
            f"    https://www.kaggle.com/datasets/{DATASET_SLUG}\n"
            f"and place twcs.csv in {DATA_DIR}/"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle download failed: {e}\n"
              "Check your Kaggle API credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()
