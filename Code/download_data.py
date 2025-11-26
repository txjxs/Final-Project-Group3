import os
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = "./coco_data"
URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip"
}

def download_file(url, dest_path):
    """
    Downloads a file from a URL with a progress bar.
    """
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}, skipping download.")
        return

    print(f" Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    print(f"Download complete: {dest_path}")


def unzip_file(zip_path, extract_to):
    """
    Unzips a file to the specified directory.
    """
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: The file {zip_path} is corrupted. Please delete it and try downloading again.")


def main():
    # 1. Create Data Directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # 2. Download and Extract Validation Set
    val_zip_path = os.path.join(DATA_DIR, "val2017.zip")
    download_file(URLS["val2017"], val_zip_path)

    # Check if unzipped folder already exists to avoid re-unzipping
    if not os.path.exists(os.path.join(DATA_DIR, "val2017")):
        unzip_file(val_zip_path, DATA_DIR)

        os.remove(val_zip_path)
    else:
        print(f"Validation data already extracted.")

    # 3. Download and Extract Train Set (Huge, ~18GB)

    train_zip_path = os.path.join(DATA_DIR, "train2017.zip")
    download_file(URLS["train2017"], train_zip_path)

    if not os.path.exists(os.path.join(DATA_DIR, "train2017")):
        unzip_file(train_zip_path, DATA_DIR)

        os.remove(train_zip_path)
    else:
        print(f"Training data already extracted.")

    print("\nAll datasets are ready!")


if __name__ == "__main__":
    main()
