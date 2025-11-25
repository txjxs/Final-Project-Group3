import os
import requests
import requests as r
import tqdm as tqdm
import zipfile as zf

URLS: dict[str, str] = {
    'ut-zap50k-data.zip': 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip',
    'ut-zap50k-feats.zip': 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-feats.zip',
    'ut-zap50k-images.zip': 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip',
    'ut-zap50k-lexi.zip': 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-lexi.zip',
    'readme.txt': 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/readme.txt'}
# %%
PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PATH), 'data')
print(DATA_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(DATA_DIR)


# %%
def download(URLS):
    for filename, url in URLS.items():
        file_path = os.path.join(DATA_DIR, filename)
        print(f'Downloading {filename}... at {file_path}')
        try:
            response = requests.get(url, stream=True)
            total_size_bytes = int(response.headers.get('content-length', 0))
            print(f'size...{total_size_bytes / 1e+6}')
            with open(file_path, 'wb') as f, tqdm.tqdm(
                    desc=filename,
                    total=total_size_bytes,
                    unit='iB',
                    unit_scale=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        except r.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
            continue

        if filename.endswith('.zip'):
            print(f"Unzipping {filename}...")
            try:
                with zf.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                    print(f'Extracted {file_path} at {DATA_DIR}')
            except zf.BadZipFile as e:
                print(f'Error extracting {file_path}: {e}')
                continue

            print(f'Cleaning up {filename}...')
            os.remove(file_path)
        else:
            print(f"Saved {filename}.")


download(URLS)

