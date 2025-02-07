import requests
import os
import zipfile
from tqdm import tqdm

from .utils import create_directory
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def download_dataset(url):
    """
        Download dataset from the cloud
    """
    # Create folder data to store dataset if it does not exist
    create_directory(RAW_DATA_DIR)

    # File name and destination path where file is stored
    zip_file_name = url.split('/')[-1]
    zip_file_path = f'{RAW_DATA_DIR}/{zip_file_name}'

    # Download dataset if data folder is empty
    if not "train" in os.listdir(RAW_DATA_DIR):
        print('Dataset does not exist, please waiting to download data from the cloud...')
        # Download zip file
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('Content-Length', 0))
        print('Start to download...')

        # Display progress when download file
        with open(zip_file_path, 'wb') as file, tqdm(
            desc='Downloading...',
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
            position=0,
        ) as progress_bar:
            for content in response.iter_content(chunk_size=1024):
                size = file.write(content)
                progress_bar.update(size)

        # Extract zip file
        print('Downloaded, please waiting to extract file...')
        print('Extracting...')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            print('Extracted!!')
        print('Removing zip file...')
        os.remove(zip_file_path)
        print('Removed!!')
        
        # Remove unnecessary files
        dirs = os.listdir(RAW_DATA_DIR)
        for dir in dirs:
            dir_path = os.path.join(RAW_DATA_DIR, dir)
            if os.path.isfile(dir_path):
                os.remove(dir_path)

    print('Done!!')

def preprocessing():
    """
        Resize image, adjust bounding box and remove sample without bounding box
    """
    pass