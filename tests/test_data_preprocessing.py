import unittest
import os
import shutil

from src.data_preprocessing import download_dataset, preprocess_data
from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE

class TestDataPreprocessing(unittest.TestCase):
    def tearDown(self):
        """
            Clean up after testing
        """
        # Remove test folder
        shutil.rmtree(DATA_DIR, ignore_errors=True)

    def test_download_dataset(self):
        """
            Download dataset from the cloud
        """
        # Fetch dataset from the cloud
        download_dataset(dest=RAW_DATA_DIR)
        print("Data downloaded")

        # Check data whether exist or not
        self.assertTrue("train" in os.listdir(RAW_DATA_DIR))
    
    def test_preprocess_data(self):
        """
            Preprocess data such as image resizing, bounding box adjustment
        """

        # Download dataset if the dataset does not exist
        download_dataset(dest=RAW_DATA_DIR)

        # Preprocess data
        preprocess_data(RAW_DATA_DIR, PROCESSED_DATA_DIR, image_size=IMAGE_SIZE)

        # Show data information before data preprocessing
        for dir_name in os.listdir(RAW_DATA_DIR):
            dir_path = os.path.join(RAW_DATA_DIR, dir_name)
            print(f"The number of files in {dir_name} folder: {len(os.listdir(dir_path))}")

        # Check data whether exist or not
        self.assertTrue("train" in os.listdir(PROCESSED_DATA_DIR))

        # Show data information before data preprocessing
        for dir_name in os.listdir(PROCESSED_DATA_DIR):
            dir_path = os.path.join(PROCESSED_DATA_DIR, dir_name)
            print(f"The number of files in {dir_name} folder: {len(os.listdir(dir_path))}")

if __name__ == "__main__":
    unittest.main()