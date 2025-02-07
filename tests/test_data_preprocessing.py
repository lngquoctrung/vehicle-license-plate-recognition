import unittest
import os
import shutil

from src.data_preprocessing import download_dataset
from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

class TestDataPreprocessing(unittest.TestCase):
    def test_download_dataset(self):
        """
            Download dataset from the cloud
        """
        # Fetch dataset from the cloud
        download_dataset()

        # Check data whether exist or not
        self.assertTrue("train" in os.listdir(RAW_DATA_DIR))

        # Remove test folder
        shutil.rmtree(DATA_DIR, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()