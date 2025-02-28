import unittest
import os
import shutil

from src.config import (
    DATA_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    IMAGE_SIZE, 
    MEAN_NORMALIZATION, 
    STD_NORMALIZATION,
    NUM_WORKERS,
    BATCH_SIZE
)
from src.dataset import LicensePlateDataset
from src.data_preprocessing import download_dataset, preprocess_data

from torchvision import transforms
from torch.utils.data import DataLoader

class TestDataset(unittest.TestCase):
    def setUp(self):
        """
            Set up the dataset and dataloader for testing
        """

        # Download dataset from Github
        download_dataset(dest=RAW_DATA_DIR)

        # Preprocess data
        preprocess_data(RAW_DATA_DIR, dest=PROCESSED_DATA_DIR, image_size=IMAGE_SIZE)

        # Get the list of file paths in each folder
        self.file_pathes = {}
        for dir_name in os.listdir(PROCESSED_DATA_DIR):
            dir_path = os.path.join(PROCESSED_DATA_DIR, dir_name)
            self.file_pathes[dir_name] = []
            for filename in os.listdir(dir_path):
                self.file_pathes[dir_name].append(os.path.join(dir_path, filename))

        # Transform images
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION)
        ])

    def tearDown(self):
        """
            Clean up after testing
        """
        # Remove test folder
        shutil.rmtree(DATA_DIR, ignore_errors=True)

    def test_dataset(self):
        """
            Create dataset and dataloader
        """

        # Declare dataset and dataloader
        train_dataset = LicensePlateDataset(file_paths=self.file_pathes["train"], transform=self.transform)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=True, 
            pin_memory=True,
            shuffle=True,
            prefetch_factor=10,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        # Get the first batch to do the test
        batch = next(iter(train_dataloader))
        images, targets = batch

        # Test the shape of the first batch
        self.assertEqual(images[0].shape, (3, 240, 240))
        self.assertEqual(len(targets), 32)

        # Display information of first 5 batches
        for idx, (images, targets) in enumerate(train_dataloader):
            if idx == 5: 
                break
            print(f"Batch {idx}: ")
            print(f"The number of images: {len(images)}")
            print(f"The number of labels: {len(targets)}")
            print("-" * 20)

if __name__ == "__main__":
    unittest.main()