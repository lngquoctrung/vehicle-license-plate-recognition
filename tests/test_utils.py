import unittest
import os
from src.utils import create_directory

class TestUtils(unittest.TestCase):
    def test_create_directory(self):
        """
            Check if the directories are created or not
        """
        # Create test directories
        test_path = "parent_test_dir/child_test_dir"
        create_directory(test_path)

        # Check the directories whether exist or not
        self.assertTrue(os.path.exists(test_path))
        print("Test folder(s) created")

        # Remove the created test directories
        os.removedirs(test_path)
        print("Removed test folder(s)")

if __name__ == "__main__":
    unittest.main()