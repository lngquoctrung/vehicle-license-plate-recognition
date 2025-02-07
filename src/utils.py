import os

def create_directory(path):
    """
        Create directories if it does not exist
    """
    # Check the path whether exist or not
    if not os.path.exists(path):
        # Create new directories
        os.makedirs(path)