import os

def create_directory(path):
    """
        Create directories if it does not exist
    """
    # Check the path whether exist or not
    if not os.path.exists(path):
        # Create new directories
        os.makedirs(path)


def split_file(file_path, chunk_size_mb=50):
    """
        Split large dataset to small files
    """
    # Create directory to store small files
    dir_path = "large_file_parts"
    create_directory(dir_path)

    # Small part size in bytes
    chunk_size = chunk_size_mb * 1024 * 1024
    file_name = os.path.basename(file_path)

    with open(file_path, "rb") as file:
        # Read file and split it to small parts
        part_num = 1
        while True:
            chunk = file.read(chunk_size)
            # Break loop if there is no more data to read
            if not chunk:
                break
            # Write small part to new files
            part_file_path = f"{dir_path}/{file_name}.part{part_num}"
            with open(part_file_path, "wb") as part_file:
                part_file.write(chunk)
            part_num += 1


def merge_files():
    pass
