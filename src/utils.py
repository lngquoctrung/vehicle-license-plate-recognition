import os

def split_file(file_path, chunk_size_mb=50):
    """
        Split large dataset to small files
    """
    # Create directory to store small files
    dir_path = "large_file_parts"
    os.makedirs(dir_path, exist_ok=True)

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

def print_tree(path, indent=""):
    # Check if path is a directory
    if not os.path.isdir(path):
        print(f"{path} is not a valid directory.")
        return
    
    # Get a list of items (files and folders) in a directory
    items = os.listdir(path)
    
    # Only take the first 5 files at most and add ellipsis if there are more
    files = [item for item in items if os.path.isfile(os.path.join(path, item))]
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]

    # Print current directory
    print(indent + f"[DIR] {os.path.basename(path)}")

    # Print subfolders
    for dir in dirs:
        print_tree(os.path.join(path, dir), indent + "    ")
    
    # Print files, print only first 5 files at most
    if files:
        for i, file in enumerate(files[:5]):
            print(indent + "    " + f"[FILE] {file}")
        if len(files) > 5:
            print(indent + "    " + "[...]")
