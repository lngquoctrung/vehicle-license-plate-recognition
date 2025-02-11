import requests
import os
import zipfile
import cv2

from lxml import etree
from tqdm import tqdm

from .utils import create_directory
from .config import DATA_URL

def download_dataset(dest="data/raw"):
    """
        Download dataset from the cloud
    """
    # Create folder data to store dataset if it does not exist
    create_directory(dest)

    # File name and destination path where file is stored
    zip_file_name = "license-plate-project.zip"
    zip_file_path = f'{dest}/{zip_file_name}'

    # Download dataset if data folder is empty
    if not "train" in os.listdir(dest):
        print('Dataset does not exist, please waiting to download data from the cloud...')
        
        # Fetch parts of dataset file
        responses = []
        total_size = 0
        num_parts = 20

        print('Start to download...')
        for i in range(num_parts):
            url = f"{DATA_URL}/license-plate-project.zip.part{i + 1}"
            try:
                response = requests.get(url, stream=True)
                total_size += int(response.headers.get('Content-Length', 0))
                if i % 5 == 0:
                    print(f"{i}/{num_parts} of data are downloaded")
                responses.append(response)
            except Exception as e:
                print(e)

        # Display progress when download file
        with open(zip_file_path, 'wb') as file, tqdm(
            desc=zip_file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
            position=0,
        ) as progress_bar:
            for response in responses:
                for content in response.iter_content(chunk_size=1024):
                    size = file.write(content)
                    progress_bar.update(size)

        # Extract zip file
        print('Downloaded, please waiting to extract file...')
        print('Extracting...')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dest)
            print('Extracted!!')
        print('Removing zip file...')
        os.remove(zip_file_path)
        print('Removed!!')
        
        # Remove unnecessary files
        dirs = os.listdir(dest)
        for dir in dirs:
            dir_path = os.path.join(dest, dir)
            if os.path.isfile(dir_path):
                os.remove(dir_path)

    print('Done!!')

def preprocess_data(data_path, dest="data/processed", image_size=(320, 320)):
    """
        Resize image, adjust bounding box and remove sample without bounding box
    """

    def has_bounding_box(file_path):
        """
            Check xml file whether has object tag or not
        """
        try:
            tree = etree.parse(file_path)
            root = tree.getroot()
            object_tag = root.findall('object')[0]
            bbox_tag = object_tag.find('bndbox')
            if bbox_tag is not None:
                return True
        except Exception as e:
            return False
        return False
    
    def preprocess_xml_file(file_path, image_size):
        """
            Remove duplicate bounding boxs and adjust bounding box
        """
        tree = etree.parse(file_path)
        root = tree.getroot()
        # Find bounding boxes
        object_elements = root.findall(".//object")
        
        # Keep the first bounding box if there have many bounding boxes
        if len(object_elements) > 1:
            for object_element in object_elements[1:]:
                root.remove(object_element)

        # Original image size
        size_element = root.find("size")
        width_element = size_element.find("width")
        height_element = size_element.find("height")
        image_width = int(width_element.text)
        image_height = int(height_element.text)
        # Update new image size
        height_element.text = str(image_size[0])
        width_element.text = str(image_size[1])

        # Ajust bounding box
        bbox_element = object_elements[0].find('bndbox')
        xmin_element = bbox_element.find("xmin")
        ymin_element = bbox_element.find("ymin")
        xmax_element = bbox_element.find("xmax")
        ymax_element = bbox_element.find("ymax")
        # Original bounding box
        xmin = int(xmin_element.text)
        ymin = int(ymin_element.text)
        xmax = int(xmax_element.text)
        ymax = int(ymax_element.text)
        # Adjust bounding box based on new image size
        new_xmin = int(xmin * (image_size[1] / image_width))
        new_ymin = int(ymin * (image_size[0] / image_height))
        new_xmax = int(xmax * (image_size[1] / image_width))
        new_ymax = int(ymax * (image_size[0] / image_height))
        # Update new bounding box in XML file
        xmin_element.text = str(new_xmin)
        ymin_element.text = str(new_ymin)
        xmax_element.text = str(new_xmax)
        ymax_element.text = str(new_ymax)

        return tree

    for dir_name in os.listdir(data_path):
        raw_dir_path = os.path.join(data_path, dir_name)
        dest_dir_path = os.path.join(dest, dir_name)

        # Create directory to store processed data
        create_directory(dest_dir_path)

        # The image file name and the xml file name are the same
        unique_filenames = set([os.path.splitext(filename)[0] for filename in os.listdir(raw_dir_path)])

        with tqdm(iterable=unique_filenames, desc=f"Preprocessing data in {dir_name} folder...") as progress_bar:
            for filename in progress_bar:
                image_path = os.path.join(raw_dir_path, f"{filename}.jpg")
                xml_path = os.path.join(raw_dir_path, f"{filename}.xml")
                output_image_path = os.path.join(dest_dir_path, f"{filename}.jpg")
                output_xml_path = os.path.join(dest_dir_path, f"{filename}.xml")

                # Ignore file without bounding box
                if has_bounding_box(xml_path):
                    # Resize image
                    image = cv2.imread(image_path)
                    resized_image = cv2.resize(image, image_size)
                    # Export new image file to destination folder
                    cv2.imwrite(output_image_path, resized_image)

                    # Preprocess XML file
                    tree = preprocess_xml_file(xml_path, image_size)
                    # Export new XML file to destination folder
                    tree.write(output_xml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")