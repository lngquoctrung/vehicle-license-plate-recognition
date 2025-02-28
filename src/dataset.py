import numpy as np
import cv2

from pathlib import Path
from lxml import etree
from PIL import Image
import torch

from torch.utils import data

class LicensePlateDataset(data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.transform = transform
        self.data = []

        # Extract data from XML files
        xml_file_paths = [file_path for file_path in file_paths if file_path.endswith(".xml")]
        for xml_file_path in xml_file_paths:
            self.data.append(self.__parse_xml(xml_file_path))

    def __parse_xml(self, xml_file_path):
        # The path of image file
        image_file_path = Path(xml_file_path).with_suffix(".jpg")

        # Retrieve image information and bounding boxes
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        # The original image size
        size_element = root.find("size")
        image_width = int(size_element.find("width").text)
        image_height = int(size_element.find("height").text)

        # Get bounding boxes
        objects = []
        for object in root.findall(".//object"):
            # Assume name is label for images
            name = object.find("name").text

            # Get bounding boxes
            bndbox_element = object.find("bndbox")
            xmin = int(bndbox_element.find("xmin").text)
            ymin = int(bndbox_element.find("ymin").text)
            xmax = int(bndbox_element.find("xmax").text)
            ymax = int(bndbox_element.find("ymax").text)

            objects.append({
                "label": name,
                "bbox": [xmin, ymin, xmax, ymax]
            })

        return {
            "image_file_path": image_file_path,
            "width": image_width,
            "height": image_height,
            "objects": objects
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and get bounding box of this image
        image = cv2.imread(self.data[idx]["image_file_path"])
        # Convert from BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert from numpy image to PIL image
        image = Image.fromarray(image)
        if self.transform:
            # Apply transformation
            image = self.transform(image)

        # Orignal width and height of image
        original_height, original_width = self.data[idx]["height"], self.data[idx]["width"]
        # Normalization label
        _, resized_height, resized_width = image.shape

        boxes = []
        labels = []
        for object in self.data[idx]["objects"]:
            box = object["bbox"]
            # Resize bounding box if image has been resized
            xmin = box[0] * (resized_width / original_width)
            ymin = box[1] * (resized_height / original_height)
            xmax = box[2] * (resized_width / original_width)
            ymax = box[3] * (resized_height / original_height)

            boxes.append([xmin, ymin, xmax, ymax])
            # Assuming all objects are license plates (class 1)
            labels.append(1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # The label
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target