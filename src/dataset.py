import os
import numpy as np
import cv2

from lxml import etree
from PIL import Image

from torch.utils import data

class LicensePlateDataset(data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.transform = transform
        self.file_paths = list(set([os.path.splitext(file_path)[0] for file_path in file_paths]))
        self.bboxes = {}

        # Get bounding box in XML files
        for file_path in self.file_paths:
            self.bboxes[file_path] = self.__get_bounding_box(f'{file_path}.xml')

    def __get_bounding_box(self, file_path):
        """
            Get bounding box from XML file
        """
        # Read XML file
        tree = etree.parse(file_path)
        # In case an image has multiple bounding boxes, only take the first bounding box
        object_tag = tree.getroot().findall('object')[0]
        bbox_tag = object_tag.find('bndbox')
        # Original bounding box
        xmin = int(bbox_tag.find('xmin').text)
        ymin = int(bbox_tag.find('ymin').text)
        xmax = int(bbox_tag.find('xmax').text)
        ymax = int(bbox_tag.find('ymax').text)

        return np.array([xmin, ymin, xmax, ymax])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_file_path = f'{self.file_paths[idx]}.jpg'

        # Load image and get bounding box of this image
        image = cv2.imread(image_file_path)
        bbox = self.bboxes[self.file_paths[idx]]

        # Orignal width and height of image
        original_height, original_width, _ = image.shape

        # Convert from BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert from numpy image to PIL image
        image = Image.fromarray(image)

        if self.transform:
            # Apply transformation
            image = self.transform(image)

        # Normalization label
        _, resized_height, resized_width = image.shape
        bbox = bbox.astype(np.float32)

        # Resize bounding box if image has been resized
        bbox[0], bbox[2] = bbox[0] * (resized_width / original_width), bbox[2] * (resized_width / original_width)
        bbox[1], bbox[3] = bbox[1] * (resized_height / original_height), bbox[3] * (resized_height / original_height)

        # Label normalization
        bbox[0], bbox[2] = bbox[0] / resized_width, bbox[2] / resized_width
        bbox[1], bbox[3] = bbox[1] / resized_height, bbox[3] / resized_height

        return image, bbox