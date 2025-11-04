import cv2
import albumentations as A

from pathlib import Path
from lxml import etree
import torch

from torch.utils import data

class LicensePlateDataset(data.Dataset):
    def __init__(self, file_paths, transforms=None):
        self.transforms = transforms
        self.data = []

        # Extract data from XML files
        xml_file_paths = [file_path for file_path in file_paths if file_path.endswith(".xml")]
        for xml_file_path in xml_file_paths:
            data_item = self.__parse_xml(xml_file_path)
            if len(data_item["objects"]) > 0:
                self.data.append(data_item)

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
        image = cv2.imread(str(self.data[idx]["image_file_path"]))
        # Convert from BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes and labels
        boxes = [obj["bbox"] for obj in self.data[idx]["objects"]]
        labels = [1 for _ in self.data[idx]["objects"]]  # All license plates

        # Get image dimensions
        height, width = image.shape[:2]

        normalized_boxes = []
        for bbox in boxes:
            xmin, ymin, xmax, ymax = bbox
            # Normalize bounding box
            norm_bbox = [
                xmin / width,
                ymin / height,
                xmax / width,
                ymax / height
            ]
            norm_bbox = [
                max(0.0, min(1.0, coord)) for coord in norm_bbox
            ]
            normalized_boxes.append(norm_bbox)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=normalized_boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            boxes = normalized_boxes

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.data))
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return image, {"boxes": boxes, "labels": labels}