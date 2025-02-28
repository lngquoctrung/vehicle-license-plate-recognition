import matplotlib.pyplot as plt
import numpy as np
import cv2

from .config import MEAN_NORMALIZATION, STD_NORMALIZATION

def display_images_and_targets(images, targets):
    """
        Display images and bounding boxes
    """
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    k = 0
    for i in range(3):
        for j in range(3):
            # Restore image before normalization
            image = images[k] * STD_NORMALIZATION[:, None, None] + MEAN_NORMALIZATION[:, None, None]

            # Normalize the image values to the range [0, 255] and cast to uint8 type
            image = image.numpy()
            image = np.ceil(image * 255).astype(np.uint8)

            # Transpose image dimensions from (C, H, W) to (H, W, C) for proper display
            image = np.transpose(image, (1, 2, 0))

            # Draw bounding boxes on image
            copy_imgage = image.copy()
            for box in targets[k]["boxes"]:
                # Points of rectangle
                start_points = (int(box[0]), int(box[1]))
                end_points = (int(box[2]), int(box[3]))

                # Color of rectangle - Azure color (RGB color)
                color = (255, 235, 0)
                # Thickness of rectangle
                thickness = 2

                # Draw rectangle for image
                cv2.rectangle(copy_imgage, start_points, end_points, color, thickness)
            # Display image
            ax[i, j].imshow(copy_imgage)
            k += 1