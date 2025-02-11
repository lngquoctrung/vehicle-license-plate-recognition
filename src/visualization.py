import matplotlib.pyplot as plt
import numpy as np

def display_features_and_labels(images, labels):
    # Display image and label
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    k = 0
    for i in range(3):
        for j in range(3):
            # Restore image before normalization
            image = images[k] * std[:, None, None] + mean[:, None, None]
            # Normalize the image values to the range [0, 255] and cast to uint8 type
            image = image.numpy()
            image = np.ceil(image * 255).astype(np.uint8)
            # Transpose image dimensions from (C, H, W) to (H, W, C) for proper display
            image = np.transpose(image, (1, 2, 0))
            # Adjust label values, scaling the coordinates by IMAGE_SIZE (assumed scaling factor)
            labels[k] = np.ceil(labels[k] * IMAGE_SIZE).astype(np.int32)
            # Points of rectangle
            start_points = (int(labels[k][0]), int(labels[k][1]))
            end_points = (int(labels[k][2]), int(labels[k][3]))
            # Color of rectangle - Azure color (RGB color)
            color = (255, 235, 0)
            # Thickness of rectangle
            thickness = 2

            # Draw rectangle for image
            copy_img = image.copy()
            drawed_img = cv2.rectangle(copy_img, start_points, end_points, color, thickness)

            # Display image
            ax[i, j].imshow(drawed_img)
            k += 1