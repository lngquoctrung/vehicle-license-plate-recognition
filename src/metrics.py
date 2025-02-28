import torch

def iou_metric(pred_box, true_box):
    # Get points of intersection box
    inter_x_min = torch.maximum(pred_box[0], true_box[0])
    inter_y_min = torch.maximum(pred_box[1], true_box[1])
    inter_x_max = torch.minimum(pred_box[2], true_box[2])
    inter_y_max = torch.minimum(pred_box[3], true_box[3])
    # Calculate width and height of intersection box
    inter_width = torch.maximum((inter_x_max - inter_x_min), torch.tensor(0.0))  # Avoid negative width
    inter_height = torch.maximum((inter_y_max - inter_y_min), torch.tensor(0.0))
    # Calculate intersection area
    inter_area = inter_width * inter_height

    # Calculate area of two bounding boxes
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    true_box_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    # Calculate union area
    union_area = (pred_box_area + true_box_area) - inter_area

    # Calculate IoU score
    return inter_area / union_area