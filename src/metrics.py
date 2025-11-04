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
    iou = inter_area / (union_area + 1e-6) # Plus epsilon to avoid division by zero
    
    return iou

def accuracy_metric(pred_boxes, true_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0
    
    correct = 0
    for pred_box in pred_boxes:
        best_iou = 0.0
        for true_box in true_boxes:
            iou = iou_metric(pred_box, true_box)
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            correct += 1
    
    accuracy = correct / len(pred_boxes)
    return accuracy