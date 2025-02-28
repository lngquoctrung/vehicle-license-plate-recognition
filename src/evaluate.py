import torch

from .config import DEVICE, IOU_THRESHOLD, NUM_CLASSES
from .model import FTFasterRCNN
from .metrics import iou_metric

def evaluate_model(checkpoint_path, test_dataloader):
    # Load model
    model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    model.eval()
    test_images = []
    test_targets = []
    test_predictions = []
    iou_scores = []

    with torch.no_grad():
        for images, targets in test_dataloader:
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]

            # Predict test images
            predictions = model(images)

            # Compare between predictions and ground truth
            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu()
                target_boxes = target["boxes"].cpu()

                for pred_box in pred_boxes:
                    best_iou = 0.0
                    for true_box in target_boxes:
                        iou = iou_metric(pred_box, true_box)
                        if iou > best_iou:
                            best_iou = iou
                    iou_scores.append(best_iou)

            test_images.extend([image.cpu() for image in images])
            test_targets.extend([{k: v.cpu() for k, v in target.items()} for target in targets])
            test_predictions.extend([{k: v.cpu() for k, v in pred.items()} for pred in predictions])

    corrects = torch.sum(torch.tensor(iou_scores) > IOU_THRESHOLD)
    print("Average IoU:", torch.mean(torch.tensor(iou_scores)).item())
    print(f"Predicted: {corrects}/{len(iou_scores)}")
    print("Percentage of predictions with IoU > 0.5:", (corrects / len(iou_scores) * 100).item(), "%")
    
    return test_images, test_targets, test_predictions