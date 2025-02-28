import unittest
import torch
import torchinfo
from src.config import NUM_CLASSES, IOU_THRESHOLD, IMAGE_SIZE
from src.model import FTFasterRCNN
from src.metrics import iou_metric

class TestModels(unittest.TestCase):
    def test_initialize_fasterrcnn_model(self):
        """
            Initialize fine tuned Faster RCNN model
        """

        # Initialize fine tuned Faster RCNN model
        faster_rcnn_model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)

        # Create a sample and predict
        dummy_input = torch.randn(1, 3, 240, 240)
        targets = [{
            "boxes": torch.tensor([[50, 50, 150, 100]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        }]

        faster_rcnn_model.train()
        with torch.no_grad():
            loss_dict = faster_rcnn_model(dummy_input, targets)
        faster_rcnn_model.eval()
        with torch.no_grad():
            predictions = faster_rcnn_model(dummy_input)
        
        # Display the output
        print(loss_dict)
        print(predictions)
        print("-" * 60)

        # Calculate IoU score
        iou_scores = []
        for pred, target in zip(predictions, targets):
            for pred_box in pred["boxes"]:
                best_iou = 0.0
                for true_box in target["boxes"]:
                    iou_score = iou_metric(pred_box, true_box)
                    if iou_score > best_iou:
                        best_iou = iou_score
                iou_scores.append(best_iou)
        print(iou_scores)
        # Assume the threshold is 0.1
        print(torch.sum(torch.tensor(iou_scores) > (IOU_THRESHOLD - 0.2)))
        print("-" * 60)

        # Display info of the model
        torchinfo.summary(faster_rcnn_model, (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), batch_dim=0, depth=3)
        print("-" * 60)

        # Check the shape of output
        self.assertEqual(predictions[0]["boxes"].shape[1], 4)
if __name__ == "__main__":
    unittest.main()