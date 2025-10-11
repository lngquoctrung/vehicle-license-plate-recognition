import torch
import torchvision
import os

from pathlib import Path
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

root_dir = Path(__file__).parent.parent.absolute()
os.environ["TORCH_HOME"] = str(root_dir / "torch_cache")

class FTFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super(FTFasterRCNN, self).__init__()

        # Load pretrained backbone model
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove fully connected layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 1024
        # Freezee pretrained parameters of backbone model
        for param in backbone.parameters():
            param.requires_grad = not freeze_backbone

        for param in backbone[-1].parameters():
            param.requires_grad = True

        # Anchor generator
        ancho_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # RoI Pooling
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=ancho_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)