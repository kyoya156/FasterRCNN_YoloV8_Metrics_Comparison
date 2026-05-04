"""
rcnn.py
-------
Builds a Faster R-CNN model (ResNet-50 + FPN backbone) using torchvision.
Supports:
  - Loading COCO pre-trained weights
  - Replacing the classification head for a custom number of classes
  - Returning the model ready for fine-tuning or inference
"""

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_faster_rcnn(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Build a Faster R-CNN model.

    Args:
        num_classes : total number of classes INCLUDING background
                      (e.g. 2 = background + license-plate)
        pretrained  : load COCO pre-trained backbone weights

    Returns:
        model (torch.nn.Module) with the box predictor replaced
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the box-predictor head to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_rcnn_optimizer(model: torch.nn.Module, lr: float = 5e-4, weight_decay: float = 5e-4):
    """SGD optimizer — standard for Faster R-CNN fine-tuning."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer


def get_rcnn_lr_scheduler(optimizer, step_size: int = 5, gamma: float = 0.5):
    """StepLR: decay LR every `step_size` epochs."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

#  Quick check

if __name__ == "__main__":
    model = build_faster_rcnn(num_classes=2, pretrained=True)
    n = count_parameters(model)
    print(f"Faster R-CNN  |  Parameters: {n:,}")
    print(model.roi_heads.box_predictor)
