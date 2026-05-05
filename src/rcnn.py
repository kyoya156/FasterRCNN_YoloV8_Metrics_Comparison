"""
rcnn.py
-------
Builds a Faster R-CNN model (ResNet-50 + FPN backbone) using torchvision.
Supports:
  - Loading COCO pre-trained weights
  - Replacing the classification head for a custom number of classes
  - Freezing the backbone for faster fine-tuning
  - Returning the model ready for fine-tuning or inference
"""

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    # Freeze backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Replace classification head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_rcnn_optimizer(model: torch.nn.Module, lr: float = 5e-4, weight_decay: float = 5e-4):
    """Only pass parameters that require grad — frozen backbone params are excluded."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer


def get_rcnn_lr_scheduler(optimizer, step_size: int = 2, gamma: float = 0.5):
    """Decay fast as we're only training the head — step every 2 epochs."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def count_parameters(model: torch.nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = build_faster_rcnn(num_classes=2, pretrained=True, freeze_backbone=True)
    total, trainable = count_parameters(model)
    print(f"Faster R-CNN  |  Total params:     {total:,}")
    print(f"              |  Trainable params: {trainable:,}  (backbone frozen)")
    print(f"              |  Frozen params:    {total - trainable:,}")
    print(model.roi_heads.box_predictor)