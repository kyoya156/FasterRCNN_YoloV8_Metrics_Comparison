"""
train.py
--------
Fine-tunes both Faster R-CNN and YOLOv8 on the license-plate dataset.

Usage:
    python train.py                  # train both
    python train.py --model rcnn     # train only Faster R-CNN
    python train.py --model yolo     # train only YOLO
    python train.py --epochs 10      # override epoch count
"""

import argparse
import torch
from pathlib import Path

from data import get_rcnn_dataloader, get_yolo_yaml_path
from rcnn import build_faster_rcnn, get_rcnn_optimizer, get_rcnn_lr_scheduler
from yolov8 import build_yolo
from util import save_rcnn, save_yolo, CHECKPOINT_DIR

#  Config

DEFAULT_EPOCHS    = 15
RCNN_BATCH_SIZE   = 4
YOLO_BATCH_SIZE   = 16
YOLO_IMG_SIZE     = 640
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

#  1. Train Faster R-CNN

def train_rcnn(epochs: int = DEFAULT_EPOCHS):
    print("\n" + "═" * 55)
    print(f"  Training Faster R-CNN  |  device={DEVICE}  |  epochs={epochs}")
    print("═" * 55)

    # Data
    train_loader, num_classes = get_rcnn_dataloader("train", batch_size=RCNN_BATCH_SIZE)
    val_loader,   _           = get_rcnn_dataloader("validation", batch_size=RCNN_BATCH_SIZE)

    # Model 
    model     = build_faster_rcnn(num_classes=num_classes, pretrained=True)
    optimizer = get_rcnn_optimizer(model)
    scheduler = get_rcnn_lr_scheduler(optimizer, step_size=max(1, epochs // 3))
    model.to(DEVICE)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, targets in train_loader:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Torchvision Faster-RCNN returns a dict of losses in train mode
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"  [RCNN] Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # Save
    save_rcnn(model)
    print("  [RCNN] Training complete ✓")
    return model

#  2. Train YOLOv8

def train_yolo(epochs: int = DEFAULT_EPOCHS):
    print("\n" + "═" * 55)
    print(f"  Training YOLOv8  |  device={DEVICE}  |  epochs={epochs}")
    print("═" * 55)

    yaml_path = get_yolo_yaml_path()
    model     = build_yolo()          # loads pre-trained yolov8n.pt

    # Ultralytics .train() handles everything: augmentation, LR scheduling, logging
    model.train(
        data       = yaml_path,
        epochs     = epochs,
        imgsz      = YOLO_IMG_SIZE,
        batch      = YOLO_BATCH_SIZE,
        device     = 0 if DEVICE == "cuda" else "cpu",
        project    = "runs/detect",
        name       = "train",
        exist_ok   = True,
        verbose    = True,
        patience   = max(5, epochs // 3),   # early-stopping patience
    )

    save_yolo()
    print("  [YOLO] Training complete ✓")
    return model

#  Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN and/or YOLOv8")
    parser.add_argument("--model",  choices=["rcnn", "yolo", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    print(f"\n[train.py] Using device: {DEVICE.upper()}")

    if args.model in ("rcnn", "both"):
        train_rcnn(epochs=args.epochs)

    if args.model in ("yolo", "both"):
        train_yolo(epochs=args.epochs)

    print("\n[train.py] All done. Checkpoints saved to:", CHECKPOINT_DIR)
