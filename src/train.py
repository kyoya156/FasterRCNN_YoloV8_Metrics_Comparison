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
YOLO_BATCH_SIZE   = 64
YOLO_IMG_SIZE     = 640
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS      = 0

#  1. Train Faster R-CNN

def train_rcnn(epochs: int = DEFAULT_EPOCHS):
    train_loader, num_classes = get_rcnn_dataloader("train", batch_size=RCNN_BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader,   _           = get_rcnn_dataloader("validation", batch_size=1)

    model     = build_faster_rcnn(num_classes=num_classes, pretrained=True)
    optimizer = get_rcnn_optimizer(model)
    scheduler = get_rcnn_lr_scheduler(optimizer, step_size=max(1, epochs // 3))
    model.to(DEVICE)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, targets in train_loader:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss = 0.0
        n_val    = 0
        model.train()   # keep train mode so it returns loss dict on val too

        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss      = sum(loss_dict.values())
                val_loss += loss.item()
                n_val    += 1

        avg_val_loss = val_loss / max(n_val, 1)
        lr_now       = scheduler.get_last_lr()[0]

        print(f"  [RCNN] Epoch {epoch:3d}/{epochs}  "
              f"train_loss={avg_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"lr={lr_now:.2e}")

    save_rcnn(model)
    print("  [RCNN] Training complete")
    return model

#  2. Train YOLOv8

def train_yolo(epochs: int = DEFAULT_EPOCHS):
    yaml_path = get_yolo_yaml_path()
    model = build_yolo()

    # Train
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        device=0 if DEVICE == "cuda" else "cpu",
        project="runs/detect",
        name="train",
        exist_ok=True,
        patience=max(5, epochs // 3),
    )

    # Explicit validation on the val split after training completes
    val_results = model.val(
        data=yaml_path,
        split="val",
        verbose=True,
    )
    print(f"  [YOLO] Val mAP@50: {val_results.box.map50:.4f}")

    save_yolo()
    print("  [YOLO] Training complete")
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
