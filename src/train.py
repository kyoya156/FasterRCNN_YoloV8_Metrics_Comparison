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
from tqdm import tqdm

from data import get_rcnn_dataloader, get_yolo_yaml_path
from rcnn import build_faster_rcnn, get_rcnn_optimizer, get_rcnn_lr_scheduler
from yolov8 import build_yolo
from util import save_rcnn, save_yolo, CHECKPOINT_DIR

# Config

DEFAULT_EPOCHS    = 5 # i used a 10000 pics dataset, so 5 epochs is a good number.
RCNN_BATCH_SIZE   = 4
YOLO_BATCH_SIZE   = 64
YOLO_IMG_SIZE     = 640
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS       = 0

# Train Faster R-CNN

def train_rcnn(epochs: int = DEFAULT_EPOCHS):
    train_loader, num_classes = get_rcnn_dataloader("train", batch_size=RCNN_BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader,   _           = get_rcnn_dataloader("validation", batch_size=1)

    model     = build_faster_rcnn(num_classes=num_classes, pretrained=True)
    optimizer = get_rcnn_optimizer(model)
    scheduler = get_rcnn_lr_scheduler(optimizer, step_size=max(1, epochs // 3))
    model.to(DEVICE)

    # Epoch-level progress bar
    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc="[RCNN] Epochs",
        unit="epoch",
        colour="cyan",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        train_bar = tqdm(
            train_loader,
            desc=f"  Train {epoch:>3}/{epochs}",
            unit="batch",
            leave=False,
            colour="green",
            dynamic_ncols=True,
        )

        for images, targets in train_bar:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            # Live loss in the batch bar suffix
            train_bar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss = 0.0
        n_val    = 0
        model.train()   # keep train mode so it returns a loss dict on val too

        val_bar = tqdm(
            val_loader,
            desc=f"  Val   {epoch:>3}/{epochs}",
            unit="batch",
            leave=False,
            colour="yellow",
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for images, targets in val_bar:
                images  = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss      = sum(loss_dict.values())
                val_loss += loss.item()
                n_val    += 1

                val_bar.set_postfix(val_loss=f"{val_loss / n_val:.4f}")

        avg_val_loss = val_loss / max(n_val, 1)
        lr_now       = scheduler.get_last_lr()[0]

        # Update the epoch bar with a summary for this epoch
        epoch_bar.set_postfix(
            train=f"{avg_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            lr=f"{lr_now:.2e}",
        )

    print()  # newline after epoch bar closes
    save_rcnn(model)
    tqdm.write("  [RCNN] Training complete ✓")
    return model

# Train YOLOv8

def train_yolo(epochs: int = DEFAULT_EPOCHS):
    yaml_path = get_yolo_yaml_path()
    model     = build_yolo()

    steps = ["Build model", "Train", "Validate", "Save checkpoint"]

    with tqdm(
        steps,
        desc="[YOLO] Pipeline",
        unit="step",
        colour="magenta",
        dynamic_ncols=True,
    ) as step_bar:
        # Build model 
        step_bar.set_description("[YOLO] Build model")
        step_bar.update(1)

        # Train 
        step_bar.set_description("[YOLO] Training")
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
        step_bar.update(1)   # step 2

        # Validate 
        step_bar.set_description("[YOLO] Validating")
        val_results = model.val(
            data=yaml_path,
            split="val",
            verbose=True,
        )
        tqdm.write(f"  [YOLO] Val mAP@50: {val_results.box.map50:.4f}")
        step_bar.update(1)   # step 3

        # Save 
        step_bar.set_description("[YOLO] Saving checkpoint")
        save_yolo()
        step_bar.update(1)   # step 4 

    tqdm.write("  [YOLO] Training completed")
    return model

# Entry point 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN and/or YOLOv8")
    parser.add_argument("--model",  choices=["rcnn", "yolo", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    print(f"\n[train.py] Using device: {DEVICE.upper()}\n")

    if args.model in ("rcnn", "both"):
        train_rcnn(epochs=args.epochs)

    if args.model in ("yolo", "both"):
        train_yolo(epochs=args.epochs)

    print(f"\n[train.py] All done. Checkpoints saved to: {CHECKPOINT_DIR}")