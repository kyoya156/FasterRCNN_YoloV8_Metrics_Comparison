"""
yolov8.py
---------
Builds / loads a YOLOv8 (or YOLO11) model using the Ultralytics library.
Supports:
  - Auto-downloading pre-trained COCO weights
  - Exposing training and inference interfaces
  - Parameter / size reporting
"""

from pathlib import Path
from ultralytics import YOLO


# Default model variant
#   yolov8n.pt  → nano   (fastest, smallest)
#   yolov8s.pt  → small
#   yolo11n.pt  → YOLO11 nano (newer architecture)
DEFAULT_WEIGHTS = "yolov8n.pt"


def build_yolo(weights: str = DEFAULT_WEIGHTS) -> YOLO:
    """
    Load a YOLO model with pre-trained COCO weights.
    If the .pt file is not present locally, Ultralytics downloads it automatically.

    Args:
        weights : model checkpoint string, e.g. 'yolov8n.pt' or 'yolo11n.pt'

    Returns:
        YOLO model instance
    """
    model = YOLO(weights)
    return model


def count_yolo_parameters(model: YOLO) -> int:
    """Return total number of parameters in the underlying PyTorch model."""
    return sum(p.numel() for p in model.model.parameters())


def get_model_file_size_mb(weights_path: str) -> float:
    """Return file size of a .pt checkpoint in MB."""
    p = Path(weights_path)
    if p.exists():
        return p.stat().st_size / (1024 ** 2)
    return 0.0

#  Quick check

if __name__ == "__main__":
    model = build_yolo()
    n = count_yolo_parameters(model)
    print(f"YOLOv8n  |  Parameters: {n:,}")
    print(f"         |  Model info:")
    model.info()
