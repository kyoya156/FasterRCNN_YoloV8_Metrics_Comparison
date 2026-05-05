"""
util.py
-------
Shared utilities:
  - Saving / loading fine-tuned model checkpoints
  - Latency measurement helper (GPU-synchronised)
  - Metric persistence (JSON)
  - Pretty console printer
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any


CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

METRICS_FILE = Path(__file__).parent.parent / "results" / "metrics.json"
METRICS_FILE.parent.mkdir(exist_ok=True)

# Faster R-CNN save / load

RCNN_CKPT = CHECKPOINT_DIR / "fasterrcnn_finetuned.pth"


def save_rcnn(model: torch.nn.Module, path: Path = RCNN_CKPT):
    torch.save(model.state_dict(), path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"[util] Faster R-CNN saved → {path}  ({size_mb:.1f} MB)")


def load_rcnn(model: torch.nn.Module, path: Path = RCNN_CKPT, device: str = "cpu") -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nRun train.py first.")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[util] Faster R-CNN loaded ← {path}")
    return model

# YOLO save / load 

YOLO_CKPT = CHECKPOINT_DIR / "yolov8_finetuned.pt"


def save_yolo(path: Path = YOLO_CKPT):
    """
    Ultralytics YOLO saves automatically to runs/detect/train/weights/best.pt.
    This helper copies that checkpoint to our standard location.

    Note: the `model` argument has been removed — it was unused (YOLO saves
    its own weights internally) and was misleading.
    """
    import shutil
    best = Path("runs/detect/train/weights/best.pt")
    if best.exists():
        shutil.copy(best, path)
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"[util] YOLO saved → {path}  ({size_mb:.1f} MB)")
    else:
        print(f"[util] Warning: best.pt not found at {best}. Training may not have completed.")


def load_yolo(path: Path = YOLO_CKPT):
    """Load a fine-tuned YOLO model from checkpoint."""
    from ultralytics import YOLO
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nRun train.py first.")
    model = YOLO(str(path))
    print(f"[util] YOLO loaded ← {path}")
    return model

# Latency measurement 

def measure_latency_rcnn(model, dataloader, device, n_samples=100) -> float:
    """
    Measure average *inference-only* time (ms) per image for Faster R-CNN.

    Key correctness points vs. the original:
      - Uses a pre-loaded fixed batch to exclude dataloading time from the measurement.
      - Calls torch.cuda.synchronize() before/after timing on GPU so that
        perf_counter doesn't return before CUDA kernels finish.
      - Runs a dedicated warm-up pass before recording times.
    """
    model.to(device)
    model.eval()

    # Pre-load up to n_samples+1 batches so dataloading is not timed
    batches = []
    for images, _ in dataloader:
        images = [img.to(device) for img in images]
        batches.append(images)
        if len(batches) > n_samples:
            break

    if len(batches) < 2:
        print("[util] Not enough samples for latency measurement.")
        return 0.0

    use_cuda = device == "cuda" and torch.cuda.is_available()

    # Warm-up (excluded from timing)
    with torch.no_grad():
        _ = model(batches[0])
        if use_cuda:
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for images in batches[1:n_samples + 1]:
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) / len(images) * 1000)   # ms per image

    return float(np.mean(times)) if times else 0.0


def measure_latency_yolo(model, image_dir: str, n_samples=100) -> float:
    """
    Measure average inference time (ms) per image for YOLO.
    Ultralytics returns timing info inside the results object.
    """
    imgs = list(Path(image_dir).rglob("*.jpg")) + \
           list(Path(image_dir).rglob("*.png"))
    imgs = imgs[:n_samples]

    if not imgs:
        print(f"[util] No images found in {image_dir} for latency measurement.")
        return 0.0

    # Warm-up
    _ = model.predict(str(imgs[0]), verbose=False)

    times = []
    for img_path in imgs:
        result = model.predict(str(img_path), verbose=False)
        # result[0].speed = {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        times.append(result[0].speed.get("inference", 0.0))

    return float(np.mean(times)) if times else 0.0

# Model size helpers

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2) if Path(path).exists() else 0.0


def count_torch_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# Metrics persistence

def save_metrics(metrics: dict[str, Any], path: Path = METRICS_FILE):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[util] Metrics saved → {path}")


def load_metrics(path: Path = METRICS_FILE) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

# Pretty printer

def print_comparison_table(metrics: dict):
    """Print a formatted comparison table from the metrics dict."""
    rcnn = metrics.get("faster_rcnn", {})
    yolo = metrics.get("yolo", {})

    def fmt(val, suffix=""):
        if isinstance(val, float):
            return f"{val:.4f}{suffix}"
        if val is None:
            return "N/A"
        return str(val) + suffix

    header = f"{'Metric':<28} {'Faster R-CNN':>14} {'YOLOv8':>14}"
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("  Model Comparison — License Plate Detection")
    print("=" * len(header))
    print(header)
    print(sep)
    print(f"{'mAP@50':<28} {fmt(rcnn.get('map50')):>14} {fmt(yolo.get('map50')):>14}")
    print(f"{'mAP@50:95':<28} {fmt(rcnn.get('map50_95')):>14} {fmt(yolo.get('map50_95')):>14}")
    print(f"{'Precision':<28} {fmt(rcnn.get('precision')):>14} {fmt(yolo.get('precision')):>14}")
    print(f"{'Recall':<28} {fmt(rcnn.get('recall')):>14} {fmt(yolo.get('recall')):>14}")
    print(f"{'Latency/image (ms)':<28} {fmt(rcnn.get('latency_ms'), ' ms'):>14} {fmt(yolo.get('latency_ms'), ' ms'):>14}")
    print(f"{'Model size (MB)':<28} {fmt(rcnn.get('size_mb'), ' MB'):>14} {fmt(yolo.get('size_mb'), ' MB'):>14}")
    print(f"{'Parameters':<28} {fmt(rcnn.get('params')):>14} {fmt(yolo.get('params')):>14}")
    print("=" * len(header))

    lat_r = rcnn.get("latency_ms") or 0
    lat_y = yolo.get("latency_ms") or 0
    if lat_r and lat_y and lat_y > 0:
        ratio   = lat_r / lat_y
        faster  = "YOLO" if ratio > 1 else "Faster R-CNN"
        ratio   = max(ratio, 1 / ratio)
        print(f"\n  ▶  {faster} is {ratio:.1f}× faster in inference.")

    map_r = rcnn.get("map50") or 0
    map_y = yolo.get("map50") or 0
    if map_r and map_y:
        better = "Faster R-CNN" if map_r > map_y else "YOLOv8"
        print(f"  ▶  {better} achieves higher mAP@50.")
    print()