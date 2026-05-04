"""
test.py
-------
Loads fine-tuned checkpoints, evaluates both models on the shared test set,
computes all metrics, and saves a comparison report.

Metrics collected:
  ─ mAP@50        (Mean Average Precision at IoU 0.50)
  ─ mAP@50:95     (COCO primary metric)
  ─ Precision
  ─ Recall
  ─ Latency (ms/image)  — averaged over test set
  ─ Model file size (MB)
  ─ Parameter count

Usage:
    python test.py               # evaluate both models
    python test.py --model rcnn  # evaluate only Faster R-CNN
    python test.py --model yolo  # evaluate only YOLO
    python test.py --no-latency  # skip latency measurement (faster)
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from data   import get_rcnn_dataloader, YOLO_ROOT
from rcnn   import build_faster_rcnn, count_parameters
from util   import (
    load_rcnn, load_yolo,
    measure_latency_rcnn, measure_latency_yolo,
    get_file_size_mb,
    save_metrics, load_metrics,
    print_comparison_table,
    RCNN_CKPT, YOLO_CKPT,
)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

#  Helper: IoU

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes [x1,y1,x2,y2]."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

#  Helper: AP from precision-recall curve

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute area under the precision-recall curve (11-point interpolation)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = precisions[recalls >= t]
        ap  += prec.max() if prec.size > 0 else 0.0
    return ap / 11.0

#  1.  Evaluate Faster R-CNN

def evaluate_rcnn(measure_latency: bool = True) -> dict:
    print("\n" + "═" * 55)
    print(f"  Evaluating Faster R-CNN  |  device={DEVICE}")
    print("═" * 55)

    test_loader, num_classes = get_rcnn_dataloader("test", batch_size=1, shuffle=False)

    # Build model and load fine-tuned weights
    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    model = load_rcnn(model, device=DEVICE)

    # ── Collect predictions
    all_preds, all_gts = [], []
    iou_thresholds = np.arange(0.50, 1.00, 0.05)

    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            preds  = model(images)

            for pred, tgt in zip(preds, targets):
                pred_boxes  = pred["boxes"].cpu()
                pred_scores = pred["scores"].cpu()
                gt_boxes    = tgt["boxes"].cpu()

                all_preds.append({"boxes": pred_boxes, "scores": pred_scores})
                all_gts.append({"boxes": gt_boxes})

    #Compute metrics
    aps_50, aps_50_95 = [], []
    all_prec, all_rec = [], []
    SCORE_THRESH      = 0.5

    for pred, gt in zip(all_preds, all_gts):
        gt_boxes   = gt["boxes"]
        pred_boxes = pred["boxes"]
        scores     = pred["scores"]

        keep       = scores >= SCORE_THRESH
        pred_boxes = pred_boxes[keep]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            aps_50.append(0.0)
            aps_50_95.append(0.0)
            continue

        iou_mat    = box_iou(pred_boxes, gt_boxes)
        best_iou   = iou_mat.max(dim=1).values.numpy()

        # AP@50
        tp_50 = (best_iou >= 0.50).astype(float)
        prec  = np.cumsum(tp_50) / (np.arange(len(tp_50)) + 1)
        rec   = np.cumsum(tp_50) / max(len(gt_boxes), 1)
        aps_50.append(compute_ap(rec, prec))
        all_prec.extend(prec.tolist())
        all_rec.extend(rec.tolist())

        # AP@50:95
        aps_iou = []
        for iou_t in iou_thresholds:
            tp = (best_iou >= iou_t).astype(float)
            p  = np.cumsum(tp) / (np.arange(len(tp)) + 1)
            r  = np.cumsum(tp) / max(len(gt_boxes), 1)
            aps_iou.append(compute_ap(r, p))
        aps_50_95.append(np.mean(aps_iou))

    map50    = float(np.mean(aps_50))    if aps_50    else 0.0
    map50_95 = float(np.mean(aps_50_95)) if aps_50_95 else 0.0
    precision = float(np.mean(all_prec)) if all_prec else 0.0
    recall    = float(np.mean(all_rec))  if all_rec  else 0.0

    # Latency
    latency_ms = 0.0
    if measure_latency:
        test_loader_single, _ = get_rcnn_dataloader("test", batch_size=1, shuffle=False)
        latency_ms = measure_latency_rcnn(model, test_loader_single, DEVICE, n_samples=100)

    # Size
    size_mb = get_file_size_mb(RCNN_CKPT)
    params  = count_parameters(model)

    metrics = {
        "map50":      map50,
        "map50_95":   map50_95,
        "precision":  precision,
        "recall":     recall,
        "latency_ms": latency_ms,
        "size_mb":    size_mb,
        "params":     params,
    }

    print(f"  mAP@50      : {map50:.4f}")
    print(f"  mAP@50:95   : {map50_95:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  Latency     : {latency_ms:.2f} ms/img")
    print(f"  Size        : {size_mb:.1f} MB")
    print(f"  Parameters  : {params:,}")

    return metrics

#  2.  Evaluate YOLOv8

def evaluate_yolo(measure_latency: bool = True) -> dict:
    print("\n" + "═" * 55)
    print(f"  Evaluating YOLOv8  |  device={DEVICE}")
    print("═" * 55)

    model    = load_yolo()
    test_dir = str(YOLO_ROOT / "test" / "images")

    # val() uses the built-in COCO metric engine
    results = model.val(
        data   = str(YOLO_ROOT / "data.yaml"),
        split  = "test",
        device = 0 if DEVICE == "cuda" else "cpu",
        verbose= False,
    )

    map50    = float(results.box.map50)
    map50_95 = float(results.box.map)
    precision= float(results.box.mp)
    recall   = float(results.box.mr)

    # Latency
    latency_ms = 0.0
    if measure_latency:
        latency_ms = measure_latency_yolo(model, test_dir, n_samples=100)

    # Size
    size_mb = get_file_size_mb(YOLO_CKPT)

    from yolov8 import count_yolo_parameters
    params = count_yolo_parameters(model)

    metrics = {
        "map50":      map50,
        "map50_95":   map50_95,
        "precision":  precision,
        "recall":     recall,
        "latency_ms": latency_ms,
        "size_mb":    size_mb,
        "params":     params,
    }

    print(f"  mAP@50      : {map50:.4f}")
    print(f"  mAP@50:95   : {map50_95:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  Latency     : {latency_ms:.2f} ms/img")
    print(f"  Size        : {size_mb:.1f} MB")
    print(f"  Parameters  : {params:,}")

    return metrics

# Save a readable CSV report

def save_csv_report(metrics: dict):
    csv_path = RESULTS / "comparison.csv"
    rows = [
        "Metric,Faster R-CNN,YOLOv8",
        f"mAP@50,{metrics['faster_rcnn'].get('map50','')},{metrics['yolo'].get('map50','')}",
        f"mAP@50:95,{metrics['faster_rcnn'].get('map50_95','')},{metrics['yolo'].get('map50_95','')}",
        f"Precision,{metrics['faster_rcnn'].get('precision','')},{metrics['yolo'].get('precision','')}",
        f"Recall,{metrics['faster_rcnn'].get('recall','')},{metrics['yolo'].get('recall','')}",
        f"Latency_ms,{metrics['faster_rcnn'].get('latency_ms','')},{metrics['yolo'].get('latency_ms','')}",
        f"Size_MB,{metrics['faster_rcnn'].get('size_mb','')},{metrics['yolo'].get('size_mb','')}",
        f"Parameters,{metrics['faster_rcnn'].get('params','')},{metrics['yolo'].get('params','')}",
    ]
    csv_path.write_text("\n".join(rows))
    print(f"[test.py] CSV saved → {csv_path}")

# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare both models")
    parser.add_argument("--model",       choices=["rcnn", "yolo", "both"], default="both")
    parser.add_argument("--no-latency",  action="store_true",
                        help="Skip latency measurement (much faster)")
    args = parser.parse_args()

    existing = load_metrics()
    metrics  = {"faster_rcnn": existing.get("faster_rcnn", {}),
                "yolo":        existing.get("yolo", {})}

    measure_lat = not args.no_latency

    if args.model in ("rcnn", "both"):
        metrics["faster_rcnn"] = evaluate_rcnn(measure_latency=measure_lat)

    if args.model in ("yolo", "both"):
        metrics["yolo"] = evaluate_yolo(measure_latency=measure_lat)

    save_metrics(metrics)
    save_csv_report(metrics)
    print_comparison_table(metrics)
