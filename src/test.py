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
  ─ Latency (ms/image)  — averaged over test set, GPU-synchronised
  ─ Model file size (MB)
  ─ Parameter count

Usage:
    python test.py               # evaluate both models
    python test.py --model rcnn  # evaluate only Faster R-CNN
    python test.py --model yolo  # evaluate only YOLO
    python test.py --no-latency  # skip latency measurement (faster)
"""

import argparse
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

# IoU

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

# AP from precision-recall curve

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute area under the precision-recall curve using 11-point interpolation.
    Inputs must be sorted ascending by recall (standard convention).
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = precisions[recalls >= t]
        ap  += prec.max() if prec.size > 0 else 0.0
    return ap / 11.0

# Dataset-level mAP

def compute_dataset_map(all_preds: list, all_gts: list, iou_thresholds: np.ndarray, score_thresh: float = 0.05):
    """
    Compute mAP correctly at the dataset level.

    Rather than averaging per-image AP (biased on small sets), we pool all
    predictions across all images, sort them globally by confidence, then
    compute a single PR curve — matching the PASCAL VOC / COCO convention.

    Parameters
    ----------
    all_preds : list of dict  {boxes: Tensor[N,4], scores: Tensor[N]}
    all_gts   : list of dict  {boxes: Tensor[M,4]}
    iou_thresholds : np.ndarray of IoU thresholds (e.g. [0.50] or np.arange(0.50,1.00,0.05))
    score_thresh   : minimum score to keep a prediction (use a low value like 0.05
                     so the PR curve covers the full recall range)

    Returns
    -------
    map_value : float
    precision : float  (at the operating point closest to the score_thresh cut)
    recall    : float
    """
    aps = []

    for iou_t in iou_thresholds:
        # Collect all predictions with image index
        det_scores, det_tp = [], []
        n_gt_total = 0

        for img_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            gt_boxes   = gt["boxes"]
            pred_boxes = pred["boxes"]
            scores     = pred["scores"]

            n_gt_total += len(gt_boxes)

            keep       = scores >= score_thresh
            pred_boxes = pred_boxes[keep]
            scores     = scores[keep]

            if len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                # All predictions are false positives
                det_scores.extend(scores.tolist())
                det_tp.extend([0] * len(scores))
                continue

            iou_mat  = box_iou(pred_boxes, gt_boxes)   # [P, G]
            matched  = torch.zeros(len(gt_boxes), dtype=torch.bool)

            # Greedy match: sort predictions by score descending
            order    = scores.argsort(descending=True)
            tp_flags = []

            for pidx in order:
                best_iou, best_gidx = iou_mat[pidx].max(0)
                if best_iou >= iou_t and not matched[best_gidx]:
                    matched[best_gidx] = True
                    tp_flags.append(1)
                else:
                    tp_flags.append(0)

            # Re-align with score-sorted order for global pooling
            sorted_scores = scores[order].tolist()
            det_scores.extend(sorted_scores)
            det_tp.extend(tp_flags)

        if n_gt_total == 0:
            aps.append(0.0)
            continue

        # Sort all detections globally by score descending
        order         = np.argsort(det_scores)[::-1]
        tp_sorted     = np.array(det_tp)[order]
        cum_tp        = np.cumsum(tp_sorted)
        cum_fp        = np.cumsum(1 - tp_sorted)

        recalls    = cum_tp / max(n_gt_total, 1)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        aps.append(compute_ap(recalls, precisions))

    map_value = float(np.mean(aps))

    # Operating-point P/R: use all detections at the score_thresh cut
    # (re-use the last iou_t=0.50 pass data; recalculate inline)
    op_prec, op_rec = 0.0, 0.0
    all_s, all_tp_50 = [], []
    n_gt_total = 0

    for pred, gt in zip(all_preds, all_gts):
        gt_boxes   = gt["boxes"]
        pred_boxes = pred["boxes"]
        scores     = pred["scores"]
        n_gt_total += len(gt_boxes)

        keep       = scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        scores     = scores[keep]

        if len(pred_boxes) == 0:
            continue

        iou_mat = box_iou(pred_boxes, gt_boxes) if len(gt_boxes) > 0 else torch.zeros(len(pred_boxes), 0)
        matched = torch.zeros(max(len(gt_boxes), 1), dtype=torch.bool)
        order   = scores.argsort(descending=True)

        for pidx in order:
            if len(gt_boxes) == 0:
                all_tp_50.append(0)
            else:
                best_iou, best_gidx = iou_mat[pidx].max(0)
                if best_iou >= 0.50 and not matched[best_gidx]:
                    matched[best_gidx] = True
                    all_tp_50.append(1)
                else:
                    all_tp_50.append(0)
            all_s.append(scores[pidx].item())

    if all_tp_50:
        ord2     = np.argsort(all_s)[::-1]
        tp_arr   = np.array(all_tp_50)[ord2]
        cum_tp2  = np.cumsum(tp_arr)
        cum_fp2  = np.cumsum(1 - tp_arr)
        precs    = cum_tp2 / (cum_tp2 + cum_fp2 + 1e-6)
        recs     = cum_tp2 / max(n_gt_total, 1)
        # Operating point = last point where recall is maximised
        op_prec  = float(precs[-1])
        op_rec   = float(recs[-1])

    return map_value, op_prec, op_rec

# Evaluate Faster R-CNN

def evaluate_rcnn(measure_latency: bool = True) -> dict:
    print("\n" + "═" * 55)
    print(f"  Evaluating Faster R-CNN  |  device={DEVICE}")
    print("═" * 55)

    test_loader, num_classes = get_rcnn_dataloader("test", batch_size=1, shuffle=False)

    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    model = load_rcnn(model, device=DEVICE)

    # Collect predictions (eval mode — no dropout, no loss dict needed)
    all_preds, all_gts = [], []
    iou_thresholds     = np.arange(0.50, 1.00, 0.05)

    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            preds  = model(images)

            for pred, tgt in zip(preds, targets):
                all_preds.append({
                    "boxes":  pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                })
                all_gts.append({"boxes": tgt["boxes"].cpu()})

    # Compute metrics 
    map50_95, _, _     = compute_dataset_map(all_preds, all_gts, iou_thresholds, score_thresh=0.05)
    map50, precision, recall = compute_dataset_map(all_preds, all_gts, np.array([0.50]), score_thresh=0.5)

    # Latency (GPU-synchronised)
    latency_ms = 0.0
    if measure_latency:
        latency_ms = measure_latency_rcnn(model, test_loader, DEVICE, n_samples=100)

    size_mb = get_file_size_mb(RCNN_CKPT)
    params_total, params_trainable = count_parameters(model)

    metrics = {
        "map50":      map50,
        "map50_95":   map50_95,
        "precision":  precision,
        "recall":     recall,
        "latency_ms": latency_ms,
        "size_mb":    size_mb,
        "params_total": params_total,
        "params_trainable": params_trainable
    }

    print(f"  mAP@50      : {map50:.4f}")
    print(f"  mAP@50:95   : {map50_95:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  Latency     : {latency_ms:.2f} ms/img")
    print(f"  Size        : {size_mb:.1f} MB")
    print(f"  Parameters  : {params_total:,} (total), {params_trainable:,} (trainable)")

    return metrics

# Evaluate YOLOv8

def _verify_yolo_split(yaml_path: str, split: str = "test") -> str:
    """
    Confirm that data.yaml defines the requested split.
    Falls back to 'val' with a warning if 'test' is absent,
    which prevents silently evaluating on the wrong set.
    """
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if split not in cfg:
        print(f"  [YOLO] WARNING: '{split}' split not found in data.yaml "
              f"(keys: {list(cfg.keys())}). Falling back to 'val'. "
              f"Results may be optimistic.")
        return "val"
    return split


def evaluate_yolo(measure_latency: bool = True) -> dict:
    print("\n" + "═" * 55)
    print(f"  Evaluating YOLOv8  |  device={DEVICE}")
    print("═" * 55)

    model     = load_yolo()
    yaml_path = str(YOLO_ROOT / "data.yaml")
    split     = _verify_yolo_split(yaml_path, split="test")   # guarded

    results  = model.val(
        data   = yaml_path,
        split  = split,
        device = 0 if DEVICE == "cuda" else "cpu",
        verbose= False,
    )

    map50    = float(results.box.map50)
    map50_95 = float(results.box.map)
    precision= float(results.box.mp)
    recall   = float(results.box.mr)

    latency_ms = 0.0
    if measure_latency:
        test_dir   = str(YOLO_ROOT / split / "images")
        latency_ms = measure_latency_yolo(model, test_dir, n_samples=100)

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
    parser.add_argument("--model",      choices=["rcnn", "yolo", "both"], default="both")
    parser.add_argument("--no-latency", action="store_true",
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