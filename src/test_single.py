"""
test_single.py
--------------
Run both fine-tuned models on a single image, print a metrics summary,
and display a 2×2 matplotlib comparison plot.

Layout
──────────────────────────────────────────────────
 [0,0] Faster R-CNN detections │ [0,1] YOLOv8 detections
──────────────────────────────────────────────────
 [1,0] Accuracy bar chart      │ [1,1] Latency & size chart
──────────────────────────────────────────────────
The bottom row is only rendered when results/metrics.json exists
(i.e. after a full python test.py run has been done at least once).

Usage:
    python test_single.py --image path/to/plate.jpg
    python test_single.py --image path/to/plate.jpg --score-thresh 0.35
    python test_single.py --image path/to/plate.jpg --save results/cmp.png
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from rcnn import build_faster_rcnn
from util import load_rcnn, load_yolo, load_metrics, RCNN_CKPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference helpers

def _infer_rcnn(model, img_tensor: torch.Tensor) -> tuple:
    """
    Run Faster R-CNN on a single CxHxW float32 tensor in [0,1].
    Returns (boxes [N,4], scores [N], latency_ms) all on CPU.
    GPU-synchronised timing so perf_counter is accurate.
    """
    model.eval()
    inp = [img_tensor.to(DEVICE)]

    # Warm-up
    with torch.no_grad():
        _ = model(inp)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inp)[0]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) * 1000.0

    return out["boxes"].cpu(), out["scores"].cpu(), lat


def _infer_yolo(model, img_path: str) -> tuple:
    """
    Run YOLOv8 on an image file path.
    Returns (boxes [N,4] xyxy, scores [N], latency_ms) all on CPU.
    One throw-away warm-up call is made before timing.
    """
    _ = model.predict(img_path, verbose=False)   # warm-up

    t0     = time.perf_counter()
    result = model.predict(img_path, verbose=False)[0]
    lat    = (time.perf_counter() - t0) * 1000.0

    return result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), lat

# Plot helpers

def _draw_boxes(ax, img_np, boxes, scores, thresh, color, title, lat):
    import matplotlib.patches as mpatches

    ax.imshow(img_np)
    ax.axis("off")

    keep   = scores >= thresh
    boxes  = boxes[keep]
    scores = scores[keep]
    n      = len(boxes)

    for box, sc in zip(boxes.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor=color, facecolor="none",
        ))
        ax.text(x1, y1 - 4, f"{sc:.2f}", color="white", fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.85))

    ax.set_title(
        f"{title}\n{n} detection{'s' if n != 1 else ''} "
        f"(thresh={thresh:.2f})  •  {lat:.1f} ms",
        fontsize=11, pad=8,
    )


def _accuracy_chart(ax, rm, ym):
    labels = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    keys   = ["map50",  "map50_95",  "precision", "recall"]
    rv = [rm.get(k, 0.0) or 0.0 for k in keys]
    yv = [ym.get(k, 0.0) or 0.0 for k in keys]
    y, h = np.arange(len(labels)), 0.35

    br = ax.barh(y + h / 2, rv, h, label="Faster R-CNN", color="#4C9BE8", zorder=3)
    by = ax.barh(y - h / 2, yv, h, label="YOLOv8",       color="#E87B4C", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title("Accuracy metrics", fontsize=10, pad=6)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, col in [(br, "#4C9BE8"), (by, "#E87B4C")]:
        for b in bar:
            v = b.get_width()
            if v > 0:
                ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=7, color=col)


def _speed_chart(ax, rm, ym):
    models    = ["Faster R-CNN", "YOLOv8"]
    latencies = [rm.get("latency_ms", 0.0) or 0.0, ym.get("latency_ms", 0.0) or 0.0]
    sizes     = [rm.get("size_mb",    0.0) or 0.0, ym.get("size_mb",    0.0) or 0.0]
    x, w      = np.arange(2), 0.35
    ax2       = ax.twinx()

    bl  = ax.bar (x - w / 2, latencies, w, label="Latency (ms)", color="#4C9BE8", alpha=0.85, zorder=3)
    bs  = ax2.bar(x + w / 2, sizes,     w, label="Size (MB)",    color="#9B4CE8", alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Latency (ms)", color="#4C9BE8", fontsize=9)
    ax2.set_ylabel("Size (MB)",   color="#9B4CE8", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#4C9BE8")
    ax2.tick_params(axis="y", labelcolor="#9B4CE8")
    ax.set_title("Speed & model size", fontsize=10, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([bl, bs], ["Latency (ms)", "Size (MB)"], fontsize=8, loc="upper right")

    for b in bl:
        v = b.get_height()
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + max(latencies) * 0.02,
                    f"{v:.1f}", ha="center", fontsize=7, color="#4C9BE8")
    for b in bs:
        v = b.get_height()
        if v > 0:
            ax2.text(b.get_x() + b.get_width() / 2, v + max(sizes) * 0.02,
                     f"{v:.1f}", ha="center", fontsize=7, color="#9B4CE8")

# Main

def run(image_path: str, score_thresh: float = 0.5, save_path: str | None = None):
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    pil_img    = Image.open(img_path).convert("RGB")
    img_np     = np.array(pil_img)
    img_tensor = TF.to_tensor(pil_img)

    # Load models
    print(f"\n[test_single] device={DEVICE.upper()}  image={img_path.name}")
    print("  Loading Faster R-CNN …")
    try:
        ckpt       = torch.load(RCNN_CKPT, map_location="cpu")
        cls_key    = [k for k in ckpt if "cls_score.weight" in k]
        num_classes = ckpt[cls_key[0]].shape[0] if cls_key else 2
    except Exception:
        num_classes = 2
    rcnn_model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    rcnn_model = load_rcnn(rcnn_model, device=DEVICE)

    print("  Loading YOLOv8 …")
    yolo_model = load_yolo()

    # Inference 
    print("  Faster R-CNN inference …")
    rcnn_boxes, rcnn_scores, rcnn_lat = _infer_rcnn(rcnn_model, img_tensor)
    print("  YOLOv8 inference …")
    yolo_boxes, yolo_scores, yolo_lat = _infer_yolo(yolo_model, str(img_path))

    # Dataset metrics (optional — from a prior test.py run)
    stored = load_metrics()
    rm     = stored.get("faster_rcnn", {})
    ym     = stored.get("yolo", {})
    rm     = {**rm, "latency_ms": rcnn_lat}   # replace with this-image latency
    ym     = {**ym, "latency_ms": yolo_lat}

    # Terminal summary 
    rn = int((rcnn_scores >= score_thresh).sum())
    yn = int((yolo_scores >= score_thresh).sum())

    print()
    print("┌──────────────────────────────────────────────────────┐")
    print("│           Single-image inference summary             │")
    print("├─────────────────────────┬─────────────┬─────────────┤")
    print("│ Metric                  │ Faster R-CNN│    YOLOv8   │")
    print("├─────────────────────────┼─────────────┼─────────────┤")
    print(f"│ Detections (≥{score_thresh:.2f})      │ {rn:>11d} │ {yn:>11d} │")
    print(f"│ Latency (ms)            │ {rcnn_lat:>11.1f} │ {yolo_lat:>11.1f} │")
    if rm.get("map50") and ym.get("map50"):
        print(f"│ mAP@50  (test set)      │ {rm['map50']:>11.4f} │ {ym['map50']:>11.4f} │")
        print(f"│ mAP@50:95 (test set)    │ {rm['map50_95']:>11.4f} │ {ym['map50_95']:>11.4f} │")
        print(f"│ Precision (test set)    │ {rm['precision']:>11.4f} │ {ym['precision']:>11.4f} │")
        print(f"│ Recall    (test set)    │ {rm['recall']:>11.4f} │ {ym['recall']:>11.4f} │")
    print("└─────────────────────────┴─────────────┴─────────────┘")

    # Plot 
    has_metrics = bool(rm.get("map50") and ym.get("map50"))
    nrows       = 2 if has_metrics else 1

    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(14, 10 if has_metrics else 5),
        gridspec_kw={"height_ratios": [2, 1]} if has_metrics else None,
    )
    fig.patch.set_facecolor("#1a1a2e")

    ax_r, ax_y = (axes[0, 0], axes[0, 1]) if has_metrics else (axes[0], axes[1])

    _draw_boxes(ax_r, img_np, rcnn_boxes, rcnn_scores, score_thresh,
                "#4C9BE8", "Faster R-CNN", rcnn_lat)
    _draw_boxes(ax_y, img_np, yolo_boxes, yolo_scores, score_thresh,
                "#E87B4C", "YOLOv8",       yolo_lat)

    for ax in (ax_r, ax_y):
        ax.set_facecolor("#0f0f1a")
        ax.title.set_color("white")

    if has_metrics:
        ax_acc, ax_spd = axes[1, 0], axes[1, 1]
        for ax in (ax_acc, ax_spd):
            ax.set_facecolor("#12122a")
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        _accuracy_chart(ax_acc, rm, ym)
        _speed_chart(ax_spd,    rm, ym)

        # Dark-style the twinx axis added by _speed_chart
        for child in fig.get_axes():
            if child not in (ax_r, ax_y, ax_acc, ax_spd):
                child.set_facecolor("#12122a")
                child.tick_params(colors="white", labelsize=8)
                child.yaxis.label.set_color("#9B4CE8")
                for spine in child.spines.values():
                    spine.set_edgecolor("#444466")

    fig.suptitle(
        f"License Plate Detection — Model Comparison\n{img_path.name}",
        color="white", fontsize=13, y=1.01,
    )
    plt.tight_layout(pad=2.0)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[test_single] Plot saved → {out}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image visual comparison of both models")
    parser.add_argument("--image",        required=True,  help="Path to the input image")
    parser.add_argument("--score-thresh", type=float, default=0.5,
                        help="Confidence threshold for drawing boxes (default: 0.5)")
    parser.add_argument("--save",         type=str,   default=None,
                        help="Save plot to this path instead of opening a window")
    args = parser.parse_args()

    run(
        image_path   = args.image,
        score_thresh = args.score_thresh,
        save_path    = args.save,
    )