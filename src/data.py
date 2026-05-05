"""
data.py
-------
Handles loading and preparation of both dataset formats:
  - COCO JSON format  → used by Faster R-CNN (Torchvision)
  - YOLOv8 format     → used by Ultralytics YOLO

Expected folder structure (matching your screenshot):
  data/
  ├── R-CNN/
  │   ├── train/         # images + _annotations.coco.json
  │   ├── validation/    # images + _annotations.coco.json
  │   └── test/          # images + _annotations.coco.json
  └── Yolo/
      ├── train/         # images + labels/
      ├── validation/    # images + labels/
      ├── test/          # images + labels/
      └── data.yaml
"""
import json
import torch
from pathlib import Path
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

#  Paths
ROOT = Path(__file__).parent.parent / "data"
RCNN_ROOT = ROOT / "R-CNN"
YOLO_ROOT = ROOT / "Yolo"
YOLO_YAML = YOLO_ROOT / "data.yaml"

#  1.  COCO Dataset  (Faster R-CNN)
class CocoLicensePlateDataset(Dataset):
    """
    Reads images + COCO-format annotations.
    Returns (image_tensor, target_dict) compatible with torchvision's
    Faster R-CNN training loop.
    """

    def __init__(self, split: str = "train", transforms=None):
        """
        Args:
            split      : 'train' | 'validation' | 'test'
            transforms : optional torchvision transforms
        """
        split_dir = RCNN_ROOT / split
        ann_file = split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {ann_file}\n"
                "Make sure your R-CNN zip is extracted under data/R-CNN/"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        # Build id → filename map
        self.img_dir = split_dir
        self.transforms = transforms or T.ToTensor()

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image_id
        self.annotations: dict[int, list] = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.ids = list(self.images.keys())

    # helpers
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = [ float(v) for v in ann["bbox"]] # COCO bbox = [x_min, y_min, w, h]
            boxes.append([x, y, x + w, y + h]) # convert to [x1,y1,x2,y2]
            labels.append(int(ann["category_id"]))

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        image = self.transforms(image)
        return image, target

    @property
    def num_classes(self):
        return len(self.categories) + 1   # +1 for background


# Torchvision Faster-RCNN needs a custom collate (variable-size targets)
def collate_fn(batch):
    return tuple(zip(*batch))
    
def get_rcnn_dataloader(split="train", batch_size=4, num_workers=0, shuffle=None):
    """Returns a DataLoader for the Faster R-CNN pipeline."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = CocoLicensePlateDataset(split=split)



    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    ), dataset.num_classes

#  2.  YOLO Dataset path helper

def get_yolo_yaml_path() -> str:
    """
    Returns the absolute path to data.yaml required by Ultralytics.
    If data.yaml doesn't exist yet, auto-generates one from folder layout.
    """
    if YOLO_YAML.exists():
        return str(YOLO_YAML.resolve())

    # Auto-generate a minimal data.yaml
    print(f"[data.py] data.yaml not found — auto-generating at {YOLO_YAML}")
    yaml_content = f"""path: {YOLO_ROOT.resolve()}
        train: train/images
        val:   validation/images
        test:  test/images

        nc: 1
        names: ['license-plate']
        """
    YOLO_YAML.write_text(yaml_content)
    return str(YOLO_YAML.resolve())

#  3.  Quick sanity-check

def verify_datasets():
    print("=" * 55)
    print("  Dataset Verification")
    print("=" * 55)

    # COCO
    for split in ("train", "validation", "test"):
        try:
            ds = CocoLicensePlateDataset(split=split)
            print(f"  [COCO] {split:12s} → {len(ds):5d} images  |  classes: {ds.num_classes}")
        except FileNotFoundError as e:
            print(f"  [COCO] {split:12s} → MISSING ({e})")

    # YOLO yaml
    yaml_path = get_yolo_yaml_path()
    print(f"\n  [YOLO] data.yaml  → {yaml_path}")

    print("=" * 55)


if __name__ == "__main__":
    verify_datasets()
