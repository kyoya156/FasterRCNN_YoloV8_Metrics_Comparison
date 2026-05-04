from ultralytics import YOLO
import time

# 1. Load Model (YOLOv8n is the best baseline)
model = YOLO('yolov8n.pt') 

# 2. Benchmark Speed
results = model.predict(source='path/to/your/images', device='0') # '0' for GPU, 'cpu' for CPU

# 3. Get Metrics (requires val labels)
# metrics = model.val(data='your_data.yaml')
# print(f"YOLO mAP: {metrics.box.map}")


import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# 1. Load Model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval().to('cuda') # or 'cpu'

# 2. Benchmark Speed
img = torch.randn(1, 3, 640, 640).to('cuda') # Dummy image for warm-up
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = model(img)
    end = time.time()

print(f"Faster R-CNN Avg Speed: {(end-start)/100:.4f}s per image")


