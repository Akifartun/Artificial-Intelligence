from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8m.pt")

# Run inference on 'bus.jpg' with arguments
model.train(data="data.yaml", epochs=200, imgsz=640)
