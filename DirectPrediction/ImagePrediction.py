from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("apple_77.jpg", show=True, imgsz=320, conf=0.5)