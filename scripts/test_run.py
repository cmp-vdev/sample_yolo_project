from ultralytics import YOLO

# Load a pre-trained YOLOv11n model
model = YOLO("../models/yolo11n.pt")

# Run inference on a test image
results = model("https://ultralytics.com/images/bus.jpg")

# Print detections
print(f"Detected {len(results[0].boxes)} objects")

for box in results[0].boxes:
    print(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}")
