from ultralytics import YOLO

model = YOLO('yolo11n-obb.pt')
result = model.train(data='train2.yaml', epochs=5000, imgsz = 1600, patience = 1000)