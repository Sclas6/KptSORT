from ultralytics import YOLO
import cv2
import tqdm
import math
import numpy as np

model = YOLO("runs/obb/train5/weights/best.pt")

cap = cv2.VideoCapture("sources/Videos/resized_0430.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video = cv2.VideoWriter(f"output/videos/yolo_obb_0430.mp4",fourcc, fps, size)
progress = tqdm.tqdm(total=frames)

while cap.isOpened():
    ret, frame = cap.read()
    results = model.predict(frame, device=0, conf=0.8, verbose=False)
    frame = results[0].plot(conf=False, labels=False) 
    
    progress.update(1)
    video.write(frame)    
    if not ret: break