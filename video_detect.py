from ultralytics import YOLO
import cv2
import os
import uuid
from PIL import Image
import numpy as np

def detect_best_face(video_path):
    model = YOLO('yolov8n.pt')  # Use yolov8n for speed, or replace with custom model
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_confidence = 0
    best_crop = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame to speed up
        if frame_count % 10 == 0:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > max_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    max_confidence = conf
                    best_crop = frame[y1:y2, x1:x2]

        frame_count += 1

    cap.release()

    if best_crop is not None:
        filename = "detected.jpg"
        save_path = os.path.join("static/Detected", filename)
        Image.fromarray(cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)).save(save_path)
        return save_path
    return None
