import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import os

st.title("YOLOv8 物件偵測")
st.write("上傳圖像或影片進行 YOLOv8 物件偵測。")

# 獲取模型文件的相對路徑
model_dir = "models"  # 子目錄名稱
model_filename = "yolov8n.pt"  # 模型文件名稱
model_path = os.path.join(model_dir, model_filename)

# 加載YOLOv8模型
model = YOLO(model_path)

# Bounding box settings
show_confidence = st.checkbox("顯示信心值", value=True)
confidence_font_size = st.slider("信心值字體大小", min_value=0.5, max_value=2.0, step=0.1, value=0.5)

# 上傳圖像或影片
uploaded_file = st.file_uploader("選擇圖像或影片...", type=["jpg", "jpeg", "png", "mp4"])

def draw_boxes(image, results, show_confidence, confidence_font_size):
    for result in results:
        x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy().astype(int)
        conf = result.boxes.conf[0]
        label = result.boxes.cls[0]
        class_name = model.names[int(label)]
        label_text = f"{class_name} {conf:.2f}" if show_confidence else class_name
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, confidence_font_size, (255, 0, 0), 2)
    return image

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 使用YOLOv8模型進行推導
        results = model(image_np)

        # 在圖像上繪製邊界框和標籤
        image_np = draw_boxes(image_np, results, show_confidence, confidence_font_size)
        st.image(image_np, caption="檢測結果", use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 使用YOLOv8模型進行推導
            results = model(frame)

            # 在影片帧上繪製邊界框和標籤
            frame = draw_boxes(frame, results, show_confidence, confidence_font_size)
            stframe.image(frame, channels="BGR")

        cap.release()

# 使用攝像頭進行即時檢測
use_webcam = st.checkbox("使用攝像頭進行即時檢測")
if use_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLOv8模型進行推導
        results = model(frame)

        # 在攝像頭帧上繪製邊界框和標籤
        frame = draw_boxes(frame, results, show_confidence, confidence_font_size)
        stframe.image(frame, channels="BGR")
    
    cap.release()
