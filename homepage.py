import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile

# Function to load YOLOv8 model based on model name
def load_model(model_name):
    if model_name == 'yolov8n':
        return torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    elif model_name == 'yolov8m':
        return torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    elif model_name == 'yolov8x':
        return torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    else:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Default to yolov5s if model_name is invalid

# Streamlit UI
st.title("YOLOv8 物件偵測")
st.write("上傳圖像或影片，或使用網絡攝像頭進行 YOLOv8 物件偵測。")

# Model selection
model_name = st.selectbox("選擇模型", ('yolov8s', 'yolov8n', 'yolov8m', 'yolov8x'))

# 加載YOLOv8模型
model = load_model(model_name)

# Bounding box settings
show_confidence = st.checkbox("顯示信心值", value=True)
confidence_font_size = st.slider("信心值字體大小", min_value=0.5, max_value=2.0, step=0.1, value=0.5)

# 上傳圖像或影片
uploaded_file = st.file_uploader("選擇圖像或影片...", type=["jpg", "jpeg", "png", "mp4"])

# 選擇使用網絡攝像頭
use_webcam = st.checkbox("使用網絡攝像頭")

# Function to process frames for detection
def process_frame(frame, model, show_confidence, confidence_font_size):
    results = model(frame)
    for result in results.xyxy[0].numpy():
        x1, y1, x2, y2, conf, cls = result
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        label = f"{model.names[int(cls)]} {conf:.2f}" if show_confidence else model.names[int(cls)]
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, confidence_font_size, (255, 0, 0), 2)
    return frame

if uploaded_file is not None or use_webcam:
    if st.button('進行推導'):
        if uploaded_file is not None:
            if uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # 使用YOLOv8模型進行推導
                processed_image = process_frame(image_np, model, show_confidence, confidence_font_size)
                
                st.image(processed_image, caption="檢測結果", use_column_width=True)  # 默认显示彩色图像
            
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
                    processed_frame = process_frame(frame, model, show_confidence, confidence_font_size)
                    
                    stframe.image(processed_frame, channels="BGR")  # 默认显示彩色视频帧
                
                cap.release()
        elif use_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            
            if not cap.isOpened():
                st.write("無法訪問網絡攝像頭。請檢查是否已經允許訪問攝像頭，或重啟瀏覽器並嘗試使用其他瀏覽器。")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("無法從網絡攝像頭讀取數據。")
                        break
                    
                    # 使用YOLOv8模型進行推導
                    processed_frame = process_frame(frame, model, show_confidence, confidence_font_size)
                    
                    stframe.image(processed_frame, channels="BGR")
                    
                    if st.button('停止攝像頭'):
                        break
                
                cap.release()
