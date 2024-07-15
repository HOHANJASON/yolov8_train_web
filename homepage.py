import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# 模型選擇
model_options = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8x": "yolov8x.pt"
}

model_choice = st.selectbox("選擇 YOLOv8 模型", list(model_options.keys()))
model_path = model_options[model_choice]
model = YOLO(model_path)

st.title("YOLOv8 物件偵測")
st.write("上傳圖像或影片進行 YOLOv8 物件偵測。")

uploaded_file = st.file_uploader("選擇圖像或影片...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        results = model(image_np)
        
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{bbox.cls}: {bbox.conf:.2f}"
                cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
        
            results = model(frame)
            
            for result in results:
                for bbox in result.boxes:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{bbox.cls}: {bbox.conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            stframe.image(frame, channels="BGR")
        
        cap.release()
