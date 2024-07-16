import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile

# Function to load YOLOv8 model based on model name
def load_model(model_name):
    try:
        if model_name == 'yolov8n':
            return torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        elif model_name == 'yolov8m':
            return torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        elif model_name == 'yolov8x':
            return torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        else:
            return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Default to yolov5s if model_name is invalid
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        return None

# Streamlit UI
def main():
    st.title("YOLOv8 物体检测")
    st.write("上传图像或视频，或使用网络摄像头进行 YOLOv8 物体检测。")

    # Model selection
    model_name = st.selectbox("选择模型", ('yolov8n', 'yolov8m', 'yolov8x', 'yolov8s'))

    # 加载YOLOv8模型
    model = load_model(model_name)

    if model is None:
        st.error("模型加载失败。请检查日志以获取详细信息。")
        return

    # Bounding box settings
    show_confidence = st.checkbox("显示置信度", value=True)
    confidence_font_size = st.slider("置信度字体大小", min_value=0.5, max_value=2.0, step=0.1, value=0.5)

    # 上传图像或视频
    uploaded_file = st.file_uploader("选择图像或视频...", type=["jpg", "jpeg", "png", "mp4"])

    # 使用网络摄像头
    use_webcam = st.checkbox("使用网络摄像头")

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
        if st.button('开始检测'):
            if uploaded_file is not None:
                if uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)

                    # 使用YOLOv8模型进行检测
                    processed_image = process_frame(image_np, model, show_confidence, confidence_font_size)

                    st.image(processed_image, caption="检测结果", use_column_width=True)  # 默认显示彩色图像

                elif uploaded_file.type == "video/mp4":
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)

                    stframe = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # 使用YOLOv8模型进行检测
                        processed_frame = process_frame(frame, model, show_confidence, confidence_font_size)

                        stframe.image(processed_frame, channels="BGR")  # 默认显示彩色视频帧

                    cap.release()
            elif use_webcam:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                if not cap.isOpened():
                    st.error("无法访问网络摄像头。请确保已允许访问摄像头，并尝试重新启动浏览器或使用其他浏览器。")
                else:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("无法从网络摄像头读取数据。")
                            break

                        # 使用YOLOv8模型进行检测
                        processed_frame = process_frame(frame, model, show_confidence, confidence_font_size)

                        stframe.image(processed_frame, channels="BGR")

                        if st.button('停止摄像头'):
                            break

                    cap.release()

if __name__ == "__main__":
    main()
