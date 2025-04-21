import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import io

# --- IMPORTANT: Set page config FIRST
st.set_page_config(page_title="Glasses Detector", page_icon="🕶️", layout="wide")

# --- Title and Description
st.title("🕶️ Glasses Detection App")
st.write("Upload an image to detect glasses using a YOLO model!")

# --- Load the YOLO model
@st.cache_resource
def load_model():
    model = YOLO('model.pt')  # <-- Put your model path here
    return model

model = load_model()

# --- Sidebar settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
st.sidebar.write("Adjust the detection confidence.")

# --- File uploader
uploaded_files = st.file_uploader(
    "Upload image(s)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# --- Processing uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        with st.spinner(f"Detecting glasses in {uploaded_file.name}..."):
            results = model.predict(
                source=np.array(image), 
                conf=confidence_threshold,
                save=False
            )

        # Show the result image
        result_image = results[0].plot()
        st.image(result_image, caption="Detection Result", use_column_width=True)

        # List the detections
        with st.expander(f"Detections in {uploaded_file.name}"):
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"🔹 **{model.names[cls_id]}** with **{conf*100:.2f}%** confidence.")

        # Download result button
        result_pil = Image.fromarray(result_image[..., ::-1])  # Convert BGR to RGB
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Result Image",
            data=byte_im,
            file_name=f"detection_{uploaded_file.name}",
            mime="image/png"
        )

# --- Footer
st.markdown("---")
st.caption("Made with ❤️ by Ahmed Awan - Glasses Detection Project 🚀")
