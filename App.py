import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

# Configuration
MODEL_URL = "https://github.com/vmalve/PCOSPredict/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
st.set_page_config(page_title="PCOS Predictor", page_icon="🧬")

# Load model
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Page UI
st.title("🧬 PCOS Ultrasound Analyzer")
st.markdown("Upload an **ultrasound image** to detect signs of **Polycystic Ovary Syndrome (PCOS)** using AI.")

# Hide the default Streamlit uploader (both drag-and-drop and file button)
st.markdown("""
    <style>
    .stFileUploader > div:first-child {
        display: none; /* Hide the default drag and drop and file button */
    }
    .stFileUploader {
        visibility: hidden; /* Hide the entire default file uploader */
    }
    </style>
""", unsafe_allow_html=True)

# Custom HTML and JS to trigger file upload
st.markdown("""
    <label for="real-file" style="background-color:#8e24aa;padding:12px 30px;color:white;border-radius:10px;cursor:pointer;font-weight:bold;">
        🖼️ Choose Image
    </label>
    <input type="file" id="real-file" accept=".jpg,.jpeg,.png" style="display:none;" onchange="document.querySelector('input[type=file]').dispatchEvent(new Event('change'));">
""", unsafe_allow_html=True)

# Trigger file upload via Streamlit’s file uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="real_uploader")

# Process the image if uploaded
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Image processing and prediction
        with st.spinner("🔍 Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

        # Show result
        st.success(f"🧠 **Prediction:** {prediction}")
        st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

    except Exception:
        st.error("⚠️ Invalid image file. Please try again.")
