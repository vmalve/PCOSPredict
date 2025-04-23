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

# Set page config
st.set_page_config(
    page_title="🧬 PCOS Image Predictor",
    page_icon="🩺",
    layout="centered",
)

# Custom styles
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        color: #6a1b9a;
        font-weight: 700;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #444;
    }
    .stButton>button {
        background-color: #8e24aa;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .card {
        background-color: #f3e5f5;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 14px;
        padding-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# UI Content
st.markdown('<div class="title">🧬 PCOS Ultrasound Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image to predict<b>Polycystic Ovary Syndrome (PCOS)</b> using AI.</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload an ultrasound image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        with st.spinner("🧠 Analyzing with AI..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                prediction = CLASS_NAMES[predicted.item()]

        st.success(f"✅ **Prediction:** `{prediction}`")
        st.info(f"📊 **Confidence Score:** `{confidence * 100:.2f}%`")
    else:
        st.warning("📎 Please upload a valid image file.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">💡 Built with ResNet18, PyTorch, and Streamlit · Made with ❤️ by the AI community</div>', unsafe_allow_html=True)
