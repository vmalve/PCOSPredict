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

# Download model if needed
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# App UI
st.set_page_config(page_title="PCOS Predictor", page_icon="🧬", layout="centered")

st.title("🧬 PCOS Predictor")
st.markdown(
    """
    Upload an **ultrasound image** to detect signs of **Polycystic Ovary Syndrome (PCOS)** using a deep learning model trained on medical imaging data.
    """
)

st.divider()
uploaded_file = st.file_uploader("📤 Upload an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image..."):
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
            prediction = CLASS_NAMES[predicted.item()]

    st.success(f"🧠 **Prediction:** {prediction}")
    st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

else:
    st.warning("📎 Please upload a valid image file.")

st.markdown("---")
st.caption("Made with ❤️ using ResNet18 and Streamlit.")
