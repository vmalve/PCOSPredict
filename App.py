import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

# Configuration
model_url = "https://github.com/vmalve/PCOSPredict/releases/download/v1.0.0/PCOS_resnet18_model.pth"
model_path = "PCOS_resnet18_model.pth"

# Download model if not exists
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        with open(model_path, "wb") as f:
            f.write(requests.get(model_url).content)

# Constants
CLASS_NAMES = ['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🔍 PCOS Prediction from Ultrasound Image")
st.markdown("Upload an ultrasound image to predict **PCOS** presence.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
            prediction = CLASS_NAMES[predicted.item()]

        st.success(f"✅ **Prediction: {prediction}**")
        st.info(f"🧠 Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
