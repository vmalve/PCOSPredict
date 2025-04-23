import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
from streamlit.components.v1 import html

# --- App Config ---
st.set_page_config(
    page_title="PCOS Predictor",
    page_icon="🧬",
    layout="centered"
)

# --- Device Detection ---
html("""
<script>
    const width = window.innerWidth;
    const device = width < 768 ? "mobile" : "desktop";
    window.parent.postMessage(
        { isStreamlitMessage: true, type: "STREAMLIT:SET_COMPONENT_VALUE", value: device }, "*"
    );
</script>
""")

query = st.query_params
device = query.get("device", ["desktop"])[0]
if "device" not in st.session_state:
    st.session_state.device = device

# --- Sidebar ---
with st.sidebar:
    st.header("📚 About")
    st.write("AI-powered tool to analyze ultrasound images for **PCOS**.")
    st.markdown("[GitHub Repo](https://github.com/vmalve/PCOSPredict)")
    st.markdown("[Contact Developer](mailto:your@email.com)")

# --- Model Config ---
MODEL_URL = "https://github.com/vmalve/PCOSPredict/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['No PCOS', 'PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Download Model if Missing ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Header ---
st.title("🧬 PCOS Ultrasound Analyzer")
st.caption("Upload an ultrasound image to detect signs of Polycystic Ovary Syndrome (PCOS) using AI.")

# --- Upload Section ---
st.markdown(
    f"You're using a **{st.session_state.device}** device.",
    help="The app adjusts layout for mobile and desktop."
)
uploaded_file = st.file_uploader("📤 Upload ultrasound image", type=["jpg", "jpeg", "png"])

# --- Prediction ---
if uploaded_file:
    try:
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

    except Exception:
        st.error("⚠️ Invalid image file. Please upload a valid JPEG or PNG.")

# --- Footer ---
st.markdown(
    "<hr style='margin-top: 3em;'><center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)
