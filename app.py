import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown(
    "<h1 style='text-align: center;'>🫁 Pneumonia Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Chest X-ray classification using MobileNetV2"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

with st.expander("ℹ️ About this app", expanded=True):
    st.markdown("""
    **What this app does**
    - Uses **MobileNetV2**, a lightweight CNN
    - Classifies chest X-rays as **Normal** or **Pneumonia**
    - Displays a **confidence score**

    **How to use**
    1. Upload a chest X-ray image (JPG / PNG)
    2. Wait a second (yes, AI needs to think)
    3. View prediction and confidence
    """)

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(
        torch.load("mobilenetv2_pneumonia.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

st.subheader("📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    with st.spinner("Analyzing X-ray..."):
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).item()

    st.divider()

    if prob > 0.5:
        st.error(
            f"🚨 **Pneumonia Detected**\n\n"
            f"Confidence: **{prob:.3f}**"
        )
    else:
        st.success(
            f"✅ **Normal**\n\n"
            f"Confidence: **{1 - prob:.3f}**"
        )
