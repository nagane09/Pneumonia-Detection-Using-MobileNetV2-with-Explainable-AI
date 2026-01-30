import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Pneumonia Detection using MobileNetV2")
st.markdown("""
**What this app does:**  
- Uses **MobileNetV2**, a lightweight convolutional neural network, to classify chest X-rays as **Normal** or **Pneumonia**.  
- Provides **confidence score** for each prediction.  

**How to use:**  
1. Upload a chest X-ray image (JPEG/PNG).  
2. Wait for the prediction to appear below the image.  
3. The app will show whether the image is classified as **Normal** or **Pneumonia**, along with a confidence score.  


""")


model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("mobilenetv2_pneumonia.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", width=300)

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(x).item()

    if prob > 0.5:
        st.error(f"Pneumonia Detected (Confidence: {prob:.3f})")
    else:
        st.success(f"Normal (Confidence: {1-prob:.3f})")
