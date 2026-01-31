from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import uuid

# ---------------- Flask Setup ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model Setup ----------------
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

# ---------------- Transforms ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- Helper Functions ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and allowed_file(file.filename):
            # Save file with unique name to avoid overwrites
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(image_path)

            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                prob = model(x).item()

            if prob > 0.5:
                prediction = "Pneumonia Detected"
                confidence = prob
            else:
                prediction = "Normal"
                confidence = 1 - prob
        else:
            error = "Invalid file type. Please upload PNG, JPG, or JPEG."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )

# ---------------- Run App ----------------
if __name__ == "__main__":
    # 0.0.0.0 for cloud hosting; remove debug=True for production
    app.run(host="0.0.0.0", port=10000, debug=True)
