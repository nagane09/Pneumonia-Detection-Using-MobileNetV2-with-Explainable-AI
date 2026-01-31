from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model --------
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

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(x).item()

            if prob > 0.5:
                prediction = "Pneumonia Detected"
                confidence = prob
            else:
                prediction = "Normal"
                confidence = 1 - prob

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
