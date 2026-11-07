import io
import json
import time
import platform
import base64

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from vit_model import ViTClassifier
from explain import vit_explain

app = Flask(__name__)

with open("data/classes.json", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

MODEL = ViTClassifier(num_classes=len(CLASS_NAMES))
MODEL.load_state_dict(torch.load("models/vit_mixup.pt", map_location="cpu"))
MODEL.eval()


def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return arr, tensor


def make_overlay(img_arr, heatmap):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_arr, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_prediction(file_bytes):
    img_arr, tensor = preprocess(file_bytes)
    start = time.time()
    pred, confidence, heatmap = vit_explain(MODEL, tensor)
    latency_ms = (time.time() - start) * 1000
    overlay_b64 = make_overlay(img_arr, heatmap)
    return {
        "label": CLASS_NAMES[pred],
        "confidence": f"{confidence * 100:.1f}%",
        "latency_ms": f"{latency_ms:.2f}",
        "device": f"{platform.system()} / CPU",
        "overlay": overlay_b64,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        result = run_prediction(file.read())
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
