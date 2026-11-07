import time
import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import app as a

with open("demo_sample.png", "rb") as f:
    file_bytes = f.read()

img_arr, tensor = a.preprocess(file_bytes)
start = time.time()
pred, confidence, heatmap = a.vit_explain(a.MODEL, tensor)
latency_ms = (time.time() - start) * 1000
label = a.CLASS_NAMES[pred]

fig = plt.figure(figsize=(6, 5))
fig.patch.set_facecolor("white")

ax_text = fig.add_axes([0.05, 0.55, 0.9, 0.4])
ax_text.axis("off")
ax_text.text(0, 0.9, "Handwritten Digit & Symbol Recognizer", fontsize=14, weight="bold")
ax_text.text(0, 0.72, "Assignment 2 - ViT (SOTA) + Attention Rollout Explainability", fontsize=9, color="#555")
ax_text.text(0, 0.45, f"Prediction: {label}", fontsize=13, weight="bold")
ax_text.text(0, 0.28, f"Confidence: {confidence*100:.1f}%", fontsize=10)
ax_text.text(0, 0.13, f"Latency: {latency_ms:.2f} ms", fontsize=10)
ax_text.text(0, -0.02, f"Device: {platform.system()} / CPU", fontsize=10)

ax_img = fig.add_axes([0.3, 0.02, 0.4, 0.45])
ax_img.imshow(img_arr, cmap="gray")
ax_img.imshow(heatmap, cmap="jet", alpha=0.5)
ax_img.axis("off")
ax_img.set_title("Attention rollout heatmap", fontsize=9)

plt.savefig("reports/app_screenshot.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved", label, confidence, latency_ms)
