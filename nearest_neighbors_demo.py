import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from vit_model import ViTClassifier
from explain import nearest_neighbors

DATA_DIR = "data"
REPORTS_DIR = "reports"

with open(f"{DATA_DIR}/classes.json", encoding="utf-8") as f:
    class_names = json.load(f)

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

model = ViTClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("models/vit_mixup.pt", map_location="cpu"))
model.eval()


def embed(images):
    tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        x = model.patch_embed(tensor)
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + model.pos_embed
        for block in model.blocks:
            x, _ = block(x)
        x = model.norm(x)
    return x[:, 0].numpy()


embeddings = embed(X_test)
query_idx = 0
query_vec = embeddings[query_idx]

idx, labels, dists = nearest_neighbors(query_vec, embeddings[1:], y_test[1:], k=5)
idx = idx + 1

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
axes[0].imshow(X_test[query_idx], cmap="gray")
axes[0].set_title(f"query: {class_names[y_test[query_idx]]}")
axes[0].axis("off")

for i, (n_idx, dist) in enumerate(zip(idx, dists)):
    axes[i + 1].imshow(X_test[n_idx], cmap="gray")
    axes[i + 1].set_title(f"{class_names[y_test[n_idx]]}\n{dist:.2f}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/nearest_neighbors_gallery.png")
plt.close()
print("saved nearest neighbor gallery")
