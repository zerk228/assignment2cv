import json
import matplotlib.pyplot as plt

with open("reports/vit_metrics.json") as f:
    pass

EPOCHS = 12
losses_no_mixup = None
losses_mixup = None

import re
log_path = "train_log.txt"
EPOCHS = 12

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, tag, title in zip(axes, ["no_mixup", "mixup"], ["without mixup", "with mixup"]):
    img = plt.imread(f"reports/training_curve_{tag}.png")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig("reports/training_curves_combined.png", dpi=150)
plt.close()
print("saved combined curves")
