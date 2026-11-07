import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(9, 2.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis("off")

boxes = [
    ("Input\n64x64", 0.2),
    ("Patch\nEmbed\n8x8", 1.9),
    ("+CLS\n+Pos\nEmbed", 3.6),
    ("Transformer\nBlock x4\n(Self-Attn+MLP)", 5.3),
    ("LayerNorm", 7.4),
    ("Linear\nHead", 8.7),
]

for label, x in boxes:
    rect = patches.FancyBboxPatch((x, 0.5), 1.4, 1.0, boxstyle="round,pad=0.05",
                                   linewidth=1.2, edgecolor="#2563eb", facecolor="#eff6ff")
    ax.add_patch(rect)
    ax.text(x + 0.7, 1.0, label, ha="center", va="center", fontsize=8)

for i in range(len(boxes) - 1):
    x1 = boxes[i][1] + 1.4
    x2 = boxes[i + 1][1]
    ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                arrowprops=dict(arrowstyle="->", color="#333"))

plt.tight_layout()
plt.savefig("reports/vit_diagram.png", dpi=150)
plt.close()
print("saved diagram")
