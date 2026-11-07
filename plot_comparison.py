import json
import matplotlib.pyplot as plt

with open("reports/baseline_metrics.json") as f:
    baseline = json.load(f)
with open("reports/vit_metrics.json") as f:
    vit = json.load(f)

all_results = {**baseline, **vit}
names = list(all_results.keys())
accs = [all_results[n]["accuracy"] for n in names]
f1s = [all_results[n]["f1"] for n in names]

x = range(len(names))
plt.figure(figsize=(8, 5))
plt.bar([i - 0.15 for i in x], accs, width=0.3, label="accuracy")
plt.bar([i + 0.15 for i in x], f1s, width=0.3, label="macro F1")
plt.xticks(list(x), names, rotation=20)
plt.ylim(0, 1.05)
plt.ylabel("score")
plt.title("A1 baselines vs A2 SOTA (ViT) comparison")
plt.legend()
plt.tight_layout()
plt.savefig("reports/metrics_comparison_a2.png")
plt.close()
print("saved comparison chart")
