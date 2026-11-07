# Assignment 2 — Handwritten Digits/Symbols: ViT SOTA Upgrade

Continuation of Assignment 1 (digits 0-9, plus, minus). Adds a Vision Transformer
(ViT) as the SOTA method, attention-rollout and Grad-CAM explainability, a mixup
ablation, and a Flask demo app.

## Setup

```
pip install -r requirements.txt
```

Data and A1 model weights are already copied into `data/` and `models/`
(`cnn_model.pt`, `baseline_model.pkl` from Assignment 1).

## Reproduce results

```
python eval_baseline.py       # evaluates A1 logistic regression + CNN on test set
python train_sota.py          # trains ViT with/without mixup, saves models + metrics
python nearest_neighbors_demo.py
python plot_comparison.py
```

All metrics are written to `reports/*.json`, plots to `reports/*.png`.

## Run the demo app

```
python app.py
```

Open http://localhost:5000, upload a digit/symbol image, get prediction +
attention-rollout heatmap + latency + device info.

## Files

- `cnn_model.py` — A1 CNN architecture (loaded for comparison + Grad-CAM)
- `vit_model.py` — ViT architecture (SOTA method) + attention rollout
- `train_sota.py` — trains ViT, with ablation (mixup on/off)
- `eval_baseline.py` — evaluates A1 baseline + CNN on the same test set
- `explain.py` — Grad-CAM, attention rollout, nearest-neighbor retrieval
- `nearest_neighbors_demo.py` — builds nearest-neighbor gallery figure
- `plot_comparison.py` — bar chart comparing all models
- `app.py` / `templates/` / `static/` — Flask demo app

## Ethics

Dataset is hand-drawn digits/symbols collected by the project author; no
personal or biometric data. The app displays a disclaimer stating predictions
come from a model trained on student-collected data and are for educational
use only.
