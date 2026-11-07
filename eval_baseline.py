import json
import time
import numpy as np
import torch
import joblib

from cnn_model import SimpleCNN
from evaluate import evaluate_predictions

DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def main():
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    with open(f"{DATA_DIR}/classes.json", encoding="utf-8") as f:
        class_names = json.load(f)

    results = {}

    baseline = joblib.load(f"{MODELS_DIR}/baseline_model.pkl")
    X_test_flat = X_test.reshape(len(X_test), -1)
    start = time.time()
    preds = baseline.predict(X_test_flat)
    latency = (time.time() - start) / len(X_test) * 1000
    metrics = evaluate_predictions(y_test, preds, class_names, model_name="a1_baseline_logreg")
    metrics["avg_latency_ms"] = latency
    results["a1_baseline_logreg"] = metrics

    cnn = SimpleCNN(num_classes=len(class_names))
    cnn.load_state_dict(torch.load(f"{MODELS_DIR}/cnn_model.pt", map_location="cpu"))
    cnn.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    start = time.time()
    with torch.no_grad():
        preds = cnn(test_tensor).argmax(dim=1).numpy()
    latency = (time.time() - start) / len(X_test) * 1000
    metrics = evaluate_predictions(y_test, preds, class_names, model_name="a1_cnn")
    metrics["avg_latency_ms"] = latency
    results["a1_cnn"] = metrics

    with open(f"{REPORTS_DIR}/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
