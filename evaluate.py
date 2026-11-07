import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

REPORTS_DIR = "reports"


def evaluate_predictions(y_true, y_pred, class_names, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nresults for model: {model_name}")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1-score:  {f1:.4f}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"confusion matrix - {model_name}")
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"matrix saved to {out_path}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
