import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from vit_model import ViTClassifier
from evaluate import evaluate_predictions

DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def mixup_batch(images, labels, num_classes, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[perm]
    onehot = torch.zeros(labels.size(0), num_classes)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    onehot_mixed = lam * onehot + (1 - lam) * onehot[perm]
    return mixed, onehot_mixed


def soft_ce(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def train(use_mixup, tag):
    device = torch.device("cpu")

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    with open(f"{DATA_DIR}/classes.json", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = ViTClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup:
                mixed_images, soft_labels = mixup_batch(images, labels, num_classes)
                logits = model(mixed_images)
                loss = soft_ce(logits, soft_labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"[{tag}] epoch {epoch}/{EPOCHS}, loss: {avg_loss:.4f}")

    model.eval()
    start = time.time()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        preds = model(test_tensor).argmax(dim=1).numpy()
    latency_ms = (time.time() - start) / len(X_test) * 1000

    metrics = evaluate_predictions(y_test, preds, class_names, model_name=f"vit_{tag}")
    metrics["avg_latency_ms"] = latency_ms

    torch.save(model.state_dict(), f"{MODELS_DIR}/vit_{tag}.pt")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title(f"ViT training curve ({tag})")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/training_curve_{tag}.png")
    plt.close()

    return metrics


def main():
    results = {}
    results["vit_no_mixup"] = train(use_mixup=False, tag="no_mixup")
    results["vit_mixup"] = train(use_mixup=True, tag="mixup")

    with open(f"{REPORTS_DIR}/vit_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nablation results (with vs without mixup):")
    for name, m in results.items():
        print(name, m)


if __name__ == "__main__":
    main()
