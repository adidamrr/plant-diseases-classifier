import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.train import run_mynn, run_resnet18
from src.utils import device


def evaluate(model, dataloader, current_device, class_names=None):
    preds = []
    y_true = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(current_device)
        y_batch = y_batch.to(current_device)
        with torch.no_grad():
            pred = torch.argmax(model(x_batch), dim=1)
            preds.extend(pred.cpu())
            y_true.extend(y_batch.cpu())

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)

    if class_names is not None:
        report = classification_report(y_true, preds, target_names=class_names)
    else:
        report = classification_report(y_true, preds)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_confusion_matrix(evaluate_result, class_names):
    cm = evaluate_result["confusion_matrix"].astype(float)
    np.fill_diagonal(cm, 0)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    short_names = [name.replace("___", " | ").replace("_", " ") for name in class_names]

    plt.figure(figsize=(12, 12))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        square=True,
        cbar=True,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mynn", "resnet18"], default="resnet18")
    args = parser.parse_args()

    if args.model == "mynn":
        model, _, train_set, val_set, test_set, train_loader, val_loader, test_loader = run_mynn()
        class_names = getattr(test_set, "classes", None)
        evaluate_result = evaluate(model, test_loader, device, class_names)
        print(evaluate_result["classification_report"])
        if class_names is not None:
            plot_confusion_matrix(evaluate_result, val_set.classes)
        return

    model, _, _, train_set, val_set, test_set, train_loader, val_loader, test_loader = run_resnet18()
    evaluate_result = evaluate(model, test_loader, device, val_set.classes)
    print(evaluate_result["classification_report"])
    plot_confusion_matrix(evaluate_result, val_set.classes)


if __name__ == "__main__":
    main()

