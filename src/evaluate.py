import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.train import run_resnet18
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.parse_args()

    model, _, train_set, val_set, test_set, train_loader, val_loader, test_loader = run_resnet18()
    evaluate_result = evaluate(model, test_loader, device, val_set.classes)
    print(evaluate_result["classification_report"])


if __name__ == "__main__":
    main()
