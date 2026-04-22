import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import build_resnet_dataloaders
from src.model import (
    build_resnet18,
    freeze_all_layers,
    unfreeze_last_block_and_fc,
)
from src.utils import (
    IDX_TO_CLASS_PATH,
    MODEL_PATH,
    device,
    get_dataset_classes,
    save_idx_to_class,
    save_model_state,
)


def train_epoch(model, opt, lossfunc, train_loader, current_device):
    model.train()
    loss_list = []
    metric_list = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(current_device)
        y_batch = y_batch.to(current_device)
        opt.zero_grad()
        pred = model(x_batch)
        loss = lossfunc(pred, y_batch)
        loss.backward()
        opt.step()

        loss_list.append(loss.item())
        metric_value = (torch.argmax(pred, dim=1) == y_batch).float().mean()
        metric_list.append(metric_value.item())
    return loss_list, metric_list


def validation(model, lossfunc, val_loader, current_device):
    model.eval()
    loss_list = []
    metric_list = []

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(current_device)
        y_batch = y_batch.to(current_device)
        with torch.no_grad():
            pred = model(x_batch)
            loss = lossfunc(pred, y_batch)
            metric_value = (torch.argmax(pred, dim=1) == y_batch).float().mean()
        loss_list.append(loss.item())
        metric_list.append(metric_value.item())

    return loss_list, metric_list


def train(epoch_num, model, opt, lossfunc, train_loader, val_loader, current_device, scheduler=None):
    train_loss_list, val_loss_list = [], []
    train_metric_list, val_metric_list = [], []
    for epoch in range(epoch_num):
        loss_train, metric_train = train_epoch(model, opt, lossfunc, train_loader, current_device)
        train_loss_list.append(np.mean(loss_train))
        train_metric_list.append(np.mean(metric_train))

        loss_val, metric_val = validation(model, lossfunc, val_loader, current_device)
        val_loss_list.append(np.mean(loss_val))
        val_metric_list.append(np.mean(metric_val))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(loss_val))
            else:
                scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epoch_num} | "
            f"train_loss={sum(loss_train) / len(loss_train):.4f} | "
            f"train_acc={sum(metric_train) / len(metric_train):.4f} | "
            f"val_loss={sum(loss_val) / len(loss_val):.4f} | "
            f"val_acc={sum(metric_val) / len(metric_val):.4f}"
        )

    return {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_metric": train_metric_list,
        "val_metric": val_metric_list,
    }


def save_training_artifacts(model, classes):
    idx_to_class = {idx: class_name for idx, class_name in enumerate(classes)}
    save_model_state(model)
    save_idx_to_class(idx_to_class)
    print(f"Saved model weights to {MODEL_PATH}")
    print(f"Saved idx_to_class mapping to {IDX_TO_CLASS_PATH}")


def run_resnet18():
    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_resnet_dataloaders()

    model = build_resnet18(38).to(device)

    freeze_all_layers(model)
    unfreeze_last_block_and_fc(model)

    lossfunc = nn.CrossEntropyLoss()
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)

    result = train(1, model, opt, lossfunc, train_loader, val_loader, device, scheduler=None)
    save_training_artifacts(model, get_dataset_classes(val_set))

    return model, result, train_set, val_set, test_set, train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.parse_args()
    run_resnet18()


if __name__ == "__main__":
    main()
