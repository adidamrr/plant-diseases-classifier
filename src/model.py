import torch
import torch.nn as nn
from torchvision import models

from src.utils import MODEL_PATH, device


def build_resnet18(num_classes=38, weights=models.ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_block_and_fc(model):
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


def load_resnet18_for_inference(model_path=MODEL_PATH, num_classes=38, current_device=device):
    artifact_loaded = model_path.exists() and model_path.stat().st_size > 0
    model = build_resnet18(num_classes, weights=None)
    if artifact_loaded:
        state_dict = torch.load(model_path, map_location=current_device)
        model.load_state_dict(state_dict)
    model = model.to(current_device)
    model.eval()
    return model, artifact_loaded
