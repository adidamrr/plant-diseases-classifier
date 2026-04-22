import json
import os
from pathlib import Path

import kagglehub
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
IDX_TO_CLASS_PATH = ARTIFACTS_DIR / "idx_to_class.json"

DEFAULT_TRAIN_PATH = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
DEFAULT_VALID_PATH = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

TRAIN_PATH = os.getenv("TRAIN_PATH", DEFAULT_TRAIN_PATH)
VALID_PATH = os.getenv("VALID_PATH", DEFAULT_VALID_PATH)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


def download_dataset():
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print("Path to dataset files:", path)
    return path


def load_idx_to_class(path=IDX_TO_CLASS_PATH):
    with open(path, "r", encoding="utf-8") as file:
        raw_mapping = json.load(file)
    return {int(idx): class_name for idx, class_name in raw_mapping.items()}


def save_idx_to_class(idx_to_class, path=IDX_TO_CLASS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_mapping = {str(idx): class_name for idx, class_name in idx_to_class.items()}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable_mapping, file, indent=2, ensure_ascii=True)


def save_model_state(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def format_class_name(class_name):
    return class_name.replace("___", ": ").replace("_", " ")


def is_allowed_image(filename, content_type):
    suffix = Path(filename or "").suffix.lower()
    return suffix in ALLOWED_EXTENSIONS and content_type in ALLOWED_CONTENT_TYPES
