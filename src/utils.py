import kagglehub
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_PATH = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"


def download_dataset():
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print("Path to dataset files:", path)
    return path

