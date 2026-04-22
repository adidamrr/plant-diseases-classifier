import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T

from src.utils import TRAIN_PATH, VALID_PATH

import numpy as np
import matplotlib.pyplot as plt

tfms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

tfms_resnet_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tfms_resnet = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def show_random_samples():
    train_set = datasets.ImageFolder(TRAIN_PATH)

    cols = 8
    rows = 2
    fig = plt.figure(figsize=(2 * cols, 2.5 * rows))
    for i in range(cols):
        for j in range(rows):
            random_index = np.random.randint(0, len(train_set))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(train_set[random_index][0])
            ax.set_xlabel(train_set[random_index][1])
    plt.show()


def build_basic_dataloaders():
    train_set = datasets.ImageFolder(TRAIN_PATH, transform=tfms)
    val_set = datasets.ImageFolder(VALID_PATH, transform=tfms)

    n = len(train_set)
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_set, test_set = random_split(
        train_set,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=2)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def build_resnet_dataloaders():
    train_set = datasets.ImageFolder(TRAIN_PATH, transform=tfms_resnet_train)
    val_set = datasets.ImageFolder(VALID_PATH, transform=tfms_resnet)

    n = len(train_set)
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_set, test_set = random_split(
        train_set,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=2)
    test_loader = DataLoader(train_set, batch_size=1024, shuffle=False, num_workers=2)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader
