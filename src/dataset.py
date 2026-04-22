import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision import transforms as T

from src.utils import TRAIN_PATH, VALID_PATH

FAST_TRAIN_SAMPLES = 1000
FAST_VAL_SAMPLES = 200
FAST_TEST_SAMPLES = 200

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

    train_limit = min(FAST_TRAIN_SAMPLES, len(train_set))
    val_limit = min(FAST_VAL_SAMPLES, len(val_set))
    test_limit = min(FAST_TEST_SAMPLES, len(test_set))

    train_set = Subset(train_set, range(train_limit))
    val_set = Subset(val_set, range(val_limit))
    test_set = Subset(test_set, range(test_limit))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader
