import torch.nn as nn
from torchvision import models


class MyNN(nn.Module):
    def __init__(self, n_output, chanels):
        super().__init__()
        self.chanels = chanels
        self.n_output = n_output

        self.conv = nn.Sequential(
            nn.Conv2d(self.chanels, 10, (5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 60, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(60, 80, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, self.n_output),
        )

    def forward(self, x):
        out = self.conv(x)
        out = nn.Flatten()(out)
        out = self.classifier(out)
        return out


def weith_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def build_resnet18(num_classes=38):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_fc_only(model):
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_last_block_and_fc(model):
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

