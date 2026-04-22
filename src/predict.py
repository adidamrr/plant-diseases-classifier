import argparse

import torch
from PIL import Image

from src.dataset import build_resnet_dataloaders, tfms, tfms_resnet
from src.train import run_mynn, run_resnet18
from src.utils import device


def predict_one_image(model, path, transform, classes, top_k, current_device):
    model.eval()
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(current_device)
    with torch.no_grad():
        pred = model(x)
        probs = torch.softmax(pred, dim=1)
        topprobs, topinds = torch.topk(probs, k=top_k, dim=1)
    topprobs = topprobs[0]
    topinds = topinds[0]

    res = []
    for p, i in zip(topprobs, topinds):
        res.append({
            "proba": p.item(),
            "class": classes[i],
        })
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=["mynn", "resnet18"], default="resnet18")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    _, val_set, _, _, _, _ = build_resnet_dataloaders()

    if args.model == "mynn":
        model, _, *_ = run_mynn()
        print(predict_one_image(model, args.image, tfms, val_set.classes, args.top_k, device))
        return

    model, *_ = run_resnet18()
    print(predict_one_image(model, args.image, tfms_resnet, val_set.classes, args.top_k, device))


if __name__ == "__main__":
    main()

