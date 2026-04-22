import argparse
import io

import torch
from PIL import Image

from src.dataset import tfms, tfms_resnet
from src.model import MyNN, load_resnet18_for_inference, weith_init
from src.utils import device, format_class_name, load_idx_to_class


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


def predict_from_pil_image(model, image, transform, idx_to_class, top_k, current_device):
    model.eval()
    x = transform(image.convert("RGB")).unsqueeze(0).to(current_device)
    with torch.no_grad():
        pred = model(x)
        probs = torch.softmax(pred, dim=1)
        topprobs, topinds = torch.topk(probs, k=top_k, dim=1)

    top_probabilities = topprobs[0].tolist()
    top_indices = topinds[0].tolist()

    predictions = []
    for probability, class_index in zip(top_probabilities, top_indices):
        predictions.append({
            "class_name": idx_to_class[class_index],
            "probability": probability,
            "display_name": format_class_name(idx_to_class[class_index]),
        })
    return predictions


def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=["mynn", "resnet18"], default="resnet18")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.model == "mynn":
        idx_to_class = load_idx_to_class()
        model = MyNN(38, 3).to(device)
        weith_init(model)
        classes = [idx_to_class[idx] for idx in range(len(idx_to_class))]
        print(predict_one_image(model, args.image, tfms, classes, args.top_k, device))
        return

    idx_to_class = load_idx_to_class()
    model, _ = load_resnet18_for_inference()
    predictions = predict_from_pil_image(
        model=model,
        image=Image.open(args.image),
        transform=tfms_resnet,
        idx_to_class=idx_to_class,
        top_k=args.top_k,
        current_device=device,
    )
    print(predictions)


if __name__ == "__main__":
    main()
