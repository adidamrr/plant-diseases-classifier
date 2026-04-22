import argparse
import io

import torch
from PIL import Image

from src.dataset import tfms_resnet
from src.model import load_resnet18_for_inference
from src.utils import device, format_class_name, load_idx_to_class


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
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

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
