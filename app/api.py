from fastapi import FastAPI

from app.schemas import PredictionItem, PredictionRequest, PredictionResponse
from src.dataset import build_resnet_dataloaders, tfms_resnet
from src.predict import predict_one_image
from src.train import run_resnet18
from src.utils import device


app = FastAPI(title="Plant Disease Classifier API")


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    _, val_set, _, _, _, _ = build_resnet_dataloaders()
    model, *_ = run_resnet18()
    predictions = predict_one_image(
        model=model,
        path=request.image_path,
        transform=tfms_resnet,
        classes=val_set.classes,
        top_k=request.top_k,
        current_device=device,
    )
    return PredictionResponse(
        predictions=[
            PredictionItem(proba=item["proba"], class_name=item["class"])
            for item in predictions
        ]
    )

