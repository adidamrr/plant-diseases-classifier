from PIL import UnidentifiedImageError
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import PredictionResponse, TopPrediction
from src.dataset import tfms_resnet
from src.model import load_resnet18_for_inference
from src.predict import load_image_from_bytes, predict_from_pil_image
from src.utils import IDX_TO_CLASS_PATH, MODEL_PATH, device, is_allowed_image, load_idx_to_class


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.idx_to_class = load_idx_to_class()
    app.state.transform = tfms_resnet
    app.state.model, app.state.model_loaded = load_resnet18_for_inference()
    app.state.model_path = str(MODEL_PATH)
    app.state.idx_to_class_path = str(IDX_TO_CLASS_PATH)
    yield


app = FastAPI(title="Plant Disease Classifier API", lifespan=lifespan)


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "idx_to_class_path": str(IDX_TO_CLASS_PATH),
        "model_loaded": app.state.model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not is_allowed_image(file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Only jpg, jpeg, and png files are supported.",
        )

    image_bytes = await file.read()
    try:
        image = load_image_from_bytes(image_bytes)
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from error

    predictions = predict_from_pil_image(
        model=app.state.model,
        image=image,
        transform=app.state.transform,
        idx_to_class=app.state.idx_to_class,
        top_k=3,
        current_device=device,
    )

    best_prediction = predictions[0]
    return PredictionResponse(
        predicted_class=best_prediction["class_name"],
        confidence=best_prediction["probability"],
        top_3=[
            TopPrediction(
                class_name=item["class_name"],
                probability=item["probability"],
            )
            for item in predictions
        ],
    )
