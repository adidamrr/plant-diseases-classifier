from pydantic import BaseModel


class PredictionRequest(BaseModel):
    image_path: str
    top_k: int = 3


class PredictionItem(BaseModel):
    proba: float
    class_name: str


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]

