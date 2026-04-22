from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: list[TopPrediction]
