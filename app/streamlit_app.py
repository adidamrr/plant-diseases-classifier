import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import format_class_name


API_URL = "http://127.0.0.1:8000/predict"


def build_bar_chart(predictions):
    labels = [format_class_name(item["class_name"]) for item in predictions]
    probabilities = [item["probability"] * 100 for item in predictions]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(labels, probabilities, color=["#2f7d32", "#7cb342", "#c0ca33"])
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    for index, value in enumerate(probabilities):
        ax.text(value + 1, index, f"{value:.1f}%", va="center")
    plt.tight_layout()
    return fig


def request_prediction(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    files = {
        "file": (uploaded_file.name, image_bytes, uploaded_file.type),
    }
    response = requests.post(API_URL, files=files, timeout=60)
    response.raise_for_status()
    return response.json(), image_bytes


st.set_page_config(page_title="Классификатор болезней растений", layout="wide")
st.title("Классификатор болезней растений")
st.write("Загрузите изображение листа, чтобы получить топ-3 предсказания модели и вероятности.")

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.image(image, width=600)

    with right_col:
        try:
            prediction_response, image_bytes = request_prediction(uploaded_file)
            top_predictions = prediction_response["top_3"]
            best_prediction = top_predictions[0]

            st.subheader("Результат")
            st.markdown(
                f"**Предсказанное заболевание:** {format_class_name(prediction_response['predicted_class'])}"
            )
            st.markdown(f"**Уверенность модели:** {prediction_response['confidence'] * 100:.1f}%")

            st.subheader("Топ-3 предсказания")
            for item in top_predictions:
                st.write(
                    f"{format_class_name(item['class_name'])} - {item['probability'] * 100:.1f}%"
                )

            st.pyplot(build_bar_chart(top_predictions))
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Ошибка API: {detail}")
        except requests.RequestException:
            st.error("Не удалось подключиться к API. Сначала запустите FastAPI через uvicorn app.api:app --reload.")
