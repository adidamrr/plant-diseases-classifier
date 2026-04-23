# Plant Disease Classifier

Классификатор болезней растений на PyTorch с одной архитектурой: `ResNet18` и fine-tuning слоёв `layer4` и `fc`.

Проект включает:

- обучение модели;
- сохранение артефактов в `artifacts/`;
- FastAPI-сервис для инференса;
- Streamlit-интерфейс для загрузки изображения и просмотра top-3 предсказаний.

## Структура

```text
.
├── app/
│   ├── api.py
│   ├── schemas.py
│   └── streamlit_app.py
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── artifacts/
│   ├── best_model.pth
│   └── idx_to_class.json
├── notebooks/
│   └── experiments.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

## Данные

Для обучения нужно скачать датасет с Kaggle:

[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

После скачивания используйте директории `train` и `valid` из датасета и передайте их через переменные окружения:

```bash
export TRAIN_PATH="/полный/путь/к/train"
export VALID_PATH="/полный/путь/к/valid"
```

## Готовая модель

Если вы не хотите обучать модель локально, можно использовать уже готовые артефакты:

- `artifacts/best_model.pth`
- `artifacts/idx_to_class.json`

Если эти файлы уже лежат в папке `artifacts/`, сервис и интерфейс можно запускать сразу без шага обучения.

## Локальный запуск

### 1. Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python3 -m src.train
```

После обучения автоматически сохраняются:

- `artifacts/best_model.pth`
- `artifacts/idx_to_class.json`

### 3. Запуск API

```bash
uvicorn app.api:app --reload
```

Проверка:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/predict`

Пример запроса:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/полный/путь/к/image.jpg"
```

### 4. Запуск Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Интерфейс будет доступен по адресу:

- `http://127.0.0.1:8501`

## Docker

### Запуск через Docker Compose

Если у вас уже есть готовая модель в `artifacts/`, можно поднять сервис и UI через Docker:

```bash
docker compose up --build
```

После запуска:

- API: `http://127.0.0.1:8000`
- Streamlit UI: `http://127.0.0.1:8501`

### Отдельный запуск только API

```bash
docker build -t plant-disease-classifier .
docker run --rm -p 8000:8000 plant-disease-classifier
```

## Основные команды

Обучение:

```bash
python3 -m src.train
```

Оценка:

```bash
python3 -m src.evaluate
```

Предсказание из CLI:

```bash
python3 -m src.predict --image /path/to/image.jpg
```

## Примечание

Для инференса проект ожидает, что `artifacts/best_model.pth` и `artifacts/idx_to_class.json` соответствуют друг другу и были получены из одной и той же обученной модели.
