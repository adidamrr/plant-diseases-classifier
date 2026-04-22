# Plant Disease Classifier

Проект перенесен из ноутбука в модульную структуру без изменения основной логики экспериментов. Поверх этого добавлены FastAPI-сервис и Streamlit UI для локального инференса.

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
├── requirements.txt
└── .gitignore
```

## Что внутри

- `notebooks/experiments.ipynb` содержит исходные эксперименты.
- `src/` содержит перенос кода обучения, оценки, предсказания и подготовки данных.
- `app/` содержит FastAPI-сервис и Streamlit UI для инференса.
- Текущий рабочий вариант основан на `ResNet18` с fine-tuning последних блоков `layer4` и `fc`.
- Для быстрой локальной проверки обучение настроено на `1` эпоху и `num_workers=0`.
- Для ускорения локальной проверки датасет в коде ограничен подмножествами: `train=1000`, `val=200`, `test=200`.
- Сервис ожидает артефакты `artifacts/best_model.pth` и `artifacts/idx_to_class.json`.

## Установка

```bash
pip install -r requirements.txt
```

## Обучение

```bash
python -m src.train
```

## Оценка

```bash
python -m src.evaluate
```

## Предсказание

```bash
python -m src.predict --image /path/to/image.jpg
```

## API

```bash
uvicorn app.api:app --reload
```

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Локальный запуск

1. Создайте и активируйте виртуальное окружение.
2. Установите зависимости из `requirements.txt`.
3. Поместите обученные веса в `artifacts/best_model.pth`.
4. Убедитесь, что `artifacts/idx_to_class.json` содержит mapping индекса в класс.
5. Запустите API:

```bash
uvicorn app.api:app --reload
```

6. Откройте проверку:

- `GET /health`
- `POST /predict`

7. Запустите UI:

```bash
streamlit run app/streamlit_app.py
```
