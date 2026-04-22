# Plant Disease Classifier

Проект перенесен из ноутбука в модульную структуру без изменения основной логики экспериментов.

## Структура

```text
.
├── app/
│   ├── api.py
│   └── schemas.py
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
- `app/` содержит минимальный FastAPI-слой для инференса.
- В ноутбуке сохранены обе ветки экспериментов: `MyNN` и `ResNet18`.
- Текущий рабочий вариант основан на `ResNet18` с fine-tuning последних блоков `layer4` и `fc`.

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

