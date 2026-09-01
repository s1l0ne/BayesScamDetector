from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import pandas as pd
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


def evaluate_models(models: list[dict], X_test: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Возвращает DataFrame с оценкой переданных моделей по F1, precision, recall и accuracy"""
    results = []

    for item in models:
        log.info(f'Оценка {item['name']}')

        y_pred = item['model'].predict(X_test)

        results.append({
            'name': item['name'],
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        })

        log.info(f'{item['name']} оценена')

    return pd.DataFrame(results)
