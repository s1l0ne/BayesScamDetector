import os

from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, Engine
import logging as log
import joblib
import json

from evaluate import evaluate_models


BASE_DIR = Path(__file__).resolve().parent.parent
TABLE_NAME = 'scam'
RESULTS_FILE = 'train_results.txt'
MODELS_DIR = BASE_DIR / 'models/'

load_dotenv(BASE_DIR / '.env')

(BASE_DIR / 'logs').mkdir(exist_ok=True)
log.basicConfig(
    level=log.INFO,
    filename=BASE_DIR / 'logs' / 'training.log',
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)


def create_metadata_json(data: pd.DataFrame, path: str | Path = MODELS_DIR) -> None:
    metadata = {}

    for _, row in data.iterrows():
        metadata[row['name']] = {
            'file': f'{row['name']}.joblib',
            'metrics': {
                'accuracy': row['accuracy'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
        }

    MODELS_DIR.mkdir(exist_ok=True)
    with open(Path(path) / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def get_experiments() -> list[dict]:
    return [
        {
            'name': 'count_multinomialNB',
            'vectorizer': CountVectorizer(ngram_range=(1, 2), min_df=2),
            'classifier': MultinomialNB()
        },
        {
            'name': 'count_bernoulliNB',
            'vectorizer': CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True),
            'classifier': BernoulliNB()
        },
        {
            'name': 'count_complementNB',
            'vectorizer': CountVectorizer(ngram_range=(1, 2), min_df=2),
            'classifier': ComplementNB()
        },

        {
            'name': 'tfidf_multinomialNB',
            'vectorizer': TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            'classifier': MultinomialNB()
        },
        {
            'name': 'tfidf_bernoulliNB',
            'vectorizer': TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            'classifier': BernoulliNB()
        },
        {
            'name': 'tfidf_complementNB',
            'vectorizer': TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            'classifier': ComplementNB()
        }
    ]


def load_dataset(engine: Engine) -> pd.DataFrame:
    """Загружает датасет из БД и возвращает DataFrame из него"""
    return pd.read_sql(
        sql=f'SELECT * FROM {TABLE_NAME}',
        con=engine
    )


def train_models(experiments: list[dict], X_train: pd.Series, y_train: pd.Series) -> list[dict]:
    """Строит модели перебором функций векторизации и ML-алгоритмов"""
    models = []

    for experiment in experiments:
        log.info(f'Обучение {experiment['name']}')
        model = Pipeline([
            ('vectorizer', experiment['vectorizer']),
            ('classifier', experiment['classifier']),
        ])

        model.fit(X_train, y_train)
        models.append(
            {
                'name': experiment['name'],
                'model': model
            }
        )
        log.info(f'{experiment['name']} обучен')

    return models


if __name__ == '__main__':
    log.info('Подключение к БД')
    engine = create_engine(os.environ['DB_URL'])
    log.info('Подключение установлено')

    log.info('Загрузка датасета из БД')
    df = load_dataset(engine)
    log.info('Датасет загружен')

    log.info('Разделение датасета на train/test')
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    log.info('Датасет разделён')

    log.info('Обучение моделей')
    experiments = get_experiments()
    models = train_models(experiments, X_train, y_train)
    log.info('Модели обучены')

    log.info('Оценка моделей')
    results = evaluate_models(models, X_test, y_test).sort_values(by='F1', ascending=False)
    log.info('Оценка моделей завершена')

    log.info(f'Запись результатов эксперимента в файл {RESULTS_FILE}')
    with open(BASE_DIR / RESULTS_FILE, 'w', encoding='utf-8') as f:
        print(results.to_string(index=False), file=f)
    log.info(f'Результаты записаны')

    log.info(f'Сохранение моделей в {MODELS_DIR}')
    for item in models:
        log.info(f'Сохранение {item['name']}')
        joblib.dump(
            item['model'],
            MODELS_DIR / f'{item['name']}.joblib'
        )
        log.info(f'{item['name']} сохранён')
    log.info(f'Модели сохранены')

    log.info(f'Сохранение метаданных о моделях в metadata.json')
    create_metadata_json(results)
    log.info(f'metadata.json сохранён')