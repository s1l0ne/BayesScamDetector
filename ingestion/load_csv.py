from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
import logging as log
import shutil
from pathlib import Path


load_dotenv()

log.basicConfig(
    level=log.INFO,
    filename='logs/load.log',
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)


def get_table_name(csv_file: str | Path) -> str:
    table_name = Path(csv_file).stem.split('_')
    if len(table_name) > 1 and table_name[-1].isdigit():
        table_name.pop()
    return '_'.join(table_name)


def load_csv_to_db(db_url: str, path: str | Path = '../csv') -> None:
    """Загружает все CSV из path в БД по db_url и перемещает их в папку to_delete"""
    path = Path(path)

    to_delete_path = Path('../to_delete')
    to_delete_path.mkdir(exist_ok=True)

    engine = create_engine(db_url)

    for csv_file in path.glob('*.csv'):
        log.info(msg=f'Working with {csv_file.name}')

        try:
            df = pd.read_csv(csv_file)

            with engine.begin() as conn:
                df.to_sql(
                    get_table_name(csv_file),
                    conn,
                    if_exists='append',
                    index=False
                )

            log.info(msg=f'{csv_file.name} is uploaded! Moving file in {to_delete_path} dir...')

            shutil.move(csv_file, to_delete_path)

            log.info(msg=f'{csv_file.name} is moved')
        except Exception:
            log.exception(msg=f'Failed to process {csv_file.name}')

    engine.dispose()


if __name__ == '__main__':
    load_csv_to_db(db_url=os.environ['DB_URL'])
