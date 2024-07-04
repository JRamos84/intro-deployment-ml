from dvc import api
import pandas as pd
import sys
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Fetching data..')

try:
    # Lectura de datos desde DVC
    with api.open('dataset/movies.csv', remote='dataset-tracker') as fd:
        movie_data = pd.read_csv(fd)
    logger.info('Loaded movie data')

    with api.open('dataset/finantials.csv', remote='dataset-tracker') as fd:
        fin_data = pd.read_csv(fd)
    logger.info('Loaded financial data')

    with api.open('dataset/opening_gross.csv', remote='dataset-tracker') as fd:
        open_data = pd.read_csv(fd)
    logger.info('Loaded opening data')
except Exception as e:
    logger.error(f'Error fetching data: {e}')
    sys.exit(1)

try:
    # Seleccionar columnas numéricas
    numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
    numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
    movie_data = movie_data[numeric_columns + ['movie_title']]

    # Fusionar DataFrames
    fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
    full_movie_data = pd.merge(open_data, fin_movie_data, on='movie_title', how='left')

    # Eliminar columnas innecesarias
    full_movie_data = full_movie_data.drop(['gross', 'movie_title'], axis=1)

    # Guardar los datos finales en un archivo CSV
    full_movie_data.to_csv('dataset/full_data.csv', index=False)
    logger.info('Data fetched and prepared...')
except Exception as e:
    logger.error(f'Error processing data: {e}')
    sys.exit(1)
