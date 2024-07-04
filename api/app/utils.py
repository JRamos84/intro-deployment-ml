from joblib import load
from sklearn.pipeline import Pipeline
from pandas import DataFrame
import os
from pydantic import BaseModel

def get_model() -> Pipeline:
    model_path = os.environ.get('MODEL_PATH', 'model/model.pkl')
    with open(model_path, 'rb') as model_file:
        model = load(model_file)
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = class_model.model_dump()  # Usar model_dump en lugar de dict
    data_frame = DataFrame([transition_dictionary])
    
    # Añadir columnas faltantes con valores por defecto
    missing_columns = {'Unnamed: 0': 0, 'release_date': 'N/A', 'movie_name': 'N/A', 
                       'domestic_gross': 0.0, 'movie_link': 'N/A'}
    
    for column, default_value in missing_columns.items():
        if column not in data_frame:
            data_frame[column] = default_value
    
    return data_frame
