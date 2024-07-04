from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

def get_model() -> Pipeline:
    model_path = os.environ.get('MODEL_PATH', 'model/model.pkl')
    with open(model_path, 'rb') as model_file:
        model = load(BytesIO(model_file.read()))
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = class_model.dict()  # Corregido para usar class_model.dict()
    data_frame = DataFrame([transition_dictionary])  # Crear DataFrame con una lista que contiene el diccionario
    return data_frame
