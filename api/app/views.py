from .models import PredictionRequest
from .utils import get_model, transform_to_dataframe

model = get_model()  # Llama a la función get_model para obtener la instancia del modelo

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict)[0]
    return max(0, prediction)
