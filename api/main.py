from fastapi import FastAPI
from .app.models import PredictionRequest,PredictionResponse
from .app.views import get_prediction

app = FastAPI(docs_url='/')

@app.post('/v1/prediction', response_model=PredictionResponse)
def make_model_prediction(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(worldwide_gross=get_prediction(request))



