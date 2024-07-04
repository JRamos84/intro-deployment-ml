from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def update_model(model: Pipeline)-> None:
    dump(model, 'model/model.pkl')


import pandas as pd

def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    """
    Guarda un reporte simple de las métricas del modelo en un archivo de texto.

    Args:
    - train_score (float): Puntuación del modelo en el conjunto de entrenamiento.
    - test_score (float): Puntuación del modelo en el conjunto de prueba.
    - validation_score (float): Puntuación del modelo en el conjunto de validación.
    - model (Pipeline): El modelo entrenado.
    """
    with open('report.txt', 'w') as report_file:
        report_file.write('# Model Pipeline Description\n')
        
        for key, value in model.named_steps.items():
            report_file.write(f'### {key}: {value.__repr__()}\n')
        
        report_file.write(f'#### Train Score: {train_score}\n')
        report_file.write(f'#### Test Score: {test_score}\n')
        report_file.write(f'#### Validation Score: {validation_score}\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
    """
    Genera y guarda una gráfica que compara las predicciones del modelo con los valores reales.

    Args:
    - y_real (pd.Series): Valores reales del conjunto de prueba.
    - y_pred (pd.Series): Valores predichos por el modelo.
    """
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax=ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig('prediction_behavior.png')