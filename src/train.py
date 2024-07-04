from utils import update_model, save_simple_metrics_report, get_model_performance_test_set
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import logging
import sys
import numpy as np
import pandas as pd

# Configuración del registro
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

# Creación del logger
logger = logging.getLogger(__name__)

# Registro de información sobre la carga de datos
logger.info('Loading Data...')

# Carga de datos desde CSV
data = pd.read_csv('dataset/full_data.csv')

# Registro de información sobre la carga del modelo
logger.info('Loading model..')

# Separación de datos en entrenamiento y prueba
X = data.drop('worldwide_gross', axis=1)
y = data['worldwide_gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Definición de las columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Definición del pipeline con imputación y modelo de regresión
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('core_model', GradientBoostingRegressor())
])

# Definición de hiperparámetros a ajustar
param_tuning = {'core_model__n_estimators': range(300, 301)}

# Creación de GridSearchCV para ajuste de hiperparámetros
grid_search_cv = GridSearchCV(model, param_grid=param_tuning, scoring='r2', cv=5)

# Registro de información sobre el inicio de la búsqueda en cuadrícula
logger.info('Starting grid search..')

# Ajuste de GridSearchCV con datos de entrenamiento
grid_search_cv.fit(X_train, y_train)

# Registro de información sobre la validación cruzada
logger.info('Cross validating with model..')

# Validación cruzada con el mejor estimador encontrado por GridSearchCV
final_result = cross_validate(grid_search_cv.best_estimator_, X_train, y_train, return_train_score=True, cv=5)
train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])

assert train_score > 0.7
assert test_score > 0.65

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

logger.info('Updating model...')
update_model(grid_search_cv.best_estimator_)

logger.info('Generating model report')
validation_score = grid_search_cv.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search_cv.best_estimator_)

y_test_pred = grid_search_cv.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')
