import os
import pickle
import xgboost as xgb
from evaluation_metrics import evaluate_model  # Importar la función de evaluación


# Especifica el directorio donde se guardaron los archivos preprocesados
load_dir = 'data/processed_data'
save_dir = 'models'
# Cargar los datos preprocesados desde el directorio especificado
with open(os.path.join(load_dir, 'X_train.pkl'), 'rb') as f:
    X_train = pickle.load(f)

with open(os.path.join(load_dir, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(load_dir, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)

with open(os.path.join(load_dir, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

params = {
    'objective': 'binary:logistic',  # Clasificación binaria
    'eval_metric': 'logloss',        # Función de pérdida (log loss)
    'max_depth': 6,                  # Profundidad máxima de los árboles
    'learning_rate': 0.1,            # Tasa de aprendizaje
    'n_estimators': 100,             # Número de árboles (estimadores)
    'subsample': 0.8,                # Submuestreo de datos
    'colsample_bytree': 0.8,         # Submuestreo de características por árbol
    'alpha': 0,                      # L1 regularization term on weights
    'lambda': 1,                     # L2 regularization term on weights
    'gamma': 0,                      # Mínima ganancia requerida para hacer una partición adicional
    'scale_pos_weight': 1,           # Manejo de clases desbalanceadas (1 para balanceado)
    'seed': 42                       # Fijar la semilla para reproducibilidad
}

# Entrenar el modelo utilizando los datos de entrenamiento
xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred_xgb = xgb_clf.predict(X_test)

print(X_test)
print(y_pred_xgb)
print(y_test)
# Evaluar el modelo XGBoost
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Guardar el modelo entrenado
with open(os.path.join(save_dir, 'xgbm_model.pkl'), 'wb') as f:
    pickle.dump(xgb_clf, f)