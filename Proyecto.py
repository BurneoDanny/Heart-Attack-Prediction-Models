import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal, HeUniform
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report

#normalizar los datos
#80-20 entranamiento y prueba, rendimiento del modelo, es diferente al error del entrenamiento
# validacion: precision en entrenamiento
# inicializacion con he, relu con fugas
#visualizar el comportamiento del error versus epocas, y visualizar si converge
#como hacer que pare automatico en las epocas cuando converja

data_dir = 'C:/Users/Ricardo/Desktop/Espol/S9/IA/Proyecto/Heart_Attack.csv'
data = pd.read_csv(data_dir)

#Verificar valores nulos o datos incompletos, dar un resumen de los datos
'''
print(data.isnull().sum())
print(data.isna().sum())
print(data.dropna())
print(data.dtypes)
print(data.describe())
'''


# Convertir la variable 'class' en una variable numérica
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])  # 0 para 'negative', 1 para 'positive'

# Separar las características (X) de la etiqueta (y)
X = data.drop('class', axis=1).values
y = data['class'].values

# Dividir los datos en conjuntos de entrenamiento, prueba y predicción
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_predict = X_test[:10]
y_true = y_test[:10]

X_test = X_test[10:]
y_test = y_test[10:]

print("Datos para entrenamiento con %d ejemplos" % len(X_train))
print("Datos para prueba con %d ejemplos" % len(X_test))
print("Datos para predicción con %d ejemplos" % len(X_predict))

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_predict = scaler.transform(X_predict)

# Crear el modelo MLP
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu', kernel_initializer=HeNormal()))  # Capa oculta con He initialization
model.add(Dense(8, activation='relu', kernel_initializer=HeNormal()))# Otra capa oculta con He initialization
model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5.keras', verbose=1, save_best_only=True)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split = 0.2, callbacks=[checkpointer],
          verbose=1, shuffle=True )

model.load_weights('mnist.model.best.hdf5.keras')
# Evaluar el modelo
#loss, accuracy = model.evaluate(X_test, y_test)
#print(f'Accuracy: {accuracy * 100:.2f}%')

y_pred_mlp = model.predict(X_test)
y_pred_mlp = (y_pred_mlp > 0.5).astype(int).flatten()

# Evaluar el modelo MLP
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Evaluar el MLP
evaluate_model(y_test, y_pred_mlp, "MLP (Keras)")

# Convertir las predicciones a 0 o 1 basadas en un umbral de 0.5
#y_pred_class = (y_pred > 0.5).astype(int).flatten()

# Mostrar las predicciones y las etiquetas reales
#for i in range(10):
#    print(f'Registro {i+1}: Predicción = {y_pred_class[i]}, Real = {y_true[i]}')


print("-------------------------------------GBM-------------------------------------")
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entrenar el modelo
gbm_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = gbm_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Mostrar un reporte de clasificación detallado
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# Crear el modelo XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Entrenar el modelo
xgb_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_xgb = xgb_model.predict(X_test)

# Evaluar el modelo XGBoost
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Evaluar el modelo XGBoost
evaluate_model(y_test, y_pred_xgb, "XGBoost")



