import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

data_dir = 'data/Heart_Attack.csv'
data = pd.read_csv(data_dir)
save_dir = 'data/processed_data'

#Verificar valores nulos o datos incompletos, dar un resumen de los datos

print(data.isnull().sum())
print(data.isna().sum())
print(data.dropna())
print(data.dtypes)
print(data.describe())



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

# Guardar los datos preprocesados en archivos .pkl

with open(os.path.join(save_dir, 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)

with open(os.path.join(save_dir, 'X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(save_dir, 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)

with open(os.path.join(save_dir, 'y_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)

# Guardar el scaler ajustado
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)