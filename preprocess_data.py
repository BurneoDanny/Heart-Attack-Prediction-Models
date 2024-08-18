import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

data_dir = 'data/Heart_Attack.csv'
data = pd.read_csv(data_dir)
save_dir = 'data/processed_data'

#Verificar valores nulos o datos incompletos, dar un resumen de los datos

'''
print(data.isnull().sum())
print(data.isna().sum())
print(data.dropna())
print(data.dtypes)
print(data.describe())
'''

#IMPLUSE exageradamente alto
valor_maximo_imp = data['impluse'].max()
valor_minimo_imp = data['impluse'].min()
print(valor_minimo_imp)
print(valor_maximo_imp)
filtered_data = data[data['impluse'] <= 260]
print(filtered_data.shape)

#PRESSURE HIGH
valor_maximo_prh = data['pressurehight'].max()
print(valor_maximo_prh)
valor_minimo_prh = data['pressurehight'].min()
print(valor_minimo_prh)
filtered_data = filtered_data[~((filtered_data['pressurehight'] < 50) & (filtered_data['class'] == 'negative'))]
print(filtered_data.shape)

#PRESSURE LOW
valor_maximo_prl = data['pressurelow'].max()
valor_minimo_prl = data['pressurelow'].min()
print(valor_maximo_prl)
print(valor_minimo_prl)

#GLUCOSE no parece tener ruido
valor_maximo_glu = data['glucose'].max()
valor_minimo_glu = data['glucose'].min()
print(valor_maximo_glu)
print(valor_minimo_glu)

#KCM no parece tener ruido
valor_maximo_kcm = data['kcm'].max()
valor_minimo_kcm = data['kcm'].min()
print(valor_maximo_kcm)
print(valor_minimo_kcm)

#TROPONINA ALTA sin ser ataque cardiaco
valor_maximo_tro = data['troponin'].max()
valor_minimo_tro = data['troponin'].min()
print(valor_maximo_tro)
print(valor_minimo_tro)
filtered_data = filtered_data[~((filtered_data['troponin'] > 9) & (filtered_data['class'] == 'negative'))]
print(filtered_data.shape)




# Convertir la variable 'class' en una variable numérica
label_encoder = LabelEncoder()
filtered_data['class'] = label_encoder.fit_transform(filtered_data['class'])  # 0 para 'negative', 1 para 'positive'

# Separar las características (X) de la etiqueta (y)
X = filtered_data.drop('class', axis=1).values
y = filtered_data['class'].values

# Extraer 4 registros aleatorios de cada clase para predicción
negative_sample = filtered_data[filtered_data['class'] == 0].sample(4, random_state=42)
positive_sample = filtered_data[filtered_data['class'] == 1].sample(4, random_state=42)

# Combinar las muestras seleccionadas
selected_samples = pd.concat([negative_sample, positive_sample])
print(selected_samples)
selected_samples.to_csv('data/prediction_data/prediction_samples.csv', index=False)
print("Los datos seleccionados han sido guardados en 'prediction_samples.csv'.")

# Eliminar los registros seleccionados del dataset original
filtered_data = filtered_data.drop(selected_samples.index)
print(filtered_data.shape)

#Técnica SMOTE para balancear las clases de los datos
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(X_resampled)
print(y_resampled)

# Convertir selected_samples a solo las características para comparar con X_resampled
selected_X = selected_samples.drop('class', axis=1).values
selected_y = selected_samples['class'].values

# Verificar y eliminar registros coincidentes en X_resampled y y_resampled
indices_to_remove = []

for i in range(len(selected_X)):
    for j in range(len(X_resampled)):
        if all(selected_X[i] == X_resampled[j]) and selected_y[i] == y_resampled[j]:
            indices_to_remove.append(j)

# Eliminar los índices coincidentes
if indices_to_remove:
    X_resampled = np.delete(X_resampled, indices_to_remove, axis=0)
    y_resampled = np.delete(y_resampled, indices_to_remove, axis=0)
    print(f"Se eliminaron {len(indices_to_remove)} registros que coincidían con los seleccionados para predicción.")
else:
    print("Ninguno de los registros seleccionados se encuentra en el dataset después de Smote.")

# Dividir los datos en conjuntos de entrenamiento, prueba y predicción
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


print("Datos para entrenamiento con %d ejemplos" % len(X_train))
print("Datos para prueba con %d ejemplos" % len(X_test))
print("Datos para predicción con %d ejemplos" % len(selected_X))



# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_predict = scaler.transform(X_predict)

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

