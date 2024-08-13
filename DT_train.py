import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from DecisionTree import DecisionTree
import pickle
import os
from evaluation_metrics import evaluate_model

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


# Definir la función de precisión
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Validación cruzada con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1234)
accuracies = []

for train_index, test_index in kf.split(X_train):
    X_train_v, X_test_v = X_train[train_index], X_train[test_index]
    y_train_v, y_test_v = y_train[train_index], y_train[test_index]

    # Crear y entrenar el modelo
    clf = DecisionTree(max_depth=10, min_samples_split=2, n_features=None)
    clf.fit(X_train_v, y_train_v)

    # Predecir con el modelo entrenado
    predictions = clf.predict(X_test_v)

    # Calcular y guardar la precisión
    acc = accuracy(y_test_v, predictions)
    accuracies.append(acc)

# Mostrar la precisión media y desviación estándar
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# Entrenar el modelo final en todos los datos disponibles
clf.fit(X_train, y_train)

#Realizar predicciones con los datos de testing para evaluar el modelo
predictions_final = clf.predict(X_test)

#Evaluar modelo
evaluate_model(y_test, predictions_final, "Decision Tree Model")

# Guardar el modelo entrenado
with open(os.path.join(save_dir, 'decision_tree_model.pkl'), 'wb') as f:
    pickle.dump(clf, f)

print("Modelo guardado exitosamente en 'decision_tree_model.pkl'")