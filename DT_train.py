import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from DecisionTree import DecisionTree
import pickle

# Cargar el dataset
data = pd.read_csv('Heart Attack.csv')
X = data.drop(columns=['class']).values  # Convertir a matriz NumPy
y = data['class']

# Codificar la variable de clase (target)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 'positive' será 1 y 'negative' será 0

# Definir la función de precisión
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Validación cruzada con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1234)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Crear y entrenar el modelo
    clf = DecisionTree(max_depth=10, min_samples_split=2, n_features=None)
    clf.fit(X_train, y_train)

    # Predecir con el modelo entrenado
    predictions = clf.predict(X_test)

    # Calcular y guardar la precisión
    acc = accuracy(y_test, predictions)
    accuracies.append(acc)

# Mostrar la precisión media y desviación estándar
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# Entrenar el modelo final en todos los datos disponibles
clf.fit(X, y)

# Guardar el modelo entrenado
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo guardado exitosamente en 'decision_tree_model.pkl'")