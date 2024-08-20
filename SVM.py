import pandas as pd
from sklearn.model_selection import train_test_split
# Para regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Para SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
from evaluation_metrics import evaluate_model

load_dir = 'data/processed_data'
save_dir = 'models'
# Cargar los datos preprocesados
with open(os.path.join(load_dir, 'X_train.pkl'), 'rb') as f:
    X_train = pickle.load(f)

with open(os.path.join(load_dir, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(load_dir, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)

with open(os.path.join(load_dir, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

print('Shape of X_Train set : {}'.format(X_train.shape))
print('Shape of y_Train set : {}'.format(y_train.shape))
print('_'*50)
print('Shape of X_test set : {}'.format(X_test.shape))
print('Shape of y_test set : {}'.format(y_test.shape))



# Definición de los hiperparámetros
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.01, 0.1, 1, 10, 100]
gammas = ['scale', 'auto']  # 'scale' es 1 / (n_features * X.var()), 'auto' es 1 / n_features

best_kernel = str()
best_C = float()
best_gamma = str()
best_acc = 0

# Bucle para probar todas las combinaciones de hiperparámetros
for kernel in kernels:
    for C in C_values:
        for gamma in gammas:
            # Entrenamiento del modelo SVM
            SVM = SVC(kernel=kernel, C=C, gamma=gamma, random_state=0, max_iter=1000000)
            SVM.fit(X_train, y_train)
            y_pred = SVM.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > best_acc:
                best_acc = score
                best_kernel = kernel
                best_C = C
                best_gamma = gamma

# Resultados
print('Best Kernel : ', best_kernel)
print('Best C : ', best_C)
print('Best Gamma : ', best_gamma)
print('Accuracy Score : ', best_acc)

# Entrenar el modelo final con los mejores hiperparámetros
SVM = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, random_state=0, probability=True)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

# Evaluación final
evaluate_model(y_test, y_pred, "SVM")

#Guardar Modelo
with open(os.path.join(save_dir, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(SVM, f)
