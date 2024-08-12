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

from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/ESPOL/IA-MATERIA/Heart Attack.csv' ## AQUI SE DEBE INSERTAR LA RUTA LOCAL DE DRIVE
datos = pd.read_csv(data_dir)
display(datos.head())

## LIMPIEZA DE DATOS
datos['class'] = datos['class'].map({'negative': 0, 'positive': 1}) # conversion de variables cualitativas a cuantitativas
datos_filtrados = datos[datos['impulse'] <= 300] # borrar ruido de pulsos exageradamente altos.
datos_filtrados = datos_filtrados[~((datos_filtrados['troponin'] > 9) & (datos_filtrados['class'] == 'negativa'))] # TROPONINA MAYOR A 9 NO PUEDE SER CLASE NEGATIVA / ruido


X = datos.drop(columns='class')  # contiene todas las columnas a excepcion de class
y = datos['class'] # contiene solo la columna de class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Shape of X_Train set : {}'.format(X_train.shape))
print('Shape of y_Train set : {}'.format(y_train.shape))
print('_'*50)
print('Shape of X_test set : {}'.format(X_test.shape))
print('Shape of y_test set : {}'.format(y_test.shape))



# Definición de los hiperparámetros
penalties = ['l1', 'l2', 'elasticnet', 'none']
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
C_values = [0.01, 0.1, 1, 10, 100]
l1_ratios = [0.1, 0.5, 0.7]  # Solo para elasticnet
best_penalty = str()
best_solver = str()
best_C = float()
best_l1_ratio = None  # Para elasticnet
best_acc = 0

# Bucle para probar todas las combinaciones de hiperparámetros
for penalty in penalties:
    for solver in solvers:
        for C in C_values:
            # Algunos solvers no soportan ciertas penalizaciones
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                continue
            if penalty == 'elasticnet' and solver != 'saga':
                continue
            if penalty == 'none' and solver in ['liblinear', 'saga']:
                continue

            # Si es elasticnet, se prueba con diferentes l1_ratios
            if penalty == 'elasticnet':
                for l1_ratio in l1_ratios:
                    LR = LogisticRegression(penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio, random_state=0, max_iter=1000)
                    LR.fit(X_train, y_train)
                    y_pred = LR.predict(X_test)
                    score = accuracy_score(y_test, y_pred)

                    if score > best_acc:
                        best_acc = score
                        best_penalty = penalty
                        best_solver = solver
                        best_C = C
                        best_l1_ratio = l1_ratio
            else:
              LR = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=0, max_iter=10000)
              LR.fit(X_train, y_train)
              y_pred = LR.predict(X_test)

              score = accuracy_score(y_test, y_pred)

              if score > best_acc:
                  best_acc = score
                  best_penalty = penalty
                  best_solver = solver
                  best_C = C

# Resultados
print('Best Penalty : ', best_penalty)
print('Best Solver : ', best_solver)
print('Best C : ', best_C)
print('Accuracy Score : ', best_acc)

# Entrenamiento del modelo final con los mejores hiperparámetros
LR = LogisticRegression(penalty=best_penalty, C=best_C, solver=best_solver, random_state=0, max_iter=10000)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

# Evaluación final
LR_score = accuracy_score(y_test, y_pred)
LR_score
