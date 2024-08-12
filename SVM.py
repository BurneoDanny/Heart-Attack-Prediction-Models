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

## LIMPIEZA O FILTRADO DE DATOS
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
            SVM = SVC(kernel=kernel, C=C, gamma=gamma, random_state=0, max_iter=10000)
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
SVM = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, random_state=0)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

# Evaluación final
SVM_score = accuracy_score(y_test, y_pred)
SVM_score
