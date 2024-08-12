import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
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
from evaluation_metrics import evaluate_model


load_dir = 'data/processed_data'
# Cargar los datos preprocesados
with open(os.path.join(load_dir, 'X_train.pkl'), 'rb') as f:
    X_train = pickle.load(f)

with open(os.path.join(load_dir, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(load_dir, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)

with open(os.path.join(load_dir, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

# Crear el modelo MLP
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))  # Capa oculta con He initialization
model.add(Dense(8, activation='relu' ))# Otra capa oculta con He initialization
model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='models/mlp1.model.best.hdf5.keras', verbose=1, save_best_only=True)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split = 0.2, callbacks=[checkpointer],
         verbose=1, shuffle=True )

# Evaluar el modelo
#loss, accuracy = model.evaluate(X_test, y_test)
#print(f'Accuracy: {accuracy * 100:.2f}%')

y_pred_mlp = model.predict(X_test)
y_pred_mlp = (y_pred_mlp > 0.5).astype(int).flatten()

# Evaluar el MLP
evaluate_model(y_test, y_pred_mlp, "MLP (Keras)")