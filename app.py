from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
import math
from preprocess_inputs import preprocess_input
app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend

# Función para cargar el modelo según el nombre del archivo
def load_model(model_name):
    if model_name != 'mlp_model.best':
        model_path = f'models/{model_name}.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded model from: {model_path}, type: {type(model)}")  
        return model
    else:
        model_path = f'models/{model_name}.keras'
        model = tf.keras.models.load_model(model_path)
        return model


def truncar_dos_decimales(numero):
    factor = 10 ** 2
    return math.floor(numero * factor) / factor
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model')

    # Verificar que el modelo fue seleccionado correctamente
    if model_type is None:
        return jsonify({"error": "No model type provided"}), 400

    try:
        if model_type == 'mlp':
            model = load_model('mlp_model.best')
        elif model_type == 'logistic_regression':
            model = load_model('logistic_regression_model')
        elif model_type == 'decision_tree':
            model = load_model('decision_tree_model')
        elif model_type == 'random_forest':
            model = load_model('random_forest_model')
        elif model_type == 'svm':
            model = load_model('svm_model')
        elif model_type == 'gbm':
            model = load_model('xgbm_model')
        else:
            return jsonify({"error": "Invalid model type"}), 400
    except FileNotFoundError:
        return jsonify({"error": f"Model '{model_type}' not found"}), 500

    try:
        features = np.array([[data['age'], data['gender'], data['impluse'], data['pressureHight'], 
                              data['pressureLow'], data['glucose'], data['kcm'], data['troponin']]])
        print(features)
        if None in features:
            raise ValueError("Some input features are None")
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        if model_type == 'mlp':
            processed_input = preprocess_input(features)
            prediction = model.predict(processed_input)[0]

        else:
            print(f"Model type: {type(model)}")
            processed_input = preprocess_input(features)
            prediction = model.predict(processed_input)[0]

        # Calcular la probabilidad de infarto (clase positiva)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1] * 100 # Clase positiva es la segunda columna
            print("Probability:", probability)
        elif model_type == 'mlp':
            if prediction > 0.5:
                probability = prediction[0] * 100
                prediction = 1
            else:
                probability = (prediction[0] * -100) + 100
                prediction = 0
        else:
            probability = None
            print("Probability is None")
        
        print("Prediction:", prediction)
        result = "Positive (Risk of Heart Attack)" if prediction == 1 else "Negative (No Risk of Heart Attack)"
        
        response = {"prediction": result}
        if probability is not None:
            response["percentage"] = f"{truncar_dos_decimales(probability):.2f}%"
        
        print("Response:", response)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def convert_types(data):
    for key, value in data.items():
        # Convertir a int si el valor puede ser convertido y el nombre del campo está en la lista de enteros
        if key in ["age", "impluse", "pressureHight", "pressureLow", "gender"]:
            try:
                data[key] = int(value)
            except ValueError:
                pass  # Maneja el caso si el valor no puede ser convertido
        # Convertir a float si el valor puede ser convertido y el nombre del campo está en la lista de flotantes
        elif key in ["glucose", "kcm", "troponin"]:
            try:
                data[key] = float(value)
            except ValueError:
                pass  # Maneja el caso si el valor no puede ser convertido
    return data

@app.route('/batchpredict', methods=['POST'])
def batchpredict():
    data = request.json
    print(data)
    convert_types(data)
    model_type = data.get('model')
    print(data["age"])
    print(model_type)
    # Verificar que el modelo fue seleccionado correctamente
    if model_type is None:
        return jsonify({"error": "No model type provided"}), 400

    try:
        if model_type == 'mlp':
            model = load_model('mlp_model.best')
        elif model_type == 'logistic_regression':
            model = load_model('logistic_regression_model')
        elif model_type == 'decision_tree':
            model = load_model('decision_tree_model')
        elif model_type == 'random_forest':
            model = load_model('random_forest_model')
        elif model_type == 'svm':
            model = load_model('svm_model')
        elif model_type == 'gbm':
            model = load_model('xgbm_model')
        else:
            return jsonify({"error": "Invalid model type"}), 400
    except FileNotFoundError:
        return jsonify({"error": f"Model '{model_type}' not found"}), 500

    try:
        features = np.array([[int(data['age']), int(data['gender']), int(data['impluse']), int(data['pressureHight']),
                              int( data['pressureLow']), float(data['glucose']), float(data['kcm']), float(data['troponin'])]])
        print(features)
        if None in features:
            raise ValueError("Some input features are None")
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        if model_type == 'mlp':
            processed_input = preprocess_input(features)
            prediction = model.predict(processed_input)[0]

        else:
            print(f"Model type: {type(model)}")
            processed_input = preprocess_input(features)
            prediction = model.predict(processed_input)[0]

        # Calcular la probabilidad de infarto (clase positiva)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1] * 100  # Clase positiva es la segunda columna
            print("Probability:", probability)
        elif model_type == 'mlp':
            if prediction > 0.5:
                probability = prediction[0] * 100
                prediction = 1
            else:
                probability = (prediction[0] * -100) + 100
                prediction = 0
        else:
            probability = None
            print("Probability is None")

        print("Prediction:", prediction)
        result = "Positive (Risk of Heart Attack)" if prediction == 1 else "Negative (No Risk of Heart Attack)"

        response = {"prediction": result}
        response["id"] = data['id']
        if probability is not None:
            response["percentage"] = f"{truncar_dos_decimales(probability):.2f}%"

        print("Response:", response)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/')
def index():
    return "If you can see this message, backend is running! and you can start making predictions in the frontend."


if __name__ == '__main__':
    app.run(debug=True)
