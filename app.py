from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend

# Función para cargar el modelo según el nombre del archivo
def load_model(model_name):
    model_path = f'models/{model_name}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model')

    # Verificar que el modelo fue seleccionado correctamente
    if model_type is None:
        return jsonify({"error": "No model type provided"}), 400

    try:
        if model_type == 'mlp':
            model = load_model('mlp_model')
        elif model_type == 'logistic_regression':
            model = load_model('logistic_regression_model')
        elif model_type == 'decision_tree':
            model = load_model('decision_tree_model')
        elif model_type == 'random_forest':
            model = load_model('random_forest_model')
        elif model_type == 'svm':
            model = load_model('svm_model')
        elif model_type == 'gbm':
            model = load_model('gbm_model')
        else:
            return jsonify({"error": "Invalid model type"}), 400
    except FileNotFoundError:
        return jsonify({"error": f"Model '{model_type}' not found"}), 500

    try:
        features = np.array([[data['age'], data['gender'], data['impluse'], data['pressureHight'], 
                              data['pressureLow'], data['glucose'], data['kcm'], data['troponin']]])
        if None in features:
            raise ValueError("Some input features are None")
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        prediction = model.predict(features)[0]
    
        # Calcular la probabilidad de infarto (clase positiva)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1] * 100  # Clase positiva es la segunda columna
            print("Probability:", probability)
        else:
            probability = None
            print("Probability is None")
        
        result = "Positive (Risk of Heart Attack)" if prediction == 1 else "Negative (No Risk of Heart Attack)"
        
        response = {"prediction": result}
        if probability is not None:
            response["percentage"] = f"{probability:.2f}%"
        
        print("Response:", response)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
