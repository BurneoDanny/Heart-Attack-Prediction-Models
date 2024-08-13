import pickle
import os
import tensorflow as tf
from evaluation_metrics import evaluate_model

load_dir_data = 'data/processed_data'
load_dir_models = 'models'

with open(os.path.join(load_dir_data, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(load_dir_data, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)
def load_model(model_name):
    if model_name != 'mlp_model.best':
        model_path = f'models/{model_name}.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded model from: {model_path}, type: {type(model)}")
        return model
    else:
        print("BRO")
        model_path = f'models/{model_name}.keras'
        model = tf.keras.models.load_model(model_path)
        return model

logistic_regression_model = load_model('logistic_regression_model')
mlp_model = load_model('mlp_model.best')
svm_model = load_model('svm_model')
gbm_model = load_model('xgbm_model')
random_forest_model = load_model('random_forest_model')
decision_tree_model = load_model('decision_tree_model')


y_pred_mlp = mlp_model.predict(X_test)
y_pred_mlp = (y_pred_mlp > 0.5).astype(int).flatten()
y_pred_gbm = gbm_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_dt = decision_tree_model.predict(X_test)
y_pred_lr = logistic_regression_model.predict(X_test)


evaluate_model(y_test, y_pred_mlp, "MLP Network Model")
evaluate_model(y_test, y_pred_lr, "Logistic Regression Model")
evaluate_model(y_test, y_pred_svm, "Supporting Vector Machines Model")
evaluate_model(y_test, y_pred_gbm, "Gradient Boosting Machine Model")
evaluate_model(y_test, y_pred_rf, "Random Forests Model")
evaluate_model(y_test, y_pred_dt, "Decision Tree Model")
