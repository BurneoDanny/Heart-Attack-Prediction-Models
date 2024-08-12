import os
import pickle
from sklearn.preprocessing import StandardScaler

load_dir = 'data/processed_data'


def preprocess_input(features):
    with open(os.path.join(load_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    input = scaler.transform(features)
    return input




