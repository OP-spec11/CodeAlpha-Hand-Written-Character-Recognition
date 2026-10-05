import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist_train.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mnist_cnn_model.h5')


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['label']).values.astype('float32') / 255.0
    y = df['label'].values.astype('int32')
    return X, y


def train_model():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, random_state=42)
    model.fit(X_train, y_train)

    val_accuracy = model.score(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    with open(MODEL_PATH.replace('.h5', '.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved to: {MODEL_PATH.replace(".h5", ".pkl")}')


if __name__ == '__main__':
    train_model()
