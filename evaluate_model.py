import os
import pickle

import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist_test.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mnist_cnn_model.pkl')


def evaluate_model():
    test_df = pd.read_csv(DATA_PATH)
    X_test = test_df.drop(columns=['label']).values.astype('float32') / 255.0
    y_test = test_df['label'].values.astype('int32')

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    accuracy = model.score(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    evaluate_model()
