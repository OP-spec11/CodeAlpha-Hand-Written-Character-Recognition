import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist_train.csv')


def explore_data():
    df = pd.read_csv(DATA_PATH)
    print('Dataset shape:', df.shape)
    print('Label distribution:')
    print(df['label'].value_counts().sort_index())

    sample = df.iloc[0, 1:].values.reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.title('Sample Digit')
    plt.axis('off')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'sample_digits.png'))
    plt.close()


if __name__ == '__main__':
    explore_data()
