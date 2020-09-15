import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import add_intercept, normalize, predict

if __name__ == '__main__':
    try:
        data = pd.read_csv('data.csv')

        f = open('learning_info.pkl', 'rb')
        info = pickle.load(f)
        thetas = info['thetas']
        mean = info['mean']
        sigma = info['sigma']
        
        f = open('sk_model.pkl', 'rb')
        model = pickle.load(f)

        min_value = data['km'].min()
        max_value = data['km'].max()
        x1 = add_intercept(normalize(np.array([[min_value]]), mean, sigma))
        x2 = add_intercept(normalize(np.array([[max_value]]), mean, sigma))

        y1 = np.squeeze(predict(x1, thetas))
        y2 = np.squeeze(predict(x2, thetas))

        y1_sk = np.squeeze(model.predict(np.array([[min_value]])))
        y2_sk = np.squeeze(model.predict(np.array([[max_value]])))

        plt.plot([min_value, max_value], [y1, y2], label='My model', color='g')
        plt.plot([min_value, max_value], [y1_sk, y2_sk], label='SKlearn model', color='orange')
        plt.scatter(list(data['km']), list(data['price']))
        plt.ylabel('Price')
        plt.xlabel('Km')
        plt.legend()
        plt.show()

    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)