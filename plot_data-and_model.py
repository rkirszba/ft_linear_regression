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
        variance = info['variance']
        
        min_value =data['km'].min()
        max_value = data['km'].max()
        x1 = add_intercept(normalize(np.array([[min_value]]), mean, variance))
        x2 = add_intercept(normalize(np.array([[max_value]]), mean, variance))
        y1 = np.squeeze(predict(x1, thetas))
        y2 = np.squeeze(predict(x2, thetas))


        plt.plot([min_value, max_value], [y1, y2])
        plt.scatter(list(data['km']), list(data['price']))
        plt.ylabel('Price')
        plt.xlabel('Km')
        plt.show()

    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)