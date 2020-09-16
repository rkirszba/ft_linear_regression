import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ft_linear_regression import FTLinearRegression

if __name__ == '__main__':
    try:
        data = pd.read_csv('data.csv')

        f = open('my_model.pkl', 'rb')
        my_model = pickle.load(f)
        
        f = open('sk_model.pkl', 'rb')
        sk_model = pickle.load(f)

        min_value = data['km'].min()
        max_value = data['km'].max()
    
        y1 = np.squeeze(my_model.predict(np.array([[min_value]])))
        y2 = np.squeeze(my_model.predict(np.array([[max_value]])))

        y1_sk = np.squeeze(sk_model.predict(np.array([[min_value]])))
        y2_sk = np.squeeze(sk_model.predict(np.array([[max_value]])))

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