import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ft_linear_regression import FTLinearRegression
from metrics import *

from sklearn import metrics

if __name__ == '__main__':

    try:
        data = pd.read_csv('data.csv')
        m = data.shape[0]
        X = np.array(data['km'], dtype=float).reshape((m, 1))
        y = np.array(data['price'], dtype=float).reshape((m, 1))

        f = open('my_model.pkl', 'rb')
        my_model = pickle.load(f)
        
        f = open('sk_model.pkl', 'rb')
        sk_model = pickle.load(f)

        y_mymodel = my_model.predict(X)
        y_skmodel = sk_model.predict(X)

        print('For my model:\n')
        print('Mean Squared Error = {}'.format(mean_squared_error(y, y_mymodel)))
        print('Root Mean Squared Error = {}'.format(root_mean_squared_error(y, y_mymodel)))
        print('Mean Absolute Error = {}'.format(mean_absolute_error(y, y_mymodel)))
        print('R2 Score = {}'.format(r2_score(y, y_mymodel)))

        print('\n')

        print('For Scikit Learn model:\n')
        print('Mean Squared Error = {}'.format(mean_squared_error(y, y_skmodel)))
        print('Root Mean Squared Error = {}'.format(root_mean_squared_error(y, y_skmodel)))
        print('Mean Absolute Error = {}'.format(mean_absolute_error(y, y_skmodel)))
        print('R2 Score = {}'.format(r2_score(y, y_skmodel)))
    
    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)
