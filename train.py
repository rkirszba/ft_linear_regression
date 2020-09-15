import sys
import pandas as pd
import numpy as np
import pickle
from utils import add_intercept, mean, variance, normalize, fit


if __name__ == '__main__':
    try:
        data = pd.read_csv('data.csv')
        if data.shape[0] < 1 or data.shape[1] != 2:
            raise Exception
        m = data.shape[0]
        X = np.array(data['km'], dtype=float).reshape((m, 1))
        mean = mean(X)
        variance = variance(X)
        X = normalize(X, mean, variance)
        X = add_intercept(X)
        y = np.array(data['price'], dtype=float).reshape((m, 1))
    except FileNotFoundError:
        print('There is no data.csv file')
        sys.exit(1)
    except:
        print('data.csv is not well formated')
        sys.exit(1)


    thetas = np.zeros((1, 2))
    thetas, costs = fit(X, y, thetas)
    info = {
        'thetas': thetas,
        'costs': costs,
        'mean': mean,
        'variance': variance
    }
    print('Final thetas')
    print(thetas)
    info_file = open('learning_info.pkl', 'wb')
    pickle.dump(info, info_file)