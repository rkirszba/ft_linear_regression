import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    try:
        data = pd.read_csv('data.csv')
        if data.shape[0] < 1 or data.shape[1] != 2:
            raise Exception
        m = data.shape[0]
        X = np.array(data['km'], dtype=float).reshape((m, 1))
        y = np.array(data['price'], dtype=float).reshape((m, 1))
        print(X)
        print(y)
        model = LinearRegression()
        model.fit(X, y)
    except FileNotFoundError:
        print('There is no data.csv file')
        sys.exit(1)
    except Exception as e:
        print('data.csv is not well formated')
        sys.exit(1)
    model_file = open('sk_model.pkl', 'wb')
    pickle.dump(model, model_file)