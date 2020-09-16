import sys
import pandas as pd
import numpy as np
import pickle
from ft_linear_regression import FTLinearRegression


if __name__ == '__main__':
    try:
        data = pd.read_csv('data.csv')
        if data.shape[0] < 1 or data.shape[1] != 2:
            raise Exception
        m = data.shape[0]
        X = np.array(data['km'], dtype=float).reshape((m, 1))
        y = np.array(data['price'], dtype=float).reshape((m, 1))
    except FileNotFoundError:
        print('There is no data.csv file')
        sys.exit(1)
    except PermissionError:
        print('Failed to open data.csv')
        sys.exit(1)
    except:
        print('data.csv is not well formated')
        sys.exit(1)

    try:
        model = FTLinearRegression().fit(X, y)
        model_file = open('my_model.pkl', 'wb')
        pickle.dump(model, model_file)
    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)