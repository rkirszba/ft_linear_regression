import sys
import pickle
import matplotlib.pyplot as plt
from ft_linear_regression import FTLinearRegression

if __name__ == '__main__':
    try:
        f = open('my_model.pkl', 'rb')
        model = pickle.load(f)
        model.plot_learning_curve()

    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)