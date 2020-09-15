import sys
import pickle
import numpy as np
from utils import estimate_price, add_intercept

if __name__ == '__main__':

    while True:
        user_input = input('Please give the mileage of the car\n')
        try:
            mileage = float(user_input)
            if mileage < 0:
                raise ValueError()
            break
        except ValueError:
            print('\'{}\' is not a correct input'.format(user_input))
            continue
        except KeyboardInterrupt:
            sys.exit(0)

    try:
        f = open('thetas.pkl', 'rb')
        thetas = pickle.load(f)
        if type(thetas) is not np.ndarray or thetas.shape != (2, 1):
            raise Exception
    except:
        thetas = np.zeros((1, 2))

    X = add_intercept(np.array([[mileage]]))
    estimated_price = np.squeeze(estimate_price(X, thetas))
    print('The estimated price for a car with mileage {} is:\n{}'.format(mileage, estimated_price))